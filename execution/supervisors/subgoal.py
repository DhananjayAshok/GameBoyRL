"""
Decompose the task into a plan, then drive the plan one step at a time.

:class:`SubgoalSupervisor` is :class:`~execution.supervisors.revising.RevisingSupervisor`
with a different answer to one question: what are the targets? It writes a plan and hands
the engine its steps.

What the extra targets bring with them:

- the judge fires at all — an intermediate step has no environment signal. The last step
  reverts to the task and is never judged.
- ``_check_regression`` — a step can undo one already cleared, which the per-step judge
  never sees.
- ``_diagnose_failure`` — repeated failure may mean the plan is wrong, in which case the
  remainder is replanned.

Repair is at plan level, not step level.
"""

from __future__ import annotations

from typing import List, Optional

from execution.supervisors.prompts import (PLAN_FLAW_PROMPT, PLAN_PROMPT,
                                           REGRESSION_CHECK_PROMPT)
from execution.supervisors.revising import RevisingSupervisor
from utils import log_warn, parse_key_value, parse_plan, parse_yes_no


class SubgoalSupervisor(RevisingSupervisor):
    """Plan first, then supervise the executor through the plan step by step.

    :param max_replans: How many times the plan may be rewritten in one episode.

    :ivar plan: The current plan. Mutated in place when the remainder is replaced.
    :ivar original_plan: The plan as first written, before any replan.
    :ivar planned: Whether an opening plan was produced at all.
    :ivar completed_steps: Steps already cleared, each with the frame that proved it.
    """

    def __init__(self, *args, max_replans: int = 2, **kwargs) -> None:
        self.max_replans = max_replans
        self.n_replans = 0
        self.plan: List[str] = []
        self.original_plan: List[str] = []
        self.planned = False
        self.completed_steps: List[dict] = []
        super().__init__(*args, **kwargs)

    def _run_config(self) -> dict:
        return {**super()._run_config(), "max_replans": self.max_replans}

    # -- Planning -------------------------------------------------------------

    def write_plan(self) -> List[str]:
        """Turn the task and the opening screen into an ordered plan, using
        :data:`PLAN_PROMPT` with whatever :meth:`_knowledge` returns.

        :return: The plan's steps, or ``[]`` when nothing could be planned.
        :rtype: List[str]
        """
        screen = self._current_frame()
        prompt = (
            PLAN_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
        )
        output = self._vlm_call("plan", texts=prompt, images=[screen])
        self.plan = parse_plan(output)
        return self.plan

    def _resolve_targets(self) -> List[Optional[str]]:
        """Write the plan, then hand its steps to the engine as targets. Returns a copy, not
        :attr:`plan` itself.

        :return: The plan's steps, or ``[None]`` when planning produced nothing.
        :rtype: List[Optional[str]]
        """
        self.completed_steps = []
        self.n_replans = 0
        self.write_plan()
        self.original_plan = list(self.plan)
        self.planned = bool(self.plan)
        if not self.plan:
            log_warn("[subgoal] no plan produced; running the task unplanned.",
                     self._parameters)
            return [None]
        return list(self.plan)

    # -- Judging and repair ---------------------------------------------------

    def _on_target_cleared(self, target: str) -> None:
        self.completed_steps.append({"step": target, "frame": self._current_frame()})

    def _check_regression(self, summaries: List[str]) -> Optional[str]:
        """Ask whether the failed attempt undid a step that had already been completed, by
        comparing the frame that proved the earlier step against the screen now. Runs only
        when a step has actually been completed.

        :return: A sentence naming what was lost, or ``None`` if nothing was.
        :rtype: Optional[str]
        """
        if not self.completed_steps:
            return None
        last = self.completed_steps[-1]
        earlier = self.completed_steps[:-1]
        earlier_block = ""
        if earlier:
            earlier_block = ("Steps completed before that one, which must also still hold:\n"
                             + "\n".join(f"- {s['step']}" for s in earlier) + "\n\n")

        prompt = (
            REGRESSION_CHECK_PROMPT
            .replace("[GAME]", self._game)
            .replace("[PREVIOUS_STEP]", last["step"])
            .replace("[EARLIER_STEPS_BLOCK]", earlier_block)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries) or "  (no actions taken)")
        )
        output = self._vlm_call(
            "regression_check",
            texts=prompt, images=[last["frame"], self._current_frame()],
        )
        if parse_yes_no(output, "Undone") is not True:
            return None
        lost = (parse_key_value(output, "What was lost") or "").strip()
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if lost.upper().startswith("NONE") or not lost:
            lost = reasoning or "progress from an earlier step is no longer visible"
        note = (f"The attempt undid earlier progress: {lost} "
                f"(the step '{last['step']}' was completed and no longer holds).")
        self._say(f"      REGRESSION: {lost}")
        return note

    def _diagnose_failure(self, index: int, targets: List[Optional[str]],
                          history: List[str], judgement: str,
                          regression: Optional[str]) -> Optional[List[str]]:
        """Ask whether the failure is the plan's fault, and replan the remainder if so.

        Only the remaining steps are replanned; cleared steps are shown as DONE and excluded.
        :data:`PLAN_FLAW_PROMPT` answers *no* by default, so *history* must be the current
        step's own attempts and no more.

        :return: The replacement steps for ``index`` onwards, or ``None`` to carry on
            hinting the current target.
        :rtype: Optional[List[str]]
        """
        if self.n_replans >= self.max_replans:
            return None

        # An episode that never produced a plan has no plan to find flawed.
        if all(text is None for text in targets):
            return None

        plan_lines = []
        for i, text in enumerate(targets):
            if text is None:
                continue
            if i < index:
                marker = "  [DONE]"
            elif i == index:
                marker = "  <-- CURRENT, just failed"
            else:
                marker = "  [not yet attempted]"
            plan_lines.append(f"  {i + 1}. {text}{marker}")

        prompt = (
            PLAN_FLAW_PROMPT
            .replace("[GAME]", self._game)
            .replace("[OVERALL_TASK]", self._task)
            .replace("[PLAN_BLOCK]", "\n".join(plan_lines))
            .replace("[FAILURE_HISTORY]", "\n\n".join(f"  attempt {i + 1}: {h}"
                                                      for i, h in enumerate(history)))
            .replace("[JUDGEMENT]", judgement or "(not judged)")
            .replace("[REGRESSION_LINE]", f"\n{regression}\n" if regression else "")
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
        )
        output = self._vlm_call("plan_flaw", texts=prompt, images=[self._current_frame()])
        if parse_yes_no(output, "Flawed") is not True:
            return None

        replacement = parse_plan(output)
        if not replacement:
            log_warn("[subgoal] plan judged flawed but no replacement was produced; "
                     "continuing with the current step.", self._parameters)
            return None

        self.n_replans += 1
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        self._say(f"      REPLAN ({self.n_replans}/{self.max_replans}): {reasoning}")
        for text in replacement:
            self._say(f"        - {text}")
        return replacement

    def _on_targets_replaced(self, targets: List[Optional[str]]) -> None:
        """Track the spliced list, not the replacement tail."""
        self.plan = [t for t in targets if t is not None]

    def _extras(self) -> dict:
        return {**super()._extras(),
                "plan": list(self.plan),
                "original_plan": list(self.original_plan),
                "planned": self.planned,
                "n_replans": self.n_replans}
