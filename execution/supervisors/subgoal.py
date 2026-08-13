"""
Decompose the task into a plan, then drive the plan one step at a time.

:class:`SubgoalSupervisor` is :class:`~execution.supervisors.revising.RevisingSupervisor`
with a different answer to one question: what are the targets? Instead of the benchmark
task alone, it writes a plan and hands the engine its steps.

Everything else it adds exists because there is now more than one target:

- ``_judge_target`` — an intermediate step has no environment signal, so a VLM decides
  whether it landed. The last step still reverts to the task and is still the
  environment's to clear.
- ``_check_regression`` — a step can undo one already cleared, which the per-step judge
  cannot see because it is only ever asked about the *current* step.
- ``_diagnose_failure`` — repeated failure may mean the plan is wrong rather than the
  attempt, in which case the remainder is replanned.

Repair is at plan level, not step level. An earlier design rewrote the failing step in
place; that can only fix wording, which assumes the route is right and only the phrasing is
wrong. Most failures worth catching are not like that — the route does not exist, the
object named is not on this screen, the game does not work as assumed — and a better-worded
version of the same wrong step cannot fix any of them.
"""

from __future__ import annotations

from typing import List, Optional

from execution.supervisors.prompts import (PLAN_FLAW_PROMPT, PLAN_PROMPT,
                                           REGRESSION_CHECK_PROMPT)
from execution.supervisors.revising import RevisingSupervisor
from utils import log_warn, parse_key_value, parse_steps, parse_yes_no


class SubgoalSupervisor(RevisingSupervisor):
    """Plan first, then supervise the executor through the plan step by step.

    :param max_replans: How many times the plan may be rewritten in one episode. Bounds
        both cost and the risk of thrashing between two readings of the same screen.

    :ivar plan: The current plan. Mutated in place when the remainder is replaced.
    :ivar completed_steps: Steps already cleared, each with the frame that proved it, so a
        later step can be checked for having undone one of them.
    """

    def __init__(self, *args, max_replans: int = 2, **kwargs) -> None:
        self.max_replans = max_replans
        self.n_replans = 0
        self.plan: List[str] = []
        self.completed_steps: List[dict] = []
        super().__init__(*args, **kwargs)

    # -- Planning -------------------------------------------------------------

    def write_plan(self) -> List[str]:
        """Turn the task and the opening screen into an ordered plan.

        Uses :data:`PLAN_PROMPT` with whatever :meth:`_knowledge` returns — nothing at this
        level, a distilled insights block at the info level. The prompt constrains the
        planner to visible artefacts and forbids naming buttons, because a button sequence
        recorded in one playthrough is wrong from any other screen, and requires every step
        to end in a visually checkable condition — without one there is nothing for
        :meth:`_judge_target` to judge against. That discipline is worth having with or
        without knowledge, which is why there is no separate knowledge-free planner.

        Returns ``[]`` when nothing could be planned, which the engine degrades to running
        the task unplanned rather than inventing a plan.
        """
        screen = self._current_frame()
        prompt = (
            PLAN_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
        )
        output = self._vlm_call("plan", texts=prompt, images=[screen])
        raw = (parse_key_value(output, "Plan") or "").strip()
        self.plan = parse_steps(raw)
        return self.plan

    def _before_targets(self) -> List[Optional[str]]:
        """Write the plan, then hand its steps to the engine as targets.

        ``[None]`` when planning produced nothing: the arm degrades to the unplanned
        baseline rather than to a fabricated plan. A copy, not ``self.plan`` itself — the
        replan path splices into the working list in place, and aliasing would make the two
        names refer to the same object before the first replan and different objects after.
        """
        self.completed_steps = []
        self.n_replans = 0
        self.write_plan()
        if not self.plan:
            log_warn("[subgoal] no plan produced; running the task unplanned.",
                     self._parameters)
            return [None]
        return list(self.plan)

    def _targets(self) -> List[Optional[str]]:
        return list(self.plan) if self.plan else [None]

    # -- Judging and repair ---------------------------------------------------

    def _judge_target(self, target: str, env_steps: list, stop_reason: str):
        return self.judge_step(env_steps, target, stop_reason)

    def _on_target_cleared(self, target: str) -> None:
        # Remembered with the frame that proved it, so a later step undoing it can be
        # spotted by comparing that frame against the screen then.
        self.completed_steps.append({"step": target, "frame": self._current_frame()})

    def _check_regression(self, summaries: List[str]) -> Optional[str]:
        """Ask whether the failed attempt undid a step that had already been completed.

        The judge cannot see this. It is asked only whether *this* step's termination
        condition is met, so a run that quits the menu it had opened, or puts back the item
        it had taken, reads as a plain failure of the current step — and the hint that
        follows tells the player to keep pushing forward from a state that has silently
        moved backwards.

        The comparison is between two frames: the screen when the earlier step was judged
        complete, and the screen now. Two images rather than a description, because the
        thing being asked about is exactly a difference between two screens, and the segment
        summaries were written without any knowledge of the earlier state.

        Runs only when a step has actually been completed, so an episode whose first step
        never lands never pays for it.

        :return: A sentence naming what was lost, or ``None`` if nothing was.
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

        Only the **remaining** steps are replanned. Steps already cleared are shown as DONE
        and explicitly excluded, because replanning finished work spends budget redoing what
        the screen already shows is done — and, worse, invites the executor to undo it.

        :data:`PLAN_FLAW_PROMPT` is written to answer *no* by default. Repeated failure is
        not evidence of a bad plan — a fumbled step fails repeatedly too — so it demands
        something visible that the plan is incompatible with, and a replan that fired on
        ordinary fumbling would throw away a sound route on the first hard step.

        :return: The replacement steps for ``index`` onwards, or ``None`` to carry on
            hinting the current target.
        """
        if self.n_replans >= self.max_replans:
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

        raw = (parse_key_value(output, "Plan") or "").strip()
        replacement = parse_steps(raw)
        if not replacement or raw.upper().startswith("NONE"):
            # Said flawed and produced nothing to replace it with. Hinting the existing step
            # is a worse plan than no plan, but it is a plan.
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
        """Track the spliced list, not the replacement.

        The engine does ``targets[index:] = replacement``, so the replacement is only the
        *tail*. Assigning it to :attr:`plan` directly drops every step already cleared —
        which showed up as an episode reporting ``0/1 steps cleared`` after two replans of a
        four-step plan.
        """
        self.plan = [t for t in targets if t is not None]

    def _extras(self) -> dict:
        return {**super()._extras(),
                "plan": list(self.plan),
                "n_replans": self.n_replans}
