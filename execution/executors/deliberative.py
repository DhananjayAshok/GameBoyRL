"""
Executors that change how a single decision is reached.

Each variant here intervenes at the point of choosing one action — scoring candidates, or
reflecting on the last one — via :meth:`_query_vlm` or :meth:`_pick_action`.  The state
carried between steps is unchanged from
:class:`~execution.executors.simple.SimpleExecutor`.

**The call that decides the action must be the last one, and must be action-tagged.**
Whichever VLM call an executor makes last before returning from :meth:`_query_vlm` is the
one that owns the resulting step (see :meth:`~execution.executors.base.Executor._vlm_call`),
and consumers that reconstruct an action sequence filter on
:data:`~execution.report.ACTION_TAGS` — currently ``"action"`` and ``"score"``. Two
executors were removed for breaking that rather than being reconciled with it: a
``SelfConsistencyExecutor`` logged several action calls per step, and a
``ConfidenceGatedExecutor`` decided the action in a ``"rethink"`` call, so every action it
took was dropped from the dataset.

Auxiliary calls (``"reflection"``) own no steps and are free to be as numerous as they
like, provided an action-tagged call comes last.

Every executor in this module uses the base class loop; none overrides :meth:`_execute`
— see the note in :mod:`execution.executors.base` for the one that does.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from gameboy_worlds.interface.action import LowLevelAction

from execution.executors.simple import SimpleExecutor
from execution.report import EnvironmentStepRecord
from utils import parse_int


class ReflectiveExecutor(SimpleExecutor):
    """
    Every *reflection_interval* env steps, calls the VLM with the recent action
    history and current frame to produce a short critique and revised plan.
    That plan is injected into subsequent step prompts.

    :param reflection_interval: Env steps between reflection calls (default 5).
    :type reflection_interval: int
    """

    def __init__(self, env, task, max_steps, max_tool_calls,
                 reflection_interval: int = 5, **kwargs):
        self._reflection_interval = reflection_interval
        self._plan_summary: str = ""
        self._steps_since_reflection: int = 0
        self._reflection_action_log: List[str] = []
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        record = super()._take_action(action_class, **kwargs)
        action_str = action_class.get_action_name(**kwargs)
        if issubclass(action_class, LowLevelAction):
            tags = "" if self._last_frame_changed else " [no change]"
            self._reflection_action_log.append(f"{action_str}{tags}")
        else:
            status = "ok" if record.action_success == 1 else "failed"
            tags = "" if self._last_frame_changed else ", no change"
            self._reflection_action_log.append(f"{action_str} [{status}{tags}]")
        self._steps_since_reflection += 1
        return record

    REFLECTION_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRIOR_PLAN]Recent actions taken: [HISTORY]

The current game screen is shown in the image.

Briefly critique whether the recent actions made progress toward the task. Then state a concise plan for the next few steps (1-2 sentences). End your response with [STOP].
[STOP]"""

    def _reflect(self, frame) -> None:
        history_str = ", ".join(self._reflection_action_log) if self._reflection_action_log else "none"
        self._reflection_action_log = []
        self._steps_since_reflection = 0
        prior_plan = f"Prior plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        prompt = (
            self.REFLECTION_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PRIOR_PLAN]", prior_plan)
            .replace("[HISTORY]", history_str)
        )
        result = self._vlm_call("reflection", texts=prompt, images=[frame], max_new_tokens=200)
        self._plan_summary = result.strip()

    def _on_step_start(self, frame) -> None:
        if self._steps_since_reflection >= self._reflection_interval:
            self._reflect(frame)

    def _context_section(self) -> str:
        return f"Current plan: {self._plan_summary}\n\n" if self._plan_summary else ""


class ActionValueEstimatorExecutor(SimpleExecutor):
    """
    Instead of asking the VLM to directly choose an action, this executor asks
    it to justify and score every available action on a 1-5 scale, then
    automatically selects the highest-scored one.  Ties are broken by order in
    the list.

    This forces systematic evaluation of all options rather than anchoring on
    the first plausible action that comes to mind.

    Score prompt response format::

        <action_string>: <one-sentence reason>: <1-5>
        <action_string>: <one-sentence reason>: <1-5>
        ...
        [STOP]

    The winning line's justification becomes the step's reasoning, so the completion
    check sees why *this* action beat the others rather than nothing at all.
    """

    #: Replaces the inherited step prompt outright — this executor never asks for an
    #: action, only for scores. The unused ``[TOOL_RESULT_BLOCK]`` / ``[TOOLS_BLOCK]`` /
    #: ``[CONTEXT_SECTION]`` / ``[ACTION_FORMAT]`` slots are simply absent, and the base
    #: substitution chain leaves them alone.
    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

[ERROR_BLOCK]You are playing a GameBoy game. The current screen is shown in the image.

Score each available action on how useful it would be RIGHT NOW for making progress toward the task (1=useless, 5=very useful), and justify each score in one sentence.

Actions:
[ACTION_LIST]

Respond with one line per action in exactly this format:
<action>: <one-sentence reason for the score>: <score>
...
[STOP]"""

    def _query_vlm(self, prompt: str, frame) -> str:
        return self._vlm_call("score", texts=prompt, images=[frame], max_new_tokens=300)

    def _parse_score_line(self, line: str) -> Optional[Tuple[str, str, int]]:
        """Split one ``<action>: <reasoning>: <score>`` line into its three fields.

        Anchored at both ends rather than on a single separator, because only the outer
        two colons are reliable: the action is everything before the **first** one and the
        score everything after the **last**, leaving the middle — reasoning, colons and
        all — untouched. An action invocation is ``name(args)`` and never contains a colon
        (:meth:`~gameboy_worlds.interface.Environment.string_to_high_level_action` splits
        on the parentheses), so the first colon is always the action's terminator even
        though the *listed* verbalisation of that action carries a colon before its
        description.

        A line with only one colon is read as an action and a score with the justification
        omitted, rather than rejected — a missing reason costs the completion check some
        context, but the score it came with is still a usable vote.

        :return: ``(action_string, reasoning, score)``, or None if the line has no colon,
            no action, or no in-range score.
        """
        stripped = line.strip()
        first, last = stripped.find(":"), stripped.rfind(":")
        if first == -1:
            return None
        action_str = stripped[:first].strip()
        reasoning = stripped[first + 1:last].strip() if last > first else ""
        # Scored through a synthetic keyed line so parse_int only ever sees the score
        # field. Handing it the whole line with the action as key — which is what this did
        # when a line was just `<action>: <score>` — would search the justification for
        # digits too, and "reach route 1, best odds: 4" would score 1. parse_int itself
        # stays: it reads "10" as 10 rather than 1, and range-checks against the 1-5 scale
        # so an out-of-range score is discarded rather than allowed to win the max.
        score = parse_int(f"score: {stripped[last + 1:]}", "score", 1, 5)
        if not action_str or score is None:
            return None
        return action_str, reasoning, score

    def _pick_action(self, vlm_output: str) -> Optional[str]:
        """The highest-scored action, or None if no line parsed.

        Sets ``_last_reasoning`` to the winning line's justification only — the losing
        actions' reasons explain a choice that was not made, so folding them in would
        misdescribe the step to the completion check. When the model omitted the
        justification the previous step's reasoning is left in place, matching what
        :meth:`~execution.executors.simple.SimpleExecutor._pick_action` does with an
        unparseable one.
        """
        best_score = -1
        best_action_str = None
        best_reasoning = ""
        for line in vlm_output.splitlines():
            parsed = self._parse_score_line(line)
            if parsed is None:
                continue
            action_str, reasoning, score = parsed
            if score > best_score:
                best_score = score
                best_action_str = action_str
                best_reasoning = reasoning
        if best_action_str is not None and best_reasoning:
            self._last_reasoning = best_reasoning
        return best_action_str

    def _on_parse_failure(self, vlm_output) -> str:
        self._record_invalid("Failed to parse action scores")
        return (
            "Could not determine a valid action from scoring. Score every action again, "
            "one per line, in exactly this format:\n"
            "  <action>: <one-sentence reason for the score>: <score 1-5>"
        )

    def _on_unrecognised_action(self, action_str: str) -> str:
        self._record_invalid(f"Unrecognised scored action: {action_str!r}",
                             reason="unrecognised action")
        return (
            f"Highest-scored action '{action_str}' is not a recognised action string. "
            "Re-score using exact action strings from the list."
        )
