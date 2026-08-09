"""
Executors that commit to a multi-step plan before acting.

Both variants here decide on a sequence of moves up front rather than choosing one
action at a time.  :class:`SequencePlannerExecutor` overrides :meth:`_execute`
outright instead of using the base class hooks — see the note in
:mod:`execution.executors.base`.

:class:`SequencePlannerExecutor` is the one executor whose VLM calls own several steps
each — see its class docstring.  :class:`SubgoalDecomposerExecutor` takes one action per
call like everything else.
"""

from __future__ import annotations

from typing import List, Optional

from gameboy_worlds.interface.action import LowLevelAction

from execution.executors.base import MAX_CONSECUTIVE_INVALID
from execution.executors.simple import SimpleExecutor
from execution.report import EnvironmentStepRecord


class SequencePlannerExecutor(SimpleExecutor):
    """
    VLM outputs a comma-separated sequence of actions per call.  The sequence
    is executed as a committed plan until it is exhausted or an action fails,
    at which point the VLM is re-queried.

    Response format::

        Reasoning: <text>
        Action: UP, UP, RIGHT, A
        [STOP]

    Under ``allow_self_termination`` the completion check runs **once per plan**, not once
    per action as it does elsewhere — at the end of a sequence that ran to exhaustion
    without a failed action.  A committed plan is the unit this executor reasons about, so
    a partially-executed one is a state its planner never intended to be judged in.
    :data:`DONE_CHECK_EVERY_K_STEPS` does not apply here; see :meth:`_done_check_due`.

    .. note:: **This is the executor that owns several steps per VLM call.**

        One ``_vlm_call("action", ...)`` per *sequence*, then one
        :class:`~execution.report.EnvironmentStepRecord` per *action in that sequence* —
        so its :attr:`~execution.report.VLMCallRecord.steps` holds N entries where every
        other executor's holds one.  That is representable because ownership is stored on
        the call record as each step is taken.

        It was disabled for a while: ``steps`` used to be a second list on the report,
        paired with the call log by position, and a call that produced three steps slid
        that pairing out of alignment from the *next* call onward.  The steps were
        attributed to the wrong calls, and every consumer inherited the offset.  Nothing
        reconstructs the pairing now, so the failure mode is gone rather than worked
        around.
    """

    # This executor reasons once per *plan*, then executes several actions from it, so the
    # reasoning the completion check is shown is the plan's rather than the step's. Say so
    # rather than presenting it as step-level reasoning.
    DONE_CHECK_REASONING_LABEL = "The reasoning given for the planned sequence this action came from was:"

    def _done_check_due(self, n_env_steps: int) -> bool:
        """Always due — this executor's cadence is the plan, not the step.

        :data:`DONE_CHECK_EVERY_K_STEPS` exists to stop the check firing after every
        environment step on executors that would otherwise pay for one per action.  This
        executor already checks once per committed plan, and its ``_execute`` decides when
        that is.  Layering *k* on top would skip the check on a plan whose last action
        happened to land on a non-due step, and the next opportunity would not come until
        the *end of the following plan* — an arbitrary number of steps later, gated on a
        counter that has nothing to do with plan boundaries.  So *k* is ignored outright
        rather than combined.
        """
        return True

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False

        pending_sequence: List[str] = []
        error_message: Optional[str] = None
        n_env_steps: int = 0
        consecutive_invalid: int = 0

        while n_env_steps < self._max_steps:
            # Re-query when sequence is exhausted
            if not pending_sequence:
                state = self._get_state()
                frame = state["core"]["current_frame"]
                prompt = self._build_sequence_prompt(error_message)
                response = self._vlm_call(
                    "action",
                    texts=prompt,
                    images=[frame],
                    )
                # Captured once per plan, and the plan gets exactly one completion check
                # (at its end), so this is the reasoning that check sees; the
                # DONE_CHECK_REASONING_LABEL override says so in the prompt.
                self._last_reasoning = self._parse_reasoning(response) or self._last_reasoning
                sequence = self._parse_sequence(response)
                if sequence is None:
                    error_message = (
                        "Your previous response could not be parsed. "
                        "You must end your response with:\n"
                        "  Action: ACTION1, ACTION2, ...\n"
                        "  [STOP]"
                    )
                    self._record_invalid(response)
                    n_env_steps += 1
                    consecutive_invalid += 1
                    if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                        self.report.termination_reason = "max_invalid"
                        return -1
                    continue
                pending_sequence = sequence
                error_message = None

            # Execute next action in sequence
            action_str = pending_sequence.pop(0)
            action_class, action_kwargs = self._env.string_to_high_level_action(action_str)
            if action_class is None:
                error_message = (
                    f"'{action_str}' in planned sequence is not a recognised action. "
                    "Re-plan with valid actions."
                )
                self._record_invalid(f"Unrecognised sequence action: {action_str!r}", reason="unrecognised action")
                pending_sequence = []  # abort remainder of sequence
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            record = self._take_action(action_class, **(action_kwargs or {}))
            n_env_steps += 1
            consecutive_invalid = 0

            if self._last_terminated:
                self.report.termination_reason = "terminated"
                return 1
            if self._last_truncated:
                self.report.termination_reason = "truncated"
                return 2

            # If action failed, abort remaining sequence and re-plan.
            # LowLevelActions always return success=0 by convention (not a failure signal),
            # so skip the check for them entirely.
            last_record = self._current_call.steps[-1]
            action_failed = (
                not issubclass(action_class, LowLevelAction)
                and isinstance(last_record, EnvironmentStepRecord)
                and last_record.action_success == 0
            )
            if action_failed:
                error_message = (
                    f"Action '{action_str}' failed (blocked or invalid). Re-plan."
                )
                pending_sequence = []

            # Checked once per plan, at its end, and only when the plan actually ran:
            # this executor commits to a sequence, so a half-executed plan is a state the
            # planner never intended to be judged in. Both guards matter and they are not
            # the same guard — `action_failed` also empties `pending_sequence`, so testing
            # only for exhaustion would still fire on the aborted case.
            if not action_failed and not pending_sequence:
                outcome = self._maybe_self_terminate(record, n_env_steps)
                if outcome is not None:
                    return outcome

        self.report.termination_reason = "max_steps"
        return 0

    SEQUENCE_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK]Available environment actions:
[ACTION_LIST]

Plan a short sequence of actions (1–5) to make progress on the task. Respond in exactly this format:
Reasoning: <your reasoning>
Action: <ACTION1, ACTION2, ...>
[STOP]"""

    def _build_sequence_prompt(self, error_message: Optional[str]) -> str:
        return (
            self.SEQUENCE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[ACTION_LIST]", self._action_list_block())
        )

    def _parse_sequence(self, response: str) -> Optional[List[str]]:
        """Parse a comma-separated action sequence from a structured VLM response."""
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                action_part = stripped[len("action:"):].strip()
                action_part = action_part.replace("[STOP]", "").strip()
                if not action_part:
                    return None
                parts = [p.strip() for p in action_part.split(",") if p.strip()]
                return parts if parts else None
        return None


class SubgoalDecomposerExecutor(SimpleExecutor):
    """
    Before the main loop, asks the VLM to decompose the task into 2-4 ordered
    subgoals.  The current subgoal is injected into every step prompt, and
    advances after a configurable number of steps or when the VLM signals completion.

    :param steps_per_subgoal: Env steps before automatically advancing to next subgoal (default 10).
    :type steps_per_subgoal: int
    """

    def __init__(self, env, task, max_steps, max_tool_calls, steps_per_subgoal: int = 10, **kwargs):
        self._steps_per_subgoal = steps_per_subgoal
        self._subgoals: List[str] = []
        self._subgoal_idx: int = 0
        self._steps_on_subgoal: int = 0
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    DECOMPOSE_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

Break this task into 2-4 clear, ordered subgoals. Each subgoal should be a short action phrase.

Respond in exactly this format:
Subgoal 1: <first subgoal>
Subgoal 2: <second subgoal>
... (up to Subgoal 4)
[STOP]"""

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]
[SUBGOAL_LINE]
You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _decompose_task(self) -> List[str]:
        """Call the VLM once to decompose the task into ordered subgoals."""
        state = self._get_state()
        frame = state["core"]["current_frame"]
        prompt = self.DECOMPOSE_PROMPT.replace("[TASK]", self._task).replace("[HINT_BLOCK]", self._hint_block())
        response = self._vlm_call(
            "decompose",
            texts=prompt,
            images=[frame],
            max_new_tokens=200,
        )
        subgoals = []
        for line in response.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("subgoal"):
                colon_idx = stripped.find(":")
                if colon_idx != -1:
                    sg = stripped[colon_idx + 1:].strip()
                    if sg:
                        subgoals.append(sg)
        return subgoals if subgoals else [self._task]

    def _on_execute_start(self) -> None:
        self._subgoals = self._decompose_task()
        self._subgoal_idx = 0
        self._steps_on_subgoal = 0
        super()._on_execute_start()

    def _on_env_step(self, record: EnvironmentStepRecord) -> None:
        self._steps_on_subgoal += 1
        if (self._steps_on_subgoal >= self._steps_per_subgoal
                and self._subgoal_idx < len(self._subgoals) - 1):
            self._subgoal_idx += 1
            self._steps_on_subgoal = 0

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        if self._subgoals:
            current_subgoal = self._subgoals[self._subgoal_idx]
            subgoal_line = f"Current subgoal ({self._subgoal_idx + 1}/{len(self._subgoals)}): {current_subgoal}"
        else:
            subgoal_line = ""
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[SUBGOAL_LINE]", subgoal_line)
        )


