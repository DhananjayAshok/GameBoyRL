"""
The reference executor and its history-carrying variant.

:class:`SimpleExecutor` is the concrete implementation of the loop that
:class:`~execution.executors.base.Executor` defines; every other variant in this
package subclasses it rather than the ABC.  :class:`HistoryAwareExecutor` adds a
block of recent history to the step prompt and changes nothing else.
"""

from __future__ import annotations

from typing import List, Optional, Type

from gameboy_worlds.interface import HighLevelAction
from gameboy_worlds.interface.action import LowLevelAction

from execution.executors.base import Executor, MAX_CONSECUTIVE_INVALID
from execution.report import EnvironmentStepRecord, SimpleReport
from utils import parse_action_line


class SimpleExecutor(Executor):
    """
    A straightforward VLM-driven executor.

    At each iteration the executor:

    1. Builds a text + image prompt from the current game frame, the task, the
       available high-level actions, the available tools (omitted once the tool
       budget is exhausted), and the result of the previous tool call (or an
       error message if the last response was unparseable).
    2. Calls the VLM and parses the structured response::

           Reasoning: <free text>
           Action: <single action string>
           [STOP]

    3. Dispatches the action:

       - **Valid tool call** (budget not yet exhausted): executes the tool,
         forwards its result to the next prompt, does **not** count as an
         environment step.
       - **Valid env action**: steps the environment, increments the env-step
         counter.
       - **Unparseable or unrecognised**: records an
         :class:`~execution.report.InvalidStepRecord` in ``steps`` (surfaced as
         :attr:`~execution.report.ExecutorReport.invalid_steps`), passes an error
         message into the next prompt, and counts as an environment step to prevent
         infinite loops.

    4. When *allow_self_termination* is enabled, asks the VLM whether the task
       is now complete (:meth:`~execution.executors.Executor._maybe_self_terminate`)
       and stops if it says yes.  The action prompt above is unaffected by that
       flag — the question is asked separately, after the step, never as part of
       choosing one.

    Terminates early when the environment signals ``terminated`` or
    ``truncated``, or when the completion check fires, recording the reason in
    :attr:`~execution.report.ExecutorReport.termination_reason`.

    .. warning:: **Subclass initialisation order**

        Per :class:`Executor` contract, call ``super().__init__()`` **last**.
    """

    #: ``[CONTEXT_SECTION]`` is where a variant splices whatever state it carries between
    #: steps — a spatial map, a belief state, a plan, an action history. It renders empty
    #: unless :meth:`_context_section` is overridden, so a variant that wants one adds three
    #: lines rather than copying this whole template to insert a placeholder.
    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][CONTEXT_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _make_report(self, task, init_kwargs, max_steps, max_tool_calls) -> SimpleReport:
        return SimpleReport(
            task=task,
            executor_name=self.__class__.__name__,
            game=self._game,
            init_kwargs=init_kwargs,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            initial_state=self._get_state(),
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _take_action(self, action_class: Type[HighLevelAction], **kwargs) -> EnvironmentStepRecord:
        """
        Execute a high-level action, record it, and stash the env's
        ``terminated`` / ``truncated`` flags on ``self`` for :meth:`_execute`
        to inspect.
        """
        before_info = self._env.get_info()
        frame_before = before_info["core"]["current_frame"]
        obs, reward, terminated, truncated, info = self._env.step_high_level_action(
            action_class, **kwargs
        )
        if "previous_action_details" in info.get("core", {}):
            _, _, transition_states, action_success, _ = info["core"]["previous_action_details"]
        else:
            transition_states, action_success = [], -1
        frame_after = info["core"]["current_frame"]

        record = EnvironmentStepRecord(
            frame_before=frame_before,
            frame_after=frame_after,
            action_class=action_class,
            kwargs=kwargs,
            transition_states=transition_states,
            action_success=action_success,
            reward=reward,
        )
        self._record_step(record)
        self._last_terminated = terminated
        self._last_truncated = truncated
        self._last_frame_changed = info["core"].get("frame_changed", True)
        return record

    # ------------------------------------------------------------------
    # Template loop hooks — override in subclasses to customise behaviour
    # ------------------------------------------------------------------

    def _on_execute_start(self) -> None:
        """Called once at the start of execution, before the main loop."""
        pass

    def _on_step_start(self, frame) -> None:
        """Called at the top of each loop iteration after getting the current frame."""
        pass

    def _query_vlm(self, prompt: str, frame) -> str:
        """Call the VLM for this step and return its raw output."""
        return self._vlm_call("action", texts=prompt, images=[frame])

    def _pick_action(self, vlm_output) -> Optional[str]:
        """Parse VLM output into an action string. Returns None on parse failure."""
        # The single funnel every action response passes through, so it is where the
        # reasoning is captured for the completion check. Kept even when the action fails
        # to parse: a stale reasoning is more useful to the judge than none, and the next
        # successful parse overwrites it.
        self._last_reasoning = self._parse_reasoning(vlm_output) or self._last_reasoning
        return self._parse_action(vlm_output)

    def _on_env_step(self, record: EnvironmentStepRecord) -> None:
        """Called after a successful env step, with the resulting record."""
        pass

    def _context_section(self) -> str:
        """State carried between steps, spliced into ``[CONTEXT_SECTION]``.

        Empty here, so the reference executor's prompt is unchanged. A variant that
        maintains something across steps overrides this instead of restating the whole
        step prompt to add a placeholder — which is how four variants came to hold
        near-identical copies of a template that then had to be edited in five places.

        Return a block ending in a blank line if non-empty; it sits directly before the
        "Reason about the best next action" line.
        """
        return ""

    def _on_parse_failure(self, vlm_output) -> str:
        """Record an unparseable response and return the error to put in the next prompt.

        A hook because the wording has to match what the variant actually asked for: an
        executor whose prompt requests a score table must not be told it should have ended
        with ``Action:``. Recording and messaging are one method because the two must stay
        in step — every invalid step produces exactly one record and one message.
        """
        self._record_invalid(str(vlm_output))
        return (
            "Your previous response could not be parsed. "
            "You must end your response with:\n"
            "  Action: <action>\n"
            "  [STOP]"
        )

    def _on_unrecognised_action(self, action_str: str) -> str:
        """Record a well-formed but unknown action and return the error for the next prompt."""
        self._record_invalid(f"Unrecognised action string: {action_str!r}",
                             reason="unrecognised action")
        return (
            f"You tried to do '{action_str}' but that is not a recognised action. DO NOT use '{action_str}' in your response. "
            "Choose exactly one from the listed actions."
        )

    def _step_template(self) -> str:
        """The prompt template :meth:`_build_prompt` fills.

        A hook rather than a direct ``self.STEP_PROMPT`` read so a variant can choose
        between templates per step — :class:`~execution.executors.stateful.ScreenDiffExecutor`
        picks a one-image or two-image wording depending on whether it has a previous
        frame — without reimplementing the substitution chain.
        """
        return self.STEP_PROMPT

    # ------------------------------------------------------------------
    # Unified execution loop
    # ------------------------------------------------------------------

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._on_execute_start()

        tool_call_message: Optional[str] = None
        error_message: Optional[str] = None
        n_env_steps: int = 0
        consecutive_invalid: int = 0

        while n_env_steps < self._max_steps:
            state = self._get_state()
            frame = state["core"]["current_frame"]
            self._on_step_start(frame)

            tool_calls_exceeded = self._n_tool_calls >= self._max_tool_calls
            prompt = self._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            vlm_output = self._query_vlm(prompt, frame)
            action_str = self._pick_action(vlm_output)

            if action_str is None:
                error_message = self._on_parse_failure(vlm_output)
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            if not tool_calls_exceeded:
                tool_result = self._try_parse_tool_call(action_str)
                if tool_result is not None:
                    tool_class, tool_kwargs = tool_result
                    record = self._use_tool(tool_class, **tool_kwargs)
                    tool_call_message = str(record.result)
                    error_message = None
                    consecutive_invalid = 0
                    continue

            action_class, action_kwargs = self._env.string_to_high_level_action(action_str)
            if action_class is not None:
                record = self._take_action(action_class, **(action_kwargs or {}))
                tool_call_message = None
                error_message = None
                n_env_steps += 1
                consecutive_invalid = 0
                self._on_env_step(record)

                if self._last_terminated:
                    self.report.termination_reason = "terminated"
                    return 1
                if self._last_truncated:
                    self.report.termination_reason = "truncated"
                    return 2

                outcome = self._maybe_self_terminate(record, n_env_steps)
                if outcome is not None:
                    return outcome
            else:
                error_message = self._on_unrecognised_action(action_str)
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1

        self.report.termination_reason = "max_steps"
        return 0

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_block(error_message: Optional[str]) -> str:
        return f"[ERROR] {error_message}\n\n" if error_message is not None else ""

    @staticmethod
    def _tool_result_block(tool_call_message: Optional[str]) -> str:
        return f"Tool result: {tool_call_message}\n\n" if tool_call_message is not None else ""

    def _action_list_block(self) -> str:
        # Only real environment actions, always. Nothing about termination is advertised
        # here, so the prompt is identical whether or not self-termination is enabled —
        # which is what keeps SFT rows harvested from the two configurations comparable.
        return "\n".join(f"  {s}" for s in self._get_action_strings().values())

    def _tools_block(self, tool_calls_exceeded: bool) -> str:
        if tool_calls_exceeded or not self.available_tools:
            return ""
        lines = [
            f"Available tool calls (do not advance the game, {self._max_tool_calls} total budget):"
        ]
        for tool_class in self.available_tools:
            lines.append(f"  {tool_class.verbalize()}")
        return "\n".join(lines) + "\n\n"

    def _action_format(self, tool_calls_exceeded: bool) -> str:
        if not tool_calls_exceeded and self.available_tools:
            return "Action: <one environment action OR one tool call>"
        return "Action: <one environment action>"

    def _build_prompt(
        self,
        tool_call_message: Optional[str],
        error_message: Optional[str],
        tool_calls_exceeded: bool,
    ) -> str:
        return (
            self._step_template()
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[TOOL_RESULT_BLOCK]", self._tool_result_block(tool_call_message))
            .replace("[ACTION_LIST]", self._action_list_block())
            .replace("[TOOLS_BLOCK]", self._tools_block(tool_calls_exceeded))
            .replace("[CONTEXT_SECTION]", self._context_section())
            .replace("[ACTION_FORMAT]", self._action_format(tool_calls_exceeded))
        )

    def _parse_action(self, response: str) -> Optional[str]:
        """Extract the action string from a structured VLM response.

        Kept as a method, delegating to :func:`utils.parsing.parse_action_line`, because it
        is part of the extension surface the executor variants are built on — a subclass
        reading actions out of a different format overrides this. The format itself is a
        cross-module contract shared with the frame tooling, so it is
        defined once in utils.parsing rather than here.
        """
        return parse_action_line(response)

    def _try_parse_tool_call(
        self, action_str: str
    ) -> Optional[tuple]:
        """
        Try to parse ``action_str`` as a call to one of :attr:`available_tools`.

        :return: ``(tool_class, kwargs)`` on success, ``None`` otherwise.
        """
        for tool_class in self.available_tools:
            try:
                kwargs = tool_class.string_to_kwargs(action_str)
                if kwargs is not None:
                    return tool_class, kwargs
            except Exception:
                continue
        return None


class HistoryAwareExecutor(SimpleExecutor):
    """
    Extends :class:`SimpleExecutor` by including the last *k* environment steps
    in each prompt so the VLM can see what it has already tried.

    :param history_k: Number of recent env steps to include in context (default 5).
    :type history_k: int
    """

    def __init__(self, env, task, max_steps, max_tool_calls, history_k: int = 5, **kwargs):
        self._history_k = history_k
        # (action_class, action_str, success_code, frame_changed)
        self._action_history: List[tuple] = []
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        record = super()._take_action(action_class, **kwargs)
        action_str = action_class.get_action_name(**kwargs)
        self._action_history.append((action_class, action_str, record.action_success, self._last_frame_changed))
        return record

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][CONTEXT_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning, specifically reason over your history as well. If you see the [no change] message on the action that you are trying, then you almost certainly have slightly misperceived the screen position of the player relative to the objects. In that case, reason about what else you can try instead of just repeating the same action.>
[ACTION_FORMAT]
[STOP]"""

    def _context_section(self) -> str:
        if not self._action_history:
            return ""
        recent = self._action_history[-self._history_k:]
        history_lines = ["Recent actions (oldest first):"]
        frame_change_hint = ""
        for action_cls, action_str, success, frame_changed in recent:
            if issubclass(action_cls, LowLevelAction):
                tags = "" if frame_changed else " [no change]"
                if not frame_changed:
                    frame_change_hint = "\nIf you have been trying to execute the same action repeatedly (specifically A or B) and especially if you get the [no change] message on your recent actions, consider that you may be stuck in a loop, and should try something else. Look at the screen deeply and use the visual cues to guide your decision making. If you are trying to interact with something, you likely have the incorrect orientation and need to slightly adjust your positioning"
                history_lines.append(f"  {action_str}{tags}")
            else:
                status = "ok" if success == 1 else ("failed" if success == 0 else "unknown")
                tags = "" if frame_changed else ", no change"
                history_lines.append(f"  {action_str}  [{status}{tags}]")
        return "\n".join(history_lines) + frame_change_hint + "\n\n"


