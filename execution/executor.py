"""
Abstract base class for all executors.

An executor is a one-shot agent that receives a game environment and a natural-
language task, then attempts to complete that task by interleaving passive tool
calls (:class:`~execution.executor_action.ExecutorAction`) with active
high-level environment steps.

Execution begins automatically inside :meth:`Executor.__init__` and cannot be
triggered a second time — the executor is sealed after construction.

.. warning:: **Subclass initialisation order**

    :meth:`Executor.__init__` calls :meth:`_execute` immediately before it
    returns.  Any attribute a subclass needs during :meth:`_execute` **must**
    be set *before* calling ``super().__init__()``.  Place the ``super()`` call
    as the **last** line of the subclass ``__init__``.

    **Correct pattern**::

        class MyExecutor(Executor):
            def __init__(self, env, task, max_steps, max_tool_calls, my_arg, **kwargs):
                self._my_arg = my_arg          # set up own state FIRST
                super().__init__(env, task, max_steps, max_tool_calls, **kwargs)  # LAST

    **Wrong pattern** (will crash — :meth:`_execute` fires before
    ``self._my_arg`` exists)::

        class MyExecutor(Executor):
            def __init__(self, env, task, max_steps, max_tool_calls, my_arg, **kwargs):
                super().__init__(...)   # triggers _execute() immediately
                self._my_arg = my_arg  # too late — _execute already ran
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from gameboy_worlds.interface import Environment, HighLevelAction

from execution.executor_action import ExecutorAction
from execution.report import EnvironmentStepRecord, ExecutorReport, SimpleReport, ToolCallRecord
from utils import load_parameters
from utils.vlm import ExecutorVLM


class Executor(ABC):
    """
    Abstract base class for game-playing executor agents.

    Subclasses implement :meth:`_execute` (game loop logic) and
    :meth:`_take_action` (how a high-level action is dispatched to the
    environment).  The base class owns the report lifecycle: it creates
    :attr:`report` before calling :meth:`_execute` and seals it immediately
    after.

    .. warning:: **Subclass initialisation order**

        ``super().__init__()`` triggers :meth:`_execute` immediately.  Set all
        subclass attributes *before* calling ``super().__init__()``.  See
        module-level docstring for the correct pattern.

    :param env: The game environment.  All environment interaction must go
        through this object — never access ``env._emulator`` directly.
    :type env: Environment
    :param task: Natural-language description of the task to complete.
    :type task: str
    :param max_steps: Maximum number of high-level environment steps allowed.
    :type max_steps: int
    :param max_tool_calls: Maximum number of passive tool calls allowed.
    :type max_tool_calls: int
    :param parameters: Optional parameter overrides forwarded to
        :func:`~utils.parameter_handling.load_parameters`.
    :type parameters: Optional[dict]
    :param kwargs: Additional subclass-specific keyword arguments.  These are
        recorded verbatim in :attr:`report.init_kwargs` but are otherwise
        ignored by the base class.

    .. attribute:: available_tools
        :type: List[Type[ExecutorAction]]

        Class-level list of :class:`~execution.executor_action.ExecutorAction`
        subclasses this executor may call.  Defaults to ``[]`` (no tools).
        Override at the subclass level::

            class MyExecutor(Executor):
                available_tools = [LocateAction, OCRAction]
    """

    available_tools: list = []

    def __init__(
        self,
        env: Environment,
        task: str,
        max_steps: int,
        max_tool_calls: int,
        parameters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self._env = env
        self._task = task
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._parameters = load_parameters(parameters)
        self._vlm = ExecutorVLM(parameters=self._parameters)

        self.report = self._make_report(task, kwargs, max_steps, max_tool_calls)

        outcome = self._execute()

        self.report.outcome = outcome
        self.report.final_state = self._get_state()

    def _make_report(
        self,
        task: str,
        init_kwargs: dict,
        max_steps: int,
        max_tool_calls: int,
    ) -> ExecutorReport:
        """
        Factory for the report object.  Override in subclasses to return a
        different :class:`~execution.report.ExecutorReport` subclass.

        Called by :meth:`__init__` before :meth:`_execute` runs, so
        ``self._get_state()`` is already available.
        """
        return ExecutorReport(
            task=task,
            init_kwargs=init_kwargs,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            initial_state=self._get_state(),
        )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _execute(self) -> int:
        """
        Run the executor's game loop.

        Called automatically by :meth:`__init__`.  Has access to
        ``self._max_steps`` and ``self._max_tool_calls`` via instance
        attributes, and should use :meth:`_take_action` and :meth:`_use_tool`
        to interact with the environment and passive tools respectively.

        :return: Executor-defined integer outcome code stored in
            :attr:`report.outcome`.
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def _take_action(self, *args: Any, **kwargs: Any) -> EnvironmentStepRecord:
        """
        Dispatch a high-level action to the environment and record the result.

        Implementations **must**:

        - Use :meth:`~gameboy_worlds.interface.Environment.step_high_level_action`
          or :meth:`~gameboy_worlds.interface.Environment.step_str` — never
          access the emulator directly.
        - Build an :class:`~execution.report.EnvironmentStepRecord` and append
          it to ``self.report.steps``.
        - Return the record.

        :return: The record of the environment step that was taken.
        :rtype: EnvironmentStepRecord
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete helpers — uniform across all executors
    # ------------------------------------------------------------------

    def _use_tool(
        self,
        executor_action_class: Type[ExecutorAction],
        **kwargs: Any,
    ) -> ToolCallRecord:
        """
        Invoke a passive :class:`~execution.executor_action.ExecutorAction` and
        record the result.

        The action receives the current environment state via
        :meth:`_get_state` but does **not** advance the emulator.  The
        resulting :class:`~execution.report.ToolCallRecord` is appended to
        ``self.report.steps`` before being returned.

        :param executor_action_class: The
            :class:`~execution.executor_action.ExecutorAction` subclass to
            invoke.
        :type executor_action_class: Type[ExecutorAction]
        :param kwargs: Additional keyword arguments forwarded to
            :meth:`~execution.executor_action.ExecutorAction.execute`.
        :return: The record of the tool call that was made.
        :rtype: ToolCallRecord
        """
        action = executor_action_class()
        result, success_code = action.execute(info=self._get_state(), **kwargs)

        record = ToolCallRecord(
            executor_action_class=executor_action_class,
            kwargs=kwargs,
            result=result,
            success_code=success_code,
        )
        self.report.steps.append(record)
        return record

    def _get_state(self) -> dict:
        """
        Return the current environment state.

        :return: State info dictionary from
            :meth:`~gameboy_worlds.interface.Environment.get_info`.
        :rtype: dict
        """
        return self._env.get_info()

    def _get_action_strings(self, return_all: bool = False) -> Dict[Type[HighLevelAction], str]:
        """
        Return the verbalized high-level actions available in the current state.

        :param return_all: If ``True``, returns every action regardless of current
            validity.  If ``False`` (default), returns only actions valid right now.
        :type return_all: bool
        :return: A dictionary mapping each :class:`~gameboy_worlds.interface.HighLevelAction`
            subclass to its verbalization and parameter format string.
        :rtype: Dict[Type[HighLevelAction], str]
        """
        return self._env.get_action_strings(return_all=return_all)


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
       - **Unparseable or unrecognised**: logs to :attr:`~execution.report.SimpleReport.invalid_steps`,
         passes an error message into the next prompt, and counts as an
         environment step to prevent infinite loops.

    Terminates early when the environment signals ``terminated`` or
    ``truncated``, recording the reason in
    :attr:`~execution.report.ExecutorReport.termination_reason`.

    .. warning:: **Subclass initialisation order**

        Per :class:`Executor` contract, call ``super().__init__()`` **last**.
    """

    def _make_report(self, task, init_kwargs, max_steps, max_tool_calls) -> SimpleReport:
        return SimpleReport(
            task=task,
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
        obs, reward, terminated, truncated, info = self._env.step_high_level_action(
            action_class, **kwargs
        )
        if "previous_action_details" in info.get("core", {}):
            _, _, transition_states, action_success, _ = info["core"]["previous_action_details"]
        else:
            transition_states, action_success = [], -1

        record = EnvironmentStepRecord(
            action_class=action_class,
            kwargs=kwargs,
            transition_states=transition_states,
            action_success=action_success,
        )
        self.report.steps.append(record)
        self._last_terminated = terminated
        self._last_truncated = truncated
        return record

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False

        tool_call_message: Optional[str] = None
        error_message: Optional[str] = None
        n_env_steps: int = 0

        while n_env_steps < self._max_steps:
            state = self._get_state()
            frame = state["core"]["current_frame"]

            n_tool_calls = sum(1 for s in self.report.steps if isinstance(s, ToolCallRecord))
            tool_calls_exceeded = n_tool_calls >= self._max_tool_calls

            prompt = self._build_prompt(tool_call_message, error_message, tool_calls_exceeded)

            response = self._vlm.infer(
                texts=prompt,
                images=[frame],
                max_new_tokens=self._parameters.get("executor_max_new_tokens", 512),
            )

            # ---- parse structured response --------------------------------
            action_str = self._parse_action(response)

            if action_str is None:
                error_message = (
                    "Your previous response could not be parsed. "
                    "You must end your response with:\n"
                    "  Action: <action>\n"
                    "  [STOP]"
                )
                tool_call_message = None
                n_env_steps += 1
                self.report.invalid_steps.append(response)
                continue

            # ---- try tool call (if budget not exhausted) ------------------
            if not tool_calls_exceeded:
                tool_result = self._try_parse_tool_call(action_str)
                if tool_result is not None:
                    tool_class, tool_kwargs = tool_result
                    record = self._use_tool(tool_class, **tool_kwargs)
                    tool_call_message = str(record.result)
                    error_message = None
                    continue  # tool calls do not count as env steps

            # ---- try env action -------------------------------------------
            action_class, action_kwargs = self._env.string_to_high_level_action(action_str)
            if action_class is not None:
                self._take_action(action_class, **(action_kwargs or {}))
                tool_call_message = None
                error_message = None
                n_env_steps += 1

                if self._last_terminated:
                    self.report.termination_reason = "terminated"
                    return 1
                if self._last_truncated:
                    self.report.termination_reason = "truncated"
                    return 2
            else:
                # Parseable format but unrecognised action string
                error_message = (
                    f"'{action_str}' is not a recognised action. "
                    "Choose exactly one from the listed actions."
                )
                tool_call_message = None
                n_env_steps += 1
                self.report.invalid_steps.append(
                    f"Unrecognised action string: {action_str!r}"
                )

        self.report.termination_reason = "max_steps"
        return 0

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        tool_call_message: Optional[str],
        error_message: Optional[str],
        tool_calls_exceeded: bool,
    ) -> str:
        """Construct the text portion of the VLM prompt for one iteration."""
        lines: List[str] = []

        lines.append(f"Task: {self._task}")
        lines.append("")
        lines.append(
            "You are playing a GameBoy game. The current screen is shown in the image."
        )
        lines.append("")

        if error_message is not None:
            lines.append(f"[ERROR] {error_message}")
            lines.append("")

        if tool_call_message is not None:
            lines.append(f"Tool result: {tool_call_message}")
            lines.append("")

        # Available env actions
        action_strings = self._get_action_strings()
        lines.append("Available environment actions:")
        for action_str in action_strings.values():
            lines.append(f"  {action_str}")
        lines.append("")

        # Available tools (suppressed once budget is exhausted)
        if not tool_calls_exceeded and self.available_tools:
            lines.append(
                "Available tool calls (do not advance the game, "
                f"{self._max_tool_calls} total budget):"
            )
            for tool_class in self.available_tools:
                lines.append(f"  {tool_class.verbalize()}")
            lines.append("")

        lines.append(
            "Reason about the best next action, then respond in exactly this format:"
        )
        lines.append("Reasoning: <your reasoning>")
        if not tool_calls_exceeded and self.available_tools:
            lines.append("Action: <one environment action OR one tool call>")
        else:
            lines.append("Action: <one environment action>")
        lines.append("[STOP]")

        return "\n".join(lines)

    def _parse_action(self, response: str) -> Optional[str]:
        """Extract the action string from a structured VLM response."""
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                action_str = stripped[len("action:"):].strip()
                action_str = action_str.replace("[STOP]", "").strip()
                return action_str if action_str else None
        return None

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
