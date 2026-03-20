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
from execution.report import EnvironmentStepRecord, ExecutorReport, ToolCallRecord
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

        self.report = ExecutorReport(
            task=task,
            init_kwargs=kwargs,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            initial_state=self._get_state(),
        )

        outcome = self._execute()
        self._vlm = ExecutorVLM(parameters=self._parameters)

        self.report.outcome = outcome
        self.report.final_state = self._get_state()

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
