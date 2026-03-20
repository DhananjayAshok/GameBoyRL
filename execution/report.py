"""
Data structures for recording a complete executor run.

:class:`ToolCallRecord` captures a single passive tool call (no emulator step).
:class:`EnvironmentStepRecord` captures a single high-level environment step.
:class:`ExecutorReport` aggregates the full run history produced by one :class:`~execution.executor.Executor` invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from gameboy_worlds.interface import HighLevelAction

from execution.executor_action import ExecutorAction


@dataclass
class ToolCallRecord:
    """
    Record of a single passive tool call made by an executor.

    Tool calls do **not** advance the emulator, so no state snapshot is stored
    here — the state is identical before and after a tool call.

    :param executor_action_class: The :class:`~execution.executor_action.ExecutorAction`
        subclass that was invoked.
    :type executor_action_class: Type[ExecutorAction]
    :param kwargs: Keyword arguments forwarded to
        :meth:`~execution.executor_action.ExecutorAction.execute`.
    :type kwargs: dict
    :param result: Return dictionary from
        :meth:`~execution.executor_action.ExecutorAction._execute`, or ``None``
        if the action arguments failed validation.
    :type result: Optional[dict]
    :param success_code: Integer success code returned alongside ``result``, or
        ``None`` if the action arguments failed validation.
    :type success_code: Optional[int]
    """

    executor_action_class: Type[ExecutorAction]
    kwargs: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    success_code: Optional[int]


@dataclass
class EnvironmentStepRecord:
    """
    Record of a single high-level environment step taken by an executor.

    Each step corresponds to one call to
    :meth:`~gameboy_worlds.interface.Environment.step_high_level_action` or
    :meth:`~gameboy_worlds.interface.Environment.step_str`.

    :param action_class: The :class:`~gameboy_worlds.interface.HighLevelAction`
        subclass that was executed. Stored as the **class**, not an instance.
    :type action_class: Type[HighLevelAction]
    :param kwargs: Keyword arguments passed to the high-level action.
    :type kwargs: dict
    :param transition_states: Ordered list of state snapshots (one per
        low-level emulator step) produced during this high-level action.
        Each entry is a ``state_tracker.report()`` dict.
    :type transition_states: List[dict]
    :param action_success: Integer success code returned by the high-level
        action.
    :type action_success: int
    """

    action_class: Type[HighLevelAction]
    kwargs: Dict[str, Any]
    transition_states: List[Dict[str, Any]]
    action_success: int


@dataclass
class ExecutorReport:
    """
    Complete record of a single executor run.

    Produced and sealed entirely by :class:`~execution.executor.Executor.__init__`.
    Subclasses never build or overwrite this object.

    :param task: Natural-language task description given to the executor.
    :type task: str
    :param init_kwargs: Subclass-specific keyword arguments captured at
        ``__init__`` time (excludes ``env``, ``task``, ``max_steps``,
        ``max_tool_calls``, and ``parameters``).
    :type init_kwargs: dict
    :param max_steps: Maximum number of environment steps the executor was
        permitted to take.
    :type max_steps: int
    :param max_tool_calls: Maximum number of tool calls the executor was
        permitted to make.
    :type max_tool_calls: int
    :param initial_state: State snapshot taken immediately before
        :meth:`~execution.executor.Executor._execute` is called.
    :type initial_state: dict
    :param steps: Interleaved, time-ordered list of
        :class:`ToolCallRecord` and :class:`EnvironmentStepRecord` objects
        produced during the run.
    :type steps: List[Union[ToolCallRecord, EnvironmentStepRecord]]
    :param final_state: State snapshot taken immediately after
        :meth:`~execution.executor.Executor._execute` returns.
        Set to ``None`` until execution completes.
    :type final_state: Optional[dict]
    :param outcome: Executor-defined integer outcome code.
        ``None`` until :meth:`~execution.executor.Executor._execute` returns.
    :type outcome: Optional[int]
    :param notes: Optional freeform commentary written by the executor.
    :type notes: Optional[str]
    """

    task: str
    init_kwargs: Dict[str, Any]
    max_steps: int
    max_tool_calls: int
    initial_state: Dict[str, Any]
    steps: List[Union[ToolCallRecord, EnvironmentStepRecord]] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    outcome: Optional[int] = None
    notes: Optional[str] = None
