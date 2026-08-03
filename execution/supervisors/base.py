"""
The supervisor contract.

A supervisor wraps one or more executor runs: it constructs the executor, lets it
play, and then does something with the report it produces — judge it, critique it,
turn it into a hint, or drive the next step of a plan.

:class:`Supervisor` fixes only that shape.  The four concrete supervisors share the
base class and almost nothing else; each lives in its own module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport, iter_call_steps
from utils import load_parameters, VLM


class Supervisor(ABC):
    """
    Abstract base class for supervisor agents.

    A supervisor owns an executor class and an environment, and can dispatch
    task requests to a fresh executor instance on demand.  Subclasses implement
    :meth:`process_executor_return` to interpret the resulting report.

    :param executor_class: The :class:`~execution.executors.Executor` subclass to use.
    :param env: The game environment passed to each executor call.
    :param game: Game name string, forwarded to the executor.
    :param max_steps: Step budget forwarded to each executor.
    :param max_tool_calls: Tool-call budget forwarded to each executor.
    :param parameters: Optional parameter overrides.
    :param executor_kwargs: Additional keyword arguments forwarded verbatim to
        the executor constructor (e.g. ``vlm_model``, ``allow_self_termination``).
    """

    def __init__(
        self,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._executor_class = executor_class
        self._env = env
        self._game = game
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._parameters = load_parameters(parameters)
        self._executor_kwargs = executor_kwargs


    def call_executor(self, task: str) -> Any:
        """
        Spin up an executor for the given task, run it to completion, then
        process and return the result.

        :param task: Natural-language task string passed to the executor.
        :return: Whatever :meth:`process_executor_return` returns.
        """
        executor = self._executor_class(
            env=self._env,
            task=task,
            game=self._game,
            max_steps=self._max_steps,
            max_tool_calls=self._max_tool_calls,
            parameters=self._parameters,
            **self._executor_kwargs,
        )
        return self.process_executor_return(executor.report)

    @abstractmethod
    def process_executor_return(self, report: ExecutorReport) -> Any:
        """
        Process the report produced by a completed executor run.

        :param report: The :class:`~execution.report.ExecutorReport` sealed by
            the executor after :meth:`~execution.executors.Executor._execute` returns.
        :return: Any result the subclass wants to surface to the caller.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Parse helpers (checker-local, no vlm_scripts dependency)
# ---------------------------------------------------------------------------



def _frame_to_call_cutoff(
    vlm_call_log: list,
    steps: list,
    safe_frame: Optional[int],
) -> Optional[int]:
    """Convert a 1-based env-frame number into a vlm_call_log slice index.

    The judge VLM reports ``safe_success_point`` as a *frame number* — the
    earliest env frame by which the task is surely complete. Downstream
    (create_dataset.py) we only have the per-episode vlm_call_log, not ``steps``,
    so we resolve the frame→call mapping here, while both lists are in hand, and
    return the number of leading vlm_call_log entries to keep.

    NOTE: the returned value is what gets stored under the ``safe_success_point``
    key (see process_executor_return) — i.e. that key carries a *call-log index*,
    NOT the original frame number. The frame number is intentionally not
    preserved.

    Mirrors the lockstep walk in report.ExecutorReport.__str__: only calls tagged
    in ``ACTION_TAGS`` consume a step, and only an ``EnvironmentStepRecord`` step
    advances a frame (tool calls / invalid actions consume a call without
    producing a frame). Returns ``None`` (no truncation downstream) when
    ``safe_frame`` is None — the judge couldn't pin down a completion frame.
    """
    if safe_frame is None:
        return None
    env_frames = 0
    for call_idx, entry, step in iter_call_steps(vlm_call_log, steps):
        if isinstance(step, EnvironmentStepRecord):
            env_frames += 1
            if env_frames >= safe_frame:
                return call_idx + 1  # keep through the call that produced this frame
    return len(vlm_call_log)


# ---------------------------------------------------------------------------
# SimpleCheckerSupervisor
# ---------------------------------------------------------------------------


