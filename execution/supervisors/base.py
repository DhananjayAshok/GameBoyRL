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
from execution.report import ExecutorReport, SupervisorReport, SupervisorVLMCallRecord
from utils import load_parameters, log_error


class Supervisor(ABC):
    """
    Abstract base class for supervisor agents.

    A supervisor owns an executor class and an environment, and can dispatch
    task requests to a fresh executor instance on demand.  Subclasses implement
    :meth:`process_executor_return` to interpret the resulting report.

    :param task: The task this supervisor is responsible for. Held here rather than on each
        subclass because every supervisor has one and :attr:`report` records it.
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
        task: str,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._task = task
        self._executor_class = executor_class
        self._env = env
        self._game = game
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._parameters = load_parameters(parameters)
        self._executor_kwargs = executor_kwargs
        #: This run's record. Built here and appended to as the run proceeds, so it exists
        #: even if :meth:`evaluate` raises — a partial record of a failed episode is worth
        #: more than nothing.
        self.report = SupervisorReport(
            task=task,
            supervisor_name=self.__class__.__name__,
            game=game,
            init_kwargs=dict(executor_kwargs),
        )
        self._evaluated = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _vlm_call(self, stage: str, vlm, **kwargs: Any) -> Any:
        """Make a supervisor VLM call and record it on :attr:`report`.

        The supervisor-side counterpart of ``Executor._vlm_call``. Every supervisor call
        must go through here — that is the whole point, since a call made directly on a VLM
        leaves no trace of why the supervisor did what it did.

        ``stage`` is an argument rather than mutable state on a wrapper object. The previous
        design set a ``stage`` attribute on a recording proxy and then called it, so a site
        that forgot to set it logged under the *previous* stage, silently corrupting the
        only record of the supervisor's reasoning. Passing it with the call makes that
        mistake unavailable.

        A batched call passes a list of prompts and gets a list back; those are recorded as
        one entry per prompt/response pair, so a windowed judgement does not collapse into a
        single unreadable record.

        :param stage: Which phase of the supervisor's reasoning this call serves.
        :param vlm: The VLM to call.
        :return: Exactly what the VLM returned, unchanged.
        """
        result = vlm.infer(**kwargs)
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            # Per-prompt images when the caller batched them that way, else the shared set.
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                self.report.event_log.append(SupervisorVLMCallRecord(
                    stage=stage, images=call_images, prompt=prompt, response=response,
                ))
        else:
            self.report.event_log.append(SupervisorVLMCallRecord(
                stage=stage, images=images, prompt=texts, response=result,
            ))
        return result

    def _vlm_caller(self, stage: str, vlm):
        """A recording ``call(**kwargs)`` for helpers that do their own batching.

        :func:`~execution.supervisors.checker.window_trajectory` and friends build the
        prompt/image lists themselves and then make one batched call. They take this
        callable rather than a VLM so the call still lands in :attr:`report` — handing them
        a raw VLM is what used to lose those calls entirely.
        """
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, vlm, **kwargs)
        return call

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run this supervisor once and return its result.

        Always returns a dict carrying ``"report"`` (this run's
        :class:`~execution.report.SupervisorReport`) plus whatever arm-specific values
        :meth:`_evaluate` surfaced. Consumers read one or the other — never a bare report,
        so an arm can add a value without changing its return *type*.

        One run per instance. The report accumulates into ``event_log``, so a second call
        would splice two episodes into one record and every count derived from it would be
        the sum of both. No caller does this today (each construction site evaluates once),
        which is exactly why it would go unnoticed.
        """
        if self._evaluated:
            log_error(
                f"{self.__class__.__name__}.evaluate() called twice on one instance. The "
                "report accumulates, so this would merge two episodes into one record. "
                "Construct a new supervisor per episode.",
                self._parameters,
            )
        self._evaluated = True
        extras = self._evaluate() or {}
        return {"report": self.report, **extras}

    @abstractmethod
    def _evaluate(self) -> Optional[dict]:
        """Drive the episode. Return only arm-specific values, or None.

        :meth:`evaluate` attaches :attr:`report` to whatever this returns, so a subclass
        never has to remember to include it.
        """
        raise NotImplementedError

    def call_executor(self, task: str) -> Any:
        """
        Spin up an executor for the given task, run it to completion, record its report on
        :attr:`report`, then process and return the result.

        The report is filed into ``event_log`` before :meth:`process_executor_return` runs,
        so the executor's run is on record even if interpreting it raises.

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
        self.report.event_log.append(executor.report)
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



