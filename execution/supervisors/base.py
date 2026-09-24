"""
The supervisor contract.

A supervisor wraps one or more executor runs: it constructs the executor, lets it
play, and then does something with the report it produces — judge it, critique it,
turn it into a hint, or drive the next step of a plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import (ExecutorReport, SupervisorReport, SupervisorVLMCallRecord,
                              per_prompt_token_counts)
from utils import VLM, load_parameters, log_error


class Supervisor(ABC):
    """
    Abstract base class for supervisor agents.

    A supervisor owns an executor class and an environment, and can dispatch
    task requests to a fresh executor instance on demand. 

    :param task: The task this supervisor is responsible for.
    :param executor_class: The :class:`~execution.executors.Executor` subclass to use.
    :param env: The game environment passed to each executor call.
    :param game: Game name string, forwarded to the executor.
    :param max_steps: Step budget forwarded to each executor.
    :param supervisor_vlm_model: The one model this supervisor reasons with. ``None`` for a
        supervisor that never calls one; construction is deferred.
    :param supervisor_vlm_kind: VLM kind for that model.
    :param max_new_tokens: Token budget for every supervisor call, one number for all stages.
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
        supervisor_vlm_model: Optional[str] = None,
        supervisor_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._task = task
        self._executor_class = executor_class
        self._env = env
        self._game = game
        self._max_steps = max_steps
        self._supervisor_vlm_model = supervisor_vlm_model
        self._supervisor_vlm_kind = supervisor_vlm_kind
        self._max_new_tokens = max_new_tokens
        self._vlm_instance: Optional[VLM] = None
        self._parameters = load_parameters(parameters)
        self._executor_kwargs = executor_kwargs
        #: This run's record. Built here and appended to as the run proceeds, so it exists
        #: even if :meth:`evaluate` raises — a partial record of a failed episode is worth
        #: more than nothing.
        self.report = SupervisorReport(
            task=task,
            supervisor_name=self.__class__.__name__,
            game=game,
            init_kwargs=self._run_config(),
        )
        self._evaluated = False

    def _run_config(self) -> dict:
        """The knobs this supervisor is running with, for :attr:`report.init_kwargs`.
        Subclasses extend it, and set their own attributes before calling
        ``super().__init__()`` so this is safe to call from there.

        :return: The config.
        :rtype: dict
        """
        return {
            "supervisor_vlm_model": self._supervisor_vlm_model,
            "supervisor_vlm_kind": self._supervisor_vlm_kind,
            "max_new_tokens": self._max_new_tokens,
            "max_steps": self._max_steps,
            "executor_kwargs": dict(self._executor_kwargs),
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def _vlm(self) -> VLM:
        """This supervisor's one model, built on first use, so a supervisor that makes no
        calls needs no model configured.

        :return: The VLM.
        :rtype: VLM
        """
        if self._vlm_instance is None:
            if not self._supervisor_vlm_model:
                log_error(
                    f"{self.__class__.__name__} tried to make a VLM call but no "
                    f"supervisor_vlm_model was given. Pass one, or use DummySupervisor if "
                    f"the arm is meant to do no reasoning.",
                    self._parameters,
                )
            self._vlm_instance = VLM(self._supervisor_vlm_model, self._supervisor_vlm_kind)
        return self._vlm_instance

    def _vlm_call(self, stage: str, **kwargs: Any) -> Any:
        """Make a supervisor VLM call and record it on :attr:`report`.

        :param stage: Which phase of the supervisor's reasoning this call serves.
        :return: Exactly what the VLM returned, unchanged.
        """
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        """Make the call and *return* its records instead of filing them.

        The half of :meth:`_vlm_call` that can run off the main thread: a pool worker calls
        this and hands the records back, and the caller appends them once the pool has joined
        so the event log stays in a deterministic order.

        :return: ``(result, records)`` — the VLM's output unchanged, and the
            :class:`~execution.report.SupervisorVLMCallRecord` list it should be filed under.
        """
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        inferred = self._vlm.infer(**kwargs)
        result, meta = inferred["output"], inferred["meta"]
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        records = []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta, len(texts))
            # Per-prompt images when the caller batched them that way, else the shared set.
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                records.append(SupervisorVLMCallRecord(
                    stage=stage, images=call_images, prompt=prompt, response=response,
                    input_tokens=token_counts[index][0],
                    output_tokens=token_counts[index][1],
                ))
        else:
            # A single prompt is one record, so meta's counts are scalars.
            records.append(SupervisorVLMCallRecord(
                stage=stage, images=images, prompt=texts, response=result,
                input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"],
            ))
        return result, records

    def _vlm_caller(self, stage: str):
        """A recording ``call(**kwargs)`` for helpers that do their own batching.
        """
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, **kwargs)
        return call

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run this supervisor once and return its result.

        Always returns a dict carrying ``"report"`` (this run's
        :class:`~execution.report.SupervisorReport`) plus whatever arm-specific values
        :meth:`_evaluate` surfaced. 
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
        """
        raise NotImplementedError

    def call_executor(self, task: str, *, hint: Optional[str] = None,
                      allow_self_termination: bool = False,
                      max_steps: Optional[int] = None) -> Any:
        """
        Spin up an executor for the given task, run it to completion, record its report on
        :attr:`report`, then process and return the result.

        :param task: Natural-language task string passed to the executor.
        :param hint: Advice for this leg, or ``None`` for an unhinted run.
        :param allow_self_termination: Whether this leg may end itself by declaring the task
            complete. Off for any leg whose success is the environment's to signal.
        :param max_steps: Step budget for this leg. Defaults to the episode's, which is what
            a supervisor running exactly one leg wants.
        :return: Whatever :meth:`process_executor_return` returns.
        """
        run_kwargs = dict(self._executor_kwargs)
        run_kwargs["hint"] = hint
        run_kwargs["allow_self_termination"] = allow_self_termination
        executor = self._executor_class(
            env=self._env,
            task=task,
            game=self._game,
            max_steps=self._max_steps if max_steps is None else max_steps,
            parameters=self._parameters,
            **run_kwargs,
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



