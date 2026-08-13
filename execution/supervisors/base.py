"""
The supervisor contract.

A supervisor wraps one or more executor runs: it constructs the executor, lets it
play, and then does something with the report it produces — judge it, critique it,
turn it into a hint, or drive the next step of a plan.

:class:`Supervisor` fixes only that shape.  The three concrete supervisors share the
base class and almost nothing else; each lives in its own module.

**One model per supervisor.**  A supervisor makes every one of its calls against a single
VLM, held here and reached through :meth:`Supervisor._vlm_call`.  Subclasses used to each
declare their own — ``checker_vlm_model``, ``hint_vlm_model``, ``plan_vlm_model`` — which
meant a caller wiring up an arm had to know which stage names which model, and the split
was never actually used: the only caller passed the same name to all of them.  There are
now exactly two models in play anywhere: the executor's, and the supervisor's.

The model that *built* the knowledge a supervisor reads is deliberately not a third: it is
a property of the artifact, recorded in :class:`~execution.info_doc.Provenance` by whoever
produced it, and no supervisor or benchmark arm takes it as a parameter.
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
    task requests to a fresh executor instance on demand.  Subclasses implement
    :meth:`process_executor_return` to interpret the resulting report.

    :param task: The task this supervisor is responsible for. Held here rather than on each
        subclass because every supervisor has one and :attr:`report` records it.
    :param executor_class: The :class:`~execution.executors.Executor` subclass to use.
    :param env: The game environment passed to each executor call.
    :param game: Game name string, forwarded to the executor.
    :param max_steps: Step budget forwarded to each executor.
    :param max_tool_calls: Tool-call budget forwarded to each executor.
    :param supervisor_vlm_model: The one model this supervisor reasons with. ``None`` for a
        supervisor that never calls one (the baseline); constructing the VLM is deferred, so
        a supervisor that does not reason does not need a model to exist.
    :param supervisor_vlm_kind: VLM kind for that model.
    :param max_new_tokens: Token budget for every supervisor call. One number rather than one
        per stage: the stages used to differ (1000 / 2400 / 4800), and the only thing that
        difference ever bought was a silent truncation when a reply outgrew its stage's
        allowance — the completion check in particular puts its verdict on the last line, so
        losing the tail reads as "not complete" and every step fails. Set to the largest of
        the old stage budgets, so no call is tighter than it was.
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
        self._max_tool_calls = max_tool_calls
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

        This used to be ``dict(executor_kwargs)`` — the *executor's* arguments, stored under
        a name that says supervisor. The effect was that none of the settings that actually
        define an arm's behaviour (``max_leg_steps``, ``max_attempts_per_target``,
        ``max_replans``, …) appeared anywhere in the archived report, so a saved episode
        could not be matched to the configuration that produced it.

        Subclasses extend it. Safe to call from ``__init__`` because every subclass sets its
        own attributes *before* calling ``super().__init__()``.
        """
        return {
            "supervisor_vlm_model": self._supervisor_vlm_model,
            "supervisor_vlm_kind": self._supervisor_vlm_kind,
            "max_new_tokens": self._max_new_tokens,
            "max_steps": self._max_steps,
            "max_tool_calls": self._max_tool_calls,
            "executor_kwargs": dict(self._executor_kwargs),
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def _vlm(self) -> VLM:
        """This supervisor's one model, built on first use.

        Deferred rather than constructed in ``__init__`` because
        :class:`~execution.supervisors.dummy.DummySupervisor` makes no calls at all and must
        keep working with no model configured — it is the control arm, and requiring it to
        name a model it never uses would be a way to accidentally give it one.

        Raises through :func:`log_error` rather than returning ``None``, so a supervisor that
        *does* reason fails at the point the model is missing instead of somewhere later
        inside an inference call with a less obvious message.
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

        The supervisor-side counterpart of ``Executor._vlm_call``. Every supervisor call
        must go through here — that is the whole point, since a call made directly on a VLM
        leaves no trace of why the supervisor did what it did.

        The model is :attr:`_vlm` and the token budget defaults to
        :attr:`_max_new_tokens`, so neither is repeated at the call sites. A caller may still
        pass ``max_new_tokens`` explicitly to override it for one call.

        ``stage`` is an argument rather than mutable state on a wrapper object. The previous
        design set a ``stage`` attribute on a recording proxy and then called it, so a site
        that forgot to set it logged under the *previous* stage, silently corrupting the
        only record of the supervisor's reasoning. Passing it with the call makes that
        mistake unavailable.

        A batched call passes a list of prompts and gets a list back; those are recorded as
        one entry per prompt/response pair, so a windowed judgement does not collapse into a
        single unreadable record.

        :param stage: Which phase of the supervisor's reasoning this call serves.
        :return: Exactly what the VLM returned, unchanged.
        """
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        """Make the call and *return* its records instead of filing them.

        The half of :meth:`_vlm_call` that can run off the main thread. A worker in a
        :class:`~concurrent.futures.ThreadPoolExecutor` calls this and hands the records
        back; the caller appends them once the pool has joined, in whatever order it
        chooses. :meth:`_vlm_call` is then just this plus the append.

        Split out because the parallel relevance pass in
        :class:`~execution.supervisors.info_subgoal.InfoSubgoalSupervisor` appended to
        :attr:`report.event_log` straight from its workers, so the records landed in
        *completion* order — which made the event log non-reproducible across reruns of the
        same episode, and undermined the property that the log's ordering is the only
        per-leg labelling there is.

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

        :func:`~execution.supervisors.checker.window_trajectory` and friends build the
        prompt/image lists themselves and then make one batched call. They take this
        callable rather than a VLM so the call still lands in :attr:`report` — handing them
        a raw VLM is what used to lose those calls entirely.
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

    def call_executor(self, task: str, *, hint: Optional[str] = None,
                      allow_self_termination: bool = False,
                      max_steps: Optional[int] = None) -> Any:
        """
        Spin up an executor for the given task, run it to completion, record its report on
        :attr:`report`, then process and return the result.

        The report is filed into ``event_log`` before :meth:`process_executor_return` runs,
        so the executor's run is on record even if interpreting it raises.

        **The three things that vary per leg are parameters, not state.** They used to be
        set on the supervisor immediately before this call —
        ``self._executor_kwargs["hint"] = ...`` and a temporary overwrite of
        ``self._max_steps``, an attribute that otherwise means the whole episode's budget.
        Passing them here is what lets the executor record what it actually ran under (see
        :meth:`~execution.executors.Executor._run_config`), and removes a
        write-then-restore dance that had to be exception-safe to be correct.

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
            max_tool_calls=self._max_tool_calls,
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



