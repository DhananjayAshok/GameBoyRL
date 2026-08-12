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

.. note:: **Variants that bypass this loop**

    Most executors customise behaviour by overriding the hooks below.  Exactly one does
    not: :class:`~execution.executors.planning.SequencePlannerExecutor` overrides
    :meth:`_execute` outright, because it runs a whole planned sequence per VLM call
    rather than one action.  Every other executor — including
    :class:`~execution.executors.deliberative.ActionValueEstimatorExecutor`, which
    customises :meth:`_query_vlm` and :meth:`_pick_action` — uses the base loop.

    Keep this note accurate: it exists so that "which executors bypass the base loop"
    is answerable from this file alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from gameboy_worlds.interface import Environment, HighLevelAction

from execution.executor_action import ExecutorAction
from execution.report import (EnvironmentStepRecord, ExecutorReport, InvalidStepRecord,
                              ExecutorToolCallRecord, ExecutorVLMCallRecord, parse_completion)
from utils import load_parameters, log_error, log_info, ExecutorVLM, parse_key_value

MAX_CONSECUTIVE_INVALID = 10
DEBUG_ON_INVALID = False


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
    :param allow_self_termination: When ``True``, an extra VLM call asks whether
        the task is now fully complete (see :meth:`_check_task_complete`).  A
        ``yes`` ends execution with outcome 3 and ``termination_reason ==
        "agent_done"``.  Defaults to ``False``, in which case nothing extra is
        called and the executor plays until the environment or the step budget
        stops it.

        How often the check runs is the executor's choice, made in its
        ``_execute`` loop: after every environment step by default, subject to
        :data:`DONE_CHECK_EVERY_K_STEPS`, but see
        :class:`SequencePlannerExecutor`, which checks once per committed plan.

        There is deliberately **no give-up mechanism**.  The check asks one
        question — is the task complete — and "this is hopeless" is not an
        answer to it.  An executor that cannot make progress runs to
        ``max_steps``.

        The action prompt is **identical** either way: nothing about
        termination is ever advertised to the acting model, so a run's prompts
        do not depend on this flag.
    :type allow_self_termination: bool
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

    #: Prompt for the post-step completion check (:meth:`_check_task_complete`).
    #:
    #: The verdict is asked for BEFORE the reasoning, which is the reverse of every other
    #: prompt here. This one is paid per environment step, so its token budget is the
    #: tightest in the codebase, and a model that narrates before answering (gemini writes a
    #: ``Reasoning:`` preamble of its own before the response proper) can spend the whole
    #: allowance without ever reaching the last line. Verdict-last means truncation costs the
    #: answer and keeps the explanation; verdict-first means it costs the explanation and
    #: keeps the answer. An unparsed verdict reads as "not complete", so the failure is
    #: silent: the check simply never fires.
    #: Two images are attached: the frame before the last action and the frame after it.
    DONE_CHECK_PROMPT = """Task: [TASK][HINT_BLOCK]

You are judging whether a task being played on a GameBoy has been FULLY completed.

Image 1 is the screen BEFORE the most recent action. Image 2 is the screen AFTER it.

Most recent action: [LAST_ACTION]
[REASONING_LABEL]
[LAST_REASONING]

[HISTORY_BLOCK]Decide whether the task as stated is now COMPLETELY accomplished — not partially, not nearly, not "the next step is obvious". If any part of the task remains to be done, the answer is no. If you cannot tell from what is visible, the answer is no.

Respond in exactly this format, with the verdict FIRST:
Complete: <yes or no>
Reasoning: <why, referring to what is visible in image 2>
[STOP]"""

    #: How the reasoning passed to the completion check is introduced. Overridden by
    #: executors whose stored reasoning is not step-level (see
    #: :class:`SequencePlannerExecutor`), so the prompt never misrepresents what the text is.
    DONE_CHECK_REASONING_LABEL = "The reasoning given for that action was:"

    #: Env actions shown to the completion check as history.
    DONE_CHECK_HISTORY_K = 8

    #: Run the completion check after every k-th environment step. 1 checks every step,
    #: which is the most responsive and the most expensive — it roughly doubles the VLM
    #: calls of a self-terminating run. Raising it to 2 or 3 halves or thirds that, at the
    #: cost of noticing completion up to k-1 steps late. The last permitted step is always
    #: checked whatever k is (see :meth:`_done_check_due`).
    DONE_CHECK_EVERY_K_STEPS = 1

    #: Token budget for the completion check, paid once per environment step. Not 300: a
    #: model that narrates before answering (gemini writes its own ``Reasoning:`` preamble)
    #: spends that entirely on prose and the reply is cut off before the verdict.
    #: :data:`DONE_CHECK_PROMPT` asks for the verdict first so truncation costs the
    #: explanation rather than the answer.
    DONE_CHECK_MAX_NEW_TOKENS = 1500

    def __init__(
        self,
        env: Environment,
        task: str,
        max_steps: int,
        max_tool_calls: int,
        game: str = "",
        parameters: Optional[dict] = None,
        vlm_model: Optional[str] = None,
        vlm_kind: Optional[str] = None,
        allow_self_termination: bool = False,
        hint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._parameters = load_parameters(parameters)
        # A blank task is never a real request, and it is silently destructive downstream:
        # ExecutorReport._save_images derives its output directory by slugifying the task,
        # and an empty slug collapses that path onto the executor directory, which it then
        # rmtree's. Refuse here rather than at the end of an episode that cost real tokens.
        if not task or not task.strip():
            log_error(
                f"Executor task must be a non-empty string, got {task!r}.",
                parameters=self._parameters,
            )
        self._env = env
        self._task = task
        self._hint = hint
        self._game = game
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._allow_self_termination = allow_self_termination
        if vlm_model is not None:
            self._parameters["executor_vlm_model"] = vlm_model
        if vlm_kind is not None:
            self._parameters["executor_vlm_kind"] = vlm_kind
        self._vlm = ExecutorVLM(parameters=self._parameters)
        self._max_new_tokens = self._parameters.get("executor_vlm_max_new_tokens", 512)
        self._last_frame_changed = True
        self._n_tool_calls = 0
        # The reasoning the acting model gave for the action it most recently chose, kept
        # for the completion check. Set by _pick_action; None until the first parse, and on
        # executors whose action call produces no reasoning at all.
        self._last_reasoning: Optional[str] = None
        # The VLM call currently being acted on. Set by _vlm_call, read by _record_step so
        # every step is filed under the call that caused it. None before the first call.
        self._current_call: Optional[ExecutorVLMCallRecord] = None

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
            executor_name=self.__class__.__name__,
            game=self._game,
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
          it via :meth:`_record_step`, which files it under the current VLM call.
        - Return the record.

        :return: The record of the environment step that was taken.
        :rtype: EnvironmentStepRecord
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete helpers — uniform across all executors
    # ------------------------------------------------------------------

    def _record_step(self, step) -> None:
        """File a step under the VLM call that caused it.

        The single write path for every step an executor takes. Ownership is recorded
        here, at the moment the step happens, rather than reconstructed later by pairing
        two lists — which is what used to go wrong whenever a call produced anything
        other than exactly one step.

        :param step: An :class:`~execution.report.EnvironmentStepRecord`,
            :class:`~execution.report.ExecutorToolCallRecord` or
            :class:`~execution.report.InvalidStepRecord`.
        """
        if self._current_call is None:
            log_error(
                f"{type(step).__name__} recorded before any VLM call was made, so it has "
                "no owning call. Every step must follow the _vlm_call that decided it.",
                parameters=self._parameters,
            )
        self._current_call.steps.append(step)

    def _record_invalid(self, response: str, reason: str = "parse failure") -> None:
        """Record an invalid step and trigger a breakpoint if DEBUG_ON_INVALID is set.

        Files an :class:`InvalidStepRecord` under the current call, so a call whose reply
        was unusable is recorded as having produced *that* rather than nothing at all —
        which is what keeps a failed parse visible in the report and countable in
        :attr:`~execution.report.ExecutorReport.invalid_steps`.

        :param reason: ``"parse failure"`` or ``"unrecognised action"``.
        """
        self._record_step(InvalidStepRecord(response=response, reason=reason))
        log_info(f"Invalid response recorded: \n{response}", parameters=self._parameters)
        if DEBUG_ON_INVALID:
            breakpoint()

    def _use_tool(
        self,
        executor_action_class: Type[ExecutorAction],
        **kwargs: Any,
    ) -> ExecutorToolCallRecord:
        """
        Invoke a passive :class:`~execution.executor_action.ExecutorAction` and
        record the result.

        The action receives the current environment state via
        :meth:`_get_state` but does **not** advance the emulator.  The
        resulting :class:`~execution.report.ExecutorToolCallRecord` is filed under the current
        VLM call (see :meth:`_record_step`) before being returned.

        :param executor_action_class: The
            :class:`~execution.executor_action.ExecutorAction` subclass to
            invoke.
        :type executor_action_class: Type[ExecutorAction]
        :param kwargs: Additional keyword arguments forwarded to
            :meth:`~execution.executor_action.ExecutorAction.execute`.
        :return: The record of the tool call that was made.
        :rtype: ExecutorToolCallRecord
        """
        action = executor_action_class()
        result, success_code = action.execute(info=self._get_state(), **kwargs)

        record = ExecutorToolCallRecord(
            executor_action_class=executor_action_class,
            kwargs=kwargs,
            result=result,
            success_code=success_code,
        )
        self._record_step(record)
        self._n_tool_calls += 1
        return record

    def _hint_block(self) -> str:
        if self._hint is None:
            return ""
        # Steps taken so far == non-tool records in report.steps (tool calls don't
        # advance the env / n_env_steps). This is the step the agent is about to take.
        steps_taken = sum(1 for s in self.report.steps if not isinstance(s, ExecutorToolCallRecord))
        step_info = f"""
[STEP_INFO] You have already taken {steps_taken + 1} actions so for this attempt. Note: this may not be the first step of the overall task and one action does not correspond to one step in the hint plan — earlier actions may already have been taken before this attempt began, so reason from what you currently see on screen rather than assuming a fresh start. [STEP_INFO_END]"""
        return step_info + f"\n[HINT_START]\nHint: {self._hint}\nNote: This hint block is a secret. You must use it to guide your decision making, but in the reasoning you say, you should pretend as if you actually just know the content of the hint. Do not refer to it explicitly. So if the hint gives you a direction, instead of saying 'the hint says go here', your reasoning should just say 'next I must go here'. [HINT_END]"

    def _get_state(self) -> dict:
        """
        Return the current environment state.

        :return: State info dictionary from
            :meth:`~gameboy_worlds.interface.Environment.get_info`.
        :rtype: dict
        """
        return self._env.get_info()

    def _vlm_call(self, tag: str, **kwargs: Any):
        """
        Invoke the VLM, log the call, and make it the owner of any steps that follow.

        This is the **only** way executors should call the VLM — never call
        ``self._vlm.infer`` directly.  All keyword arguments are forwarded
        verbatim to :meth:`~utils.vlm.ExecutorVLM.infer`.

        Exactly **one** :class:`~execution.report.ExecutorVLMCallRecord` is appended per call,
        and it becomes :attr:`_current_call`.  Every step recorded afterwards —
        :meth:`_take_action`, :meth:`_use_tool`, :meth:`_record_invalid` — is filed
        under it, so a call owns precisely what it caused.  A later ``_vlm_call``
        takes ownership from here on.

        ``n_outputs > 1`` is **not supported**: several sampled responses to one call
        have no single owner for the resulting step, and logging one record per sample
        is what made an executor's calls and steps drift apart.  Ask once, or make each
        sample its own ``_vlm_call``.

        :param tag: Short label for the call's role, e.g. ``"action"``,
            ``"reflection"``, ``"map_update"``, ``"decompose"``, ``"score"``,
            ``"done_check"``.

            .. important:: If this call is the one that decides the action, its tag must
                be in :data:`~execution.report.ACTION_TAGS`. Ownership of the step that
                follows goes to the **last** call made, and the consumers that
                reconstruct an action sequence filter on that set — so an action decided
                in an auxiliary call is silently dropped from the dataset.
        :type tag: str
        :param kwargs: Keyword arguments forwarded to
            :meth:`~utils.vlm.ExecutorVLM.infer`.
        :return: The raw VLM output string.
        """
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        result = self._vlm.infer(**kwargs)
        texts = kwargs["texts"]
        images = kwargs["images"]
        assert isinstance(texts, str)
        if isinstance(result, list):
            log_error(
                f"_vlm_call({tag!r}) received {len(result)} responses. Multi-sample calls "
                "are unsupported: the steps that follow would have no single owning call, "
                "which is how VLM calls and steps came to be mis-paired. Drop n_outputs, "
                "or issue one _vlm_call per sample.",
                parameters=self._parameters,
            )
        record = ExecutorVLMCallRecord(tag=tag, prompt=texts, images=images, response=result)
        self.report.vlm_call_log.append(record)
        self._current_call = record
        return result

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

    # ------------------------------------------------------------------
    # Self-termination — the post-step completion check
    # ------------------------------------------------------------------

    def _parse_reasoning(self, response: str) -> Optional[str]:
        """The ``Reasoning:`` line of an action response, or ``None``."""
        return parse_key_value(response, "Reasoning")

    def _recent_actions_block(self, k: Optional[int] = None) -> str:
        """
        The last *k* environment actions, oldest first, for the completion check.

        Derived from :attr:`report.steps` rather than from any subclass's own bookkeeping,
        so every executor gets history here without duplicating
        :class:`HistoryAwareExecutor`'s state. That class keeps its own list for its own
        prompt; the two serve different callers and are deliberately not merged.
        """
        k = self.DONE_CHECK_HISTORY_K if k is None else k
        env_steps = [s for s in self.report.steps if isinstance(s, EnvironmentStepRecord)]
        if not env_steps:
            return ""
        lines = ["Recent actions (oldest first, the last one is the action judged above):"]
        for step in env_steps[-k:]:
            lines.append(f"  {step.action_class.get_action_name(**step.kwargs)}")
        return "\n".join(lines) + "\n\n"

    def _build_done_check_prompt(self, record: EnvironmentStepRecord) -> str:
        last_action = record.action_class.get_action_name(**record.kwargs)
        reasoning = self._last_reasoning or "(no reasoning was recorded for this action)"
        return (
            self.DONE_CHECK_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[LAST_ACTION]", last_action)
            .replace("[REASONING_LABEL]", self.DONE_CHECK_REASONING_LABEL)
            .replace("[LAST_REASONING]", reasoning)
            .replace("[HISTORY_BLOCK]", self._recent_actions_block())
        )

    def _check_task_complete(self, record: EnvironmentStepRecord) -> bool:
        """
        Ask the VLM whether the task is now fully complete.

        One call, tagged ``"done_check"``, showing the frames either side of *record* along
        with the task, the hint, the action just taken, the reasoning that chose it and the
        recent action history. Consumes no environment step and no tool budget.

        The judge is the executor's own model — the same :attr:`_vlm`, so it follows the
        ``vlm_model`` constructor override. Judging with a stronger model than the one
        acting would make ``agent_done`` mean something different per run and stop
        self-terminated episodes being comparable across the benchmark.

        The response format is owned by :func:`~execution.report.parse_completion`, not by
        this class — the report renderer and the plan supervisor read the same verdict off
        the call log and must agree with the decision made here.

        :param record: The step record just appended by :meth:`_take_action`.
        :return: ``True`` only on an explicit ``Complete: yes``.
        """
        prompt = self._build_done_check_prompt(record)
        response = self._vlm_call(
            "done_check",
            texts=prompt,
            images=[record.frame_before, record.frame_after],
            max_new_tokens=self.DONE_CHECK_MAX_NEW_TOKENS,
        )
        verdict = parse_completion(response)
        if verdict is None:
            # Logged, not recorded via _record_invalid: that list and MAX_CONSECUTIVE_INVALID
            # are about the *acting* model's formatting, and a judge's bad formatting must
            # not push an executor toward max_invalid. This is the only site that can tell
            # "said no" from "did not answer" apart, so it is the only one that reports it.
            log_info(
                f"Completion check response had no parseable 'Complete:' line, treating as "
                f"not complete:\n{response}",
                parameters=self._parameters,
            )
        return verdict is True

    def _maybe_self_terminate(self, record: EnvironmentStepRecord, n_env_steps: int) -> Optional[int]:
        """
        Run the completion check and end the run if it says the task is done.

        Call from ``_execute`` immediately after a successful environment step, **after**
        the environment's own ``terminated`` / ``truncated`` branches, passing the loop's
        own budget counter::

            outcome = self._maybe_self_terminate(record, n_env_steps)
            if outcome is not None:
                return outcome

        The ordering is not cosmetic: the environment's verdict is ground truth and the
        agent's is an opinion, so when both fire on the same step the ground truth is what
        gets recorded. Reversing them would quietly turn benchmark success rates into agent
        self-assessments.

        Because the check only ever runs after a step has been taken, a zero-step
        ``agent_done`` is impossible by construction.

        The check is skipped on steps that are not due one — see
        :meth:`_done_check_due`, which implements :data:`DONE_CHECK_EVERY_K_STEPS`.

        :param record: The step record just appended by :meth:`_take_action`.
        :param n_env_steps: The calling loop's budget counter, *after* it was incremented
            for this step. This is the quantity ``max_steps`` bounds, and it is not the
            same as the number of environment actions taken — see :meth:`_done_check_due`.
        :return: ``3`` when the check says the task is complete, ``None`` otherwise.
        :rtype: Optional[int]
        """
        if not self._allow_self_termination:
            return None
        if not self._done_check_due(n_env_steps):
            return None
        if not self._check_task_complete(record):
            return None
        self.report.termination_reason = "agent_done"
        return 3

    def _done_check_due(self, n_env_steps: int) -> bool:
        """Whether this step is one the completion check runs on.

        The check is the most expensive thing in an episode: one two-image VLM call per
        environment step, so it roughly doubles the call count of any run with
        ``allow_self_termination`` set. :data:`DONE_CHECK_EVERY_K_STEPS` trades latency of
        detection for that cost — at *k* the run notices it has finished up to *k-1* steps
        late, and pays a *k*-th of the checks.

        **Two different counters, deliberately.** *k* counts environment actions actually
        dispatched (``EnvironmentStepRecord``s), because that is what the check costs money
        against — a step the agent burned on an unparseable response produced no new frame
        for a judge to look at. The step *budget*, though, is spent by invalid steps too:
        every ``_execute`` loop increments its ``n_env_steps`` on a parse failure and an
        unrecognised action as well as on a real action. So "is this the last permitted
        step" can only be answered by the budget counter the loop itself is running on,
        which is why it is passed in rather than recomputed here.

        The final permitted step is **always** checked regardless of *k*. Without that, a
        budget that is not a multiple of *k* would end with its last steps unexamined, and
        an executor that finished on one of them would run out its budget and report
        ``max_steps`` — which reads as failure. That is the one moment where a missed check
        cannot be recovered later, so it is never the one that gets skipped. Testing that
        against the action count instead would break the guarantee outright: after a single
        invalid step the action count trails the budget permanently and never reaches
        ``max_steps``, so the last step would go unchecked in exactly the runs the
        guarantee exists for.

        :param n_env_steps: The calling loop's budget counter, after it was incremented for
            this step. Compared against ``max_steps``; never used for the *k* cadence.
        :return: ``True`` when the check should run after the step just taken.
        :rtype: bool
        """
        k = max(1, self.DONE_CHECK_EVERY_K_STEPS)
        if k == 1:
            return True
        actions_taken = sum(1 for step in self.report.steps
                            if isinstance(step, EnvironmentStepRecord))
        return actions_taken % k == 0 or n_env_steps >= self._max_steps


