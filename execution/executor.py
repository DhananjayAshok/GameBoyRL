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
from gameboy_worlds.interface.action import LowLevelAction

from execution.executor_action import ExecutorAction
from execution.report import (EnvironmentStepRecord, ExecutorReport, InvalidStepRecord, SimpleReport,
                              ToolCallRecord, VLMCallRecord, parse_completion)
from utils import load_parameters, log_info, ExecutorVLM, parse_key_value

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
        self._env = env
        self._task = task
        self._hint = hint
        self._game = game
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._allow_self_termination = allow_self_termination
        self._parameters = load_parameters(parameters)
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
          it to ``self.report.steps``.
        - Return the record.

        :return: The record of the environment step that was taken.
        :rtype: EnvironmentStepRecord
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete helpers — uniform across all executors
    # ------------------------------------------------------------------

    def _record_invalid(self, response: str, reason: str = "parse failure") -> None:
        """Append an invalid step and trigger a breakpoint if DEBUG_ON_INVALID is set.

        Records both in ``invalid_steps`` (for counts/back-compat) and as an
        ordered :class:`InvalidStepRecord` in ``steps``, so ``steps`` stays a
        complete 1:1 log of every action call's outcome and the report renderer
        can pair calls to steps without guessing.

        :param reason: ``"parse failure"`` or ``"unrecognised action"``.
        """
        self.report.invalid_steps.append(response)
        self.report.steps.append(InvalidStepRecord(response=response, reason=reason))
        log_info(f"Invalid response recorded: \n{response}", parameters=self._parameters)
        if DEBUG_ON_INVALID:
            breakpoint()

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
        self._n_tool_calls += 1
        return record

    def _hint_block(self) -> str:
        if self._hint is None:
            return ""
        # Steps taken so far == non-tool records in report.steps (tool calls don't
        # advance the env / n_env_steps). This is the step the agent is about to take.
        steps_taken = sum(1 for s in self.report.steps if not isinstance(s, ToolCallRecord))
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
        Invoke the VLM and log every response to :attr:`report.vlm_call_log`.

        This is the **only** way executors should call the VLM — never call
        ``self._vlm.infer`` directly.  All keyword arguments are forwarded
        verbatim to :meth:`~utils.vlm.ExecutorVLM.infer`.

        When ``n_outputs > 1`` the VLM returns a :class:`list` of strings;
        each element is logged as a separate :class:`~execution.report.VLMCallRecord`
        with the same *tag*.  The raw return value (``str`` or ``List[str]``)
        is returned unchanged so callers can use it as before.

        :param tag: Short label for the call's role, e.g. ``"action"``,
            ``"reflection"``, ``"map_update"``, ``"belief_update"``,
            ``"decompose"``, ``"score"``, ``"rethink"``, ``"propose"``,
            ``"challenge"``, ``"decide"``, ``"done_check"``.
        :type tag: str
        :param kwargs: Keyword arguments forwarded to
            :meth:`~utils.vlm.ExecutorVLM.infer`.
        :return: Raw VLM output — a single string or a list of strings when
            ``n_outputs > 1``.
        """
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        result = self._vlm.infer(**kwargs)
        texts = kwargs["texts"]
        images = kwargs["images"]
        assert isinstance(texts, str)
        if isinstance(result, list):
            for i, r in enumerate(result):
                self.report.vlm_call_log.append(VLMCallRecord(tag=tag, prompt=texts, images=images, response=r))
        else:
            self.report.vlm_call_log.append(VLMCallRecord(tag=tag, prompt=texts, images=images, response=result))
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

    4. When *allow_self_termination* is enabled, asks the VLM whether the task
       is now complete (:meth:`~execution.executor.Executor._maybe_self_terminate`)
       and stops if it says yes.  The action prompt above is unaffected by that
       flag — the question is asked separately, after the step, never as part of
       choosing one.

    Terminates early when the environment signals ``terminated`` or
    ``truncated``, or when the completion check fires, recording the reason in
    :attr:`~execution.report.ExecutorReport.termination_reason`.

    .. warning:: **Subclass initialisation order**

        Per :class:`Executor` contract, call ``super().__init__()`` **last**.
    """

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK]Reason about the best next action, then respond in exactly this format:
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
        self.report.steps.append(record)
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
                error_message = (
                    "Your previous response could not be parsed. "
                    "You must end your response with:\n"
                    "  Action: <action>\n"
                    "  [STOP]"
                )
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                self._record_invalid(str(vlm_output))
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
                error_message = (
                    f"You tried to do '{action_str}' but that is not a recognised action. DO NOT use '{action_str}' in your response. "
                    "Choose exactly one from the listed actions."
                )
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                self._record_invalid(f"Unrecognised action string: {action_str!r}", reason="unrecognised action")
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
            self.STEP_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[TOOL_RESULT_BLOCK]", self._tool_result_block(tool_call_message))
            .replace("[ACTION_LIST]", self._action_list_block())
            .replace("[TOOLS_BLOCK]", self._tools_block(tool_calls_exceeded))
            .replace("[ACTION_FORMAT]", self._action_format(tool_calls_exceeded))
        )

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


class HistoryAwareExecutor(SimpleExecutor):
    """
    Extends :class:`SimpleExecutor` by including the last *k* environment steps
    in each prompt so the VLM can see what it has already tried.

    :param history_k: Number of recent env steps to include in context (default 5).
    :type history_k: int
    """

    def __init__(self, env, task, max_steps, max_tool_calls, history_k: int = 5, **kwargs):
        self._history_k = history_k
        self._action_history: List[tuple] = []  # (action_class, action_str, success_code)
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

[TOOLS_BLOCK][HISTORY_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning, specifically reason over your history as well. If you see the [no change] message on the action that you are trying, then you almost certainly have slightly misperceived the screen position of the player relative to the objects. In that case, reason about what else you can try instead of just repeating the same action.>
[ACTION_FORMAT]
[STOP]"""

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        if self._action_history:
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
            history_section = "\n".join(history_lines) + frame_change_hint + "\n\n"
        else:
            history_section = ""
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[HISTORY_SECTION]", history_section)
        )


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
            last_record = self.report.steps[-1]
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


class ScreenDiffExecutor(SimpleExecutor):
    """
    Passes both the previous frame and the current frame to the VLM, prompting
    it to reason about what changed before choosing an action.  On the very
    first step only the current frame is available.
    """

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        self._prev_frame = None
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _on_execute_start(self) -> None:
        self._prev_frame = None

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        return self._build_diff_prompt(tool_call_message, error_message, tool_calls_exceeded)

    def _query_vlm(self, prompt: str, frame) -> str:
        images = [self._prev_frame, frame] if self._prev_frame is not None else [frame]
        return self._vlm_call("action", texts=prompt, images=images)

    def _on_env_step(self, record: EnvironmentStepRecord) -> None:
        self._prev_frame = record.frame_before

    DIFF_PROMPT_SINGLE = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    DIFF_PROMPT_PAIR = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. Image 1 is the PREVIOUS screen, Image 2 is the CURRENT screen. Note what changed between frames to understand the effect of your last action.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _build_diff_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        template = self.DIFF_PROMPT_PAIR if self._prev_frame is not None else self.DIFF_PROMPT_SINGLE
        return (
            template
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[TOOL_RESULT_BLOCK]", self._tool_result_block(tool_call_message))
            .replace("[ACTION_LIST]", self._action_list_block())
            .replace("[TOOLS_BLOCK]", self._tools_block(tool_calls_exceeded))
            .replace("[ACTION_FORMAT]", self._action_format(tool_calls_exceeded))
        )


class SelfConsistencyExecutor(SimpleExecutor):
    """
    Samples the VLM *k* times at a given temperature and takes a majority vote
    on the chosen action.  Falls back to the first parseable response if there
    is no majority.

    :param k: Number of samples per decision (default 3).
    :type k: int
    :param temperature: Sampling temperature (default 0.7).
    :type temperature: float
    """

    def __init__(self, env, task, max_steps, max_tool_calls,
                 k: int = 3, temperature: float = 0.7, **kwargs):
        self._k = k
        self._temperature = temperature
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _query_vlm(self, prompt: str, frame) -> List[str]:
        return self._vlm_call(
            "action",
            texts=prompt,
            images=[frame],
            temperature=self._temperature,
            n_outputs=self._k,
        )

    def _pick_action(self, vlm_output: List[str]) -> Optional[str]:
        action_str = self._majority_vote(vlm_output)
        # k samples means k candidate reasonings. The completion check wants the one that
        # argued for the action actually taken, not an arbitrary sample — showing a losing
        # sample's reasoning beside the winning action would describe a step that never
        # happened.
        for response in vlm_output:
            parsed = self._parse_action(response)
            if parsed is not None and action_str is not None and parsed.lower() == action_str.lower():
                self._last_reasoning = self._parse_reasoning(response) or self._last_reasoning
                break
        return action_str

    def _majority_vote(self, responses: List[str]) -> Optional[str]:
        """Parse each response and return the most common action string."""
        parsed = []
        for r in responses:
            action = self._parse_action(r)
            if action is not None:
                parsed.append(action)
        if not parsed:
            return None
        # Majority vote (case-insensitive key, return original casing of first occurrence)
        counts: Dict[str, int] = {}
        first_seen: Dict[str, str] = {}
        for a in parsed:
            key = a.lower()
            counts[key] = counts.get(key, 0) + 1
            if key not in first_seen:
                first_seen[key] = a
        best_key = max(counts, key=lambda k: counts[k])
        return first_seen[best_key]


class ReflectiveExecutor(SimpleExecutor):
    """
    Every *reflection_interval* env steps, calls the VLM with the recent action
    history and current frame to produce a short critique and revised plan.
    That plan is injected into subsequent step prompts.

    :param reflection_interval: Env steps between reflection calls (default 5).
    :type reflection_interval: int
    """

    def __init__(self, env, task, max_steps, max_tool_calls,
                 reflection_interval: int = 5, **kwargs):
        self._reflection_interval = reflection_interval
        self._plan_summary: str = ""
        self._steps_since_reflection: int = 0
        self._reflection_action_log: List[str] = []
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        record = super()._take_action(action_class, **kwargs)
        action_str = action_class.get_action_name(**kwargs)
        if issubclass(action_class, LowLevelAction):
            tags = "" if self._last_frame_changed else " [no change]"
            self._reflection_action_log.append(f"{action_str}{tags}")
        else:
            status = "ok" if record.action_success == 1 else "failed"
            tags = "" if self._last_frame_changed else ", no change"
            self._reflection_action_log.append(f"{action_str} [{status}{tags}]")
        self._steps_since_reflection += 1
        return record

    REFLECTION_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRIOR_PLAN]Recent actions taken: [HISTORY]

The current game screen is shown in the image.

Briefly critique whether the recent actions made progress toward the task. Then state a concise plan for the next few steps (1-2 sentences). End your response with [STOP].
[STOP]"""

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][PLAN_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _reflect(self, frame) -> None:
        history_str = ", ".join(self._reflection_action_log) if self._reflection_action_log else "none"
        self._reflection_action_log = []
        self._steps_since_reflection = 0
        prior_plan = f"Prior plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        prompt = (
            self.REFLECTION_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PRIOR_PLAN]", prior_plan)
            .replace("[HISTORY]", history_str)
        )
        result = self._vlm_call("reflection", texts=prompt, images=[frame], max_new_tokens=200)
        self._plan_summary = result.strip()

    def _on_step_start(self, frame) -> None:
        if self._steps_since_reflection >= self._reflection_interval:
            self._reflect(frame)

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        plan_section = f"Current plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[PLAN_SECTION]", plan_section)
        )


# ---------------------------------------------------------------------------
# Second generation executors
# ---------------------------------------------------------------------------

class SpatialMapExecutor(SimpleExecutor):
    """
    After each env step, calls the VLM to describe what is visible in each
    cardinal direction in compact notation.  The resulting spatial map is
    injected into subsequent prompts to aid navigation.

    Example map entry: "N: wall, S: open path, E: pokemon centre entrance, W: grass"
    """

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        self._spatial_map: str = ""
        self._last_action_str: str = ""
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    MAP_UPDATE_PROMPT = """Task: [TASK][HINT_BLOCK]

[ACTION_CONTEXT]The current game screen is shown in the image.

Describe what you can see in each direction using short phrases. Respond in exactly this format:
N: <what is north>
S: <what is south>
E: <what is east>
W: <what is west>
Here: <describe current location>
[STOP]"""

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][SPATIAL_MAP_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _update_map(self, frame, last_action_str: str) -> None:
        action_context = f"You just took the action: {last_action_str}.\n\n" if last_action_str else ""
        prompt = (
            self.MAP_UPDATE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ACTION_CONTEXT]", action_context)
        )
        result = self._vlm_call("map_update", texts=prompt, images=[frame], max_new_tokens=100)
        # Extract only the N/S/E/W/Here lines
        lines = []
        for line in result.splitlines():
            s = line.strip()
            if s.lower().startswith(("n:", "s:", "e:", "w:", "here:")):
                lines.append(s)
        if lines:
            self._spatial_map = " | ".join(lines)

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        action_str = self._get_action_strings(return_all=True).get(action_class, action_class.__name__)
        self._last_action_str = action_str
        return super()._take_action(action_class, **kwargs)

    def _on_execute_start(self) -> None:
        self._spatial_map = ""
        self._last_action_str = ""

    def _on_step_start(self, frame) -> None:
        self._update_map(frame, self._last_action_str)

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        map_section = f"Spatial map: {self._spatial_map}\n\n" if self._spatial_map else ""
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[SPATIAL_MAP_SECTION]", map_section)
        )


class ConfidenceGatedExecutor(SimpleExecutor):
    """
    Asks the VLM to also output a confidence score 1-5 with each action.
    If the score is at or below *low_confidence_threshold*, the VLM is
    re-queried once with an explicit "think harder" instruction before
    the action is executed.

    Response format::

        Reasoning: <text>
        Confidence: <1-5>
        Action: <action>
        [STOP]

    :param low_confidence_threshold: Re-query if confidence ≤ this value (default 2).
    :type low_confidence_threshold: int
    """

    def __init__(self, env, task, max_steps, max_tool_calls,
                 low_confidence_threshold: int = 2, **kwargs):
        self._low_confidence_threshold = low_confidence_threshold
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _parse_confidence(self, response: str) -> Optional[int]:
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("confidence:"):
                val = stripped[len("confidence:"):].strip()
                for ch in val:
                    if ch.isdigit() and 1 <= int(ch) <= 5:
                        return int(ch)
        return None

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
Confidence: <1-5 how confident you are this is the right action>
[ACTION_FORMAT]
[STOP]"""

    RETHINK_PROMPT = """[FRAME_CONTEXT]

Your previous response had low confidence:
[ORIGINAL_RESPONSE]

Think more carefully. What are you missing? Reconsider all options, then provide your final answer with higher confidence if possible.
Reasoning: <your revised reasoning>
Confidence: <1-5>
Action: <one environment action>
[STOP]"""

    def _build_rethink_prompt(self, original_response: str, frame_context: str) -> str:
        return (
            self.RETHINK_PROMPT
            .replace("[FRAME_CONTEXT]", frame_context)
            .replace("[ORIGINAL_RESPONSE]", original_response)
        )

    def _query_vlm(self, prompt: str, frame) -> str:
        response = self._vlm_call("action", texts=prompt, images=[frame])
        confidence = self._parse_confidence(response)
        if confidence is not None and confidence <= self._low_confidence_threshold:
            rethink_prompt = self._build_rethink_prompt(response, prompt)
            response = self._vlm_call("rethink", texts=rethink_prompt, images=[frame])
        return response


# ---------------------------------------------------------------------------
# Third generation executors
# ---------------------------------------------------------------------------

class ActionValueEstimatorExecutor(SimpleExecutor):
    """
    Instead of asking the VLM to directly choose an action, this executor asks
    it to score every available action on a 1-5 scale, then automatically
    selects the highest-scored one.  Ties are broken by order in the list.

    This forces systematic evaluation of all options rather than anchoring on
    the first plausible action that comes to mind.

    Score prompt response format::

        <action_string>: <1-5>
        <action_string>: <1-5>
        ...
        [STOP]
    """

    SCORE_PROMPT = """Task: [TASK][HINT_BLOCK]

[ERROR_BLOCK]You are playing a GameBoy game. The current screen is shown in the image.

Score each available action on how useful it would be RIGHT NOW for making progress toward the task (1=useless, 5=very useful).

Actions:
[ACTION_LIST]

Respond with one line per action in exactly this format:
<action>: <score>
...
[STOP]"""

    def _score_actions(self, frame, error_message: Optional[str]) -> Optional[str]:
        """Ask VLM to score each available action. Returns best action string or None."""
        action_strings = self._get_action_strings()
        if not action_strings:
            return None
        action_lines = list(action_strings.values())
        prompt = (
            self.SCORE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[ACTION_LIST]", "\n".join(f"  {s}" for s in action_lines))
        )
        response = self._vlm_call("score", texts=prompt, images=[frame], max_new_tokens=300)

        # Parse scores
        best_score = -1
        best_action_str = None
        for line in response.splitlines():
            stripped = line.strip()
            if ":" not in stripped:
                continue
            # Find last colon to split action from score
            last_colon = stripped.rfind(":")
            action_part = stripped[:last_colon].strip()
            score_part = stripped[last_colon + 1:].strip()
            # Parse score digit
            score = None
            for ch in score_part:
                if ch.isdigit():
                    score = int(ch)
                    break
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_action_str = action_part

        return best_action_str

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False

        error_message: Optional[str] = None
        n_env_steps: int = 0
        consecutive_invalid: int = 0

        while n_env_steps < self._max_steps:
            state = self._get_state()
            frame = state["core"]["current_frame"]

            action_str = self._score_actions(frame, error_message)

            if action_str is None:
                error_message = "Could not determine a valid action from scoring. Try again."
                n_env_steps += 1
                consecutive_invalid += 1
                self._record_invalid("Failed to parse action scores")
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            action_class, action_kwargs = self._env.string_to_high_level_action(action_str)
            if action_class is not None:
                record = self._take_action(action_class, **(action_kwargs or {}))
                error_message = None
                n_env_steps += 1
                consecutive_invalid = 0
                if self._last_terminated:
                    self.report.termination_reason = "terminated"
                    return 1
                if self._last_truncated:
                    self.report.termination_reason = "truncated"
                    return 2

                # Scoring produces no reasoning, so _last_reasoning stays None and the
                # check runs on the frames and the action name alone.
                outcome = self._maybe_self_terminate(record, n_env_steps)
                if outcome is not None:
                    return outcome
            else:
                error_message = (
                    f"Highest-scored action '{action_str}' is not a recognised action string. "
                    "Re-score using exact action strings from the list."
                )
                n_env_steps += 1
                consecutive_invalid += 1
                self._record_invalid(f"Unrecognised scored action: {action_str!r}", reason="unrecognised action")
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1

        self.report.termination_reason = "max_steps"
        return 0


class BeliefStateExecutor(SimpleExecutor):
    """
    Maintains a structured belief state about the game world as a set of
    key-value facts (not prose).  After each env step the VLM updates the
    belief state based on the new frame.  The belief state is injected into
    every step prompt.

    Example belief state::

        location: outside Pokemon Centre
        obstacles: wall to the north, path to the south
        goal_proximity: very close
        last_action_result: moved south successfully
    """

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        self._belief_state: str = ""
        self._last_action_str: str = ""
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    BELIEF_UPDATE_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRIOR_BELIEF][LAST_ACTION]The current game screen is shown in the image.

Update the belief state as a compact list of facts. Use short key: value pairs, one per line. Include:
- location: where you appear to be
- obstacles: what is blocking movement
- goal_proximity: how close you are to the task goal
- last_action_result: what the last action achieved

Respond only with the key: value pairs. End your response with [STOP].
[STOP]"""

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][BELIEF_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _update_belief(self, frame, last_action_str: str) -> None:
        prior_belief = f"Prior belief state:\n{self._belief_state}\n\n" if self._belief_state else ""
        last_action = f"Last action taken: {last_action_str}\n\n" if last_action_str else ""
        prompt = (
            self.BELIEF_UPDATE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PRIOR_BELIEF]", prior_belief)
            .replace("[LAST_ACTION]", last_action)
        )
        result = self._vlm_call("belief_update", texts=prompt, images=[frame], max_new_tokens=120)
        # Keep only key: value lines
        lines = []
        for line in result.splitlines():
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("["):
                lines.append(stripped)
        if lines:
            self._belief_state = "\n".join(lines)

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        action_str = self._get_action_strings(return_all=True).get(action_class, action_class.__name__)
        self._last_action_str = action_str
        record = super()._take_action(action_class, **kwargs)
        return record

    def _on_execute_start(self) -> None:
        self._belief_state = ""
        self._last_action_str = ""
        state = self._get_state()
        self._update_belief(state["core"]["current_frame"], "")

    def _on_env_step(self, record: EnvironmentStepRecord) -> None:
        new_state = self._get_state()
        self._update_belief(new_state["core"]["current_frame"], self._last_action_str)

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        belief_section = f"Current belief state:\n{self._belief_state}\n\n" if self._belief_state else ""
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[BELIEF_SECTION]", belief_section)
        )


class AdversarialSamplingExecutor(SimpleExecutor):
    """
    Three-call decision process per step:

    1. **Propose**: VLM proposes a candidate action with reasoning.
    2. **Challenge**: A second VLM call acts as devil's advocate — argues why
       the proposed action might be wrong.
    3. **Decide**: A third VLM call receives both perspectives and makes the
       final decision.

    More expensive but forces consideration of counterarguments before acting.
    """

    def _propose(self, prompt: str, frame) -> Optional[str]:
        """First call: propose a candidate action."""
        return self._vlm_call(
            "propose",
            texts=prompt, images=[frame],
        )

    CHALLENGE_PROMPT = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

Another agent proposed the following:
[PROPOSAL]

Act as devil's advocate. In 2-3 sentences, argue why this action might be WRONG or suboptimal. What could go wrong? What better alternative exists? End your response with [STOP].
[STOP]"""

    DECIDE_PROMPT = """[ORIGINAL_PROMPT]

--- Proposed action ---
[PROPOSAL]

--- Devil's advocate argument ---
[CHALLENGE]

Consider both perspectives. Make your FINAL decision. You may stick with the original or choose differently.
Reasoning: <final reasoning>
Action: <one environment action>
[STOP]"""

    def _challenge(self, proposal: str, frame) -> str:
        """Second call: argue against the proposal."""
        prompt = (
            self.CHALLENGE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PROPOSAL]", proposal)
        )
        return self._vlm_call("challenge", texts=prompt, images=[frame], max_new_tokens=200)

    def _decide(self, proposal: str, challenge: str, original_prompt: str, frame) -> str:
        """Third call: final decision given proposal and challenge."""
        prompt = (
            self.DECIDE_PROMPT
            .replace("[ORIGINAL_PROMPT]", original_prompt)
            .replace("[PROPOSAL]", proposal)
            .replace("[CHALLENGE]", challenge)
        )
        return self._vlm_call("decide", texts=prompt, images=[frame], max_new_tokens=400)

    def _on_execute_start(self) -> None:
        self._last_proposal: str = ""

    def _query_vlm(self, prompt: str, frame) -> str:
        proposal = self._propose(prompt, frame)
        challenge = self._challenge(proposal, frame)
        final_response = self._decide(proposal, challenge, prompt, frame)
        self._last_proposal = proposal
        return final_response

    def _pick_action(self, vlm_output: str) -> Optional[str]:
        action_str = self._parse_action(vlm_output)
        source = vlm_output
        if action_str is None:
            # Falling back to the proposal means the action came from there, so its
            # reasoning is the one that explains the step.
            action_str = self._parse_action(self._last_proposal)
            source = self._last_proposal
        self._last_reasoning = self._parse_reasoning(source) or self._last_reasoning
        return action_str
