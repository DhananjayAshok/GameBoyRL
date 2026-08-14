"""
Abstract base class for all executors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from gameboy_worlds.interface import Environment, HighLevelAction

from execution.executor_action import ExecutorAction
from execution.report import (ACTION_TAGS, EnvironmentStepRecord, ExecutorReport,
                              InvalidStepRecord, ExecutorToolCallRecord,
                              ExecutorVLMCallRecord, parse_completion,
                              per_prompt_token_counts)
from utils import load_parameters, log_error, log_info, log_warn, ExecutorVLM, parse_key_value

#: Consecutive unparseable/unrecognised action replies before the executor gives up
MAX_CONSECUTIVE_INVALID = 4
DEBUG_ON_INVALID = False


class Executor(ABC):
    """
    Abstract base class for game-playing executor agents.

    .. warning:: **Subclass initialisation order**

        ``super().__init__()`` triggers :meth:`_execute` immediately.  Set all
        subclass attributes *before* calling ``super().__init__()``. 

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
    :type allow_self_termination: bool
    :param kwargs: Additional subclass-specific keyword arguments.  These are
        recorded verbatim in :attr:`report.init_kwargs` but are otherwise
        ignored by the base class.

    .. attribute:: available_tools
        :type: List[Type[ExecutorAction]]
    """

    available_tools: list = []

    DONE_CHECK_PROMPT = """
Task: [TASK][HINT_BLOCK]

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

    DONE_CHECK_REASONING_LABEL = "The reasoning given for that action was:"

    #: Env actions shown to the completion check as history.
    DONE_CHECK_HISTORY_K = 8

    #: Run the completion check after every k-th environment step.
    DONE_CHECK_EVERY_K_STEPS = 1

    #: Token budget for the completion check
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
        self._last_reasoning: Optional[str] = None
        self._current_call: Optional[ExecutorVLMCallRecord] = None

        self.report = self._make_report(task, self._run_config(kwargs), max_steps,
                                        max_tool_calls)

        outcome = self._execute()

        self.report.outcome = outcome
        self.report.final_state = self._get_state()

    def _run_config(self, extra_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        The configuration this run actually resolved to, for :attr:`report.init_kwargs`.
        """
        return {
            "hint": self._hint,
            "allow_self_termination": self._allow_self_termination,
            "vlm_model": self._parameters.get("executor_vlm_model"),
            "vlm_kind": self._parameters.get("executor_vlm_kind"),
            "max_new_tokens": self._max_new_tokens,
            **extra_kwargs,
        }

    def _make_report(
        self,
        task: str,
        init_kwargs: dict,
        max_steps: int,
        max_tool_calls: int,
    ) -> ExecutorReport:
        """
        Factory for the report object. 
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

        # TODO: This currently doesn't account for actions that are well defined in the HighLevelAction space of the controller but isn't valid in the current state

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

    def _record_step(self, step) -> None:
        """
        File a step under the VLM call that caused it.

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
        """
        Record an invalid step and trigger a breakpoint if DEBUG_ON_INVALID is set.

        :attr:`~execution.report.ExecutorReport.invalid_steps`.

        :param reason: ``"parse failure"`` or ``"unrecognised action"``.
        """
        self._record_step(InvalidStepRecord(response=response, reason=reason))
        log_warn(f"Invalid response recorded: \n{response}", parameters=self._parameters)
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
        steps_taken = sum(1 for s in self.report.steps if isinstance(s, EnvironmentStepRecord))
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

        Exactly **one** :class:`~execution.report.ExecutorVLMCallRecord` is appended per
        prompt, and — for a single-prompt call — it becomes :attr:`_current_call`.  Every
        step recorded afterwards — :meth:`_take_action`, :meth:`_use_tool`,
        :meth:`_record_invalid` — is filed under it, so a call owns precisely what it
        caused.  A later ``_vlm_call`` takes ownership from here on.

        **Batching.**  Pass a *list* of prompts to ask several independent questions in one
        round trip; the reply is a list in the same order, and one record is appended per
        prompt/response pair.  A batched call is auxiliary by construction:

        - it does **not** become :attr:`_current_call`, so ownership of the next step stays
          with whichever single call last claimed it.  A batch has no single owner, and
          silently handing it one is precisely how calls and steps came to be mis-paired.
        - its tag must not be in :data:`~execution.report.ACTION_TAGS`.  An action decided
          across several prompts has no owning call for the step it produces.

        ``n_outputs > 1`` is still **not supported**, and is a different thing from
        batching: several *samples of one prompt* have no single owner for the resulting
        step, whereas several *prompts* each own nothing.  Ask once, or make each sample
        its own ``_vlm_call``.

        :param tag: Short label for the call's role — ``"action"``, ``"score"`` or
            ``"done_check"`` today, plus any auxiliary tag a future variant needs.

            .. important:: If this call is the one that decides the action, its tag must
                be in :data:`~execution.report.ACTION_TAGS`. Ownership of the step that
                follows goes to the **last** call made, and the consumers that
                reconstruct an action sequence filter on that set — so an action decided
                in an auxiliary call is silently dropped from the dataset.
        :type tag: str
        :param kwargs: Keyword arguments forwarded to
            :meth:`~utils.vlm.ExecutorVLM.infer`.
        :return: The raw VLM output string, or a list of them for a batched call.
        """
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        texts = kwargs["texts"]
        images = kwargs["images"]

        if isinstance(texts, list):
            return self._batched_vlm_call(tag, **kwargs)

        response = self._vlm.infer(**kwargs)
        result, meta = response["output"], response["meta"]
        if isinstance(result, list):
            log_error(
                f"_vlm_call({tag!r}) received {len(result)} responses for one prompt. "
                "Multi-sample calls are unsupported: the steps that follow would have no "
                "single owning call, which is how VLM calls and steps came to be "
                "mis-paired. Drop n_outputs, or issue one _vlm_call per sample.",
                parameters=self._parameters,
            )
        # A single prompt is one record, so meta's counts are scalars.
        record = ExecutorVLMCallRecord(
            tag=tag, prompt=texts, images=images, response=result,
            input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"],
        )
        self.report.vlm_call_log.append(record)
        self._current_call = record
        return result

    def _batched_vlm_call(self, tag: str, **kwargs: Any) -> list:
        """
        Several prompts in one round trip, recorded one entry per pair.
        """
        texts = kwargs["texts"]
        images = kwargs["images"]
        if tag in ACTION_TAGS:
            log_error(
                f"_vlm_call({tag!r}) was batched with {len(texts)} prompts, but "
                f"{tag!r} is an action tag. A batch owns no steps, so the action it "
                "decided would be filed under some earlier call. Decide the action in a "
                "single call, and batch only auxiliary work.",
                parameters=self._parameters,
            )

        results = self._vlm.infer(**kwargs)
        raw_responses, meta = results["output"], results["meta"]
        responses = raw_responses if isinstance(raw_responses, list) else [raw_responses] * len(texts)
        if len(responses) != len(texts):
            # One record per prompt/response pair is the invariant every consumer reads
            # the log under; a short reply list would silently drop prompts from it.
            log_error(
                f"_vlm_call({tag!r}) sent {len(texts)} prompts and got "
                f"{len(responses)} responses back. The call log records one entry per "
                "pair, so the two must match.",
                parameters=self._parameters,
            )

        token_counts = per_prompt_token_counts(meta, len(texts))
        for index, (prompt, response) in enumerate(zip(texts, responses)):
            call_images = (images[index]
                           if index < len(images) and isinstance(images[index], list)
                           else images)
            self.report.vlm_call_log.append(ExecutorVLMCallRecord(
                tag=tag, prompt=prompt, images=call_images, response=response,
                input_tokens=token_counts[index][0],
                output_tokens=token_counts[index][1],
            ))
        # _current_call is deliberately left alone: see the batching note on _vlm_call.
        return responses

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
            log_warn(
                f"Completion check response had no parseable 'Complete:' line, treating as "
                f"not complete:\n{response}",
                parameters=self._parameters,
            )
        return verdict is True

    def _maybe_self_terminate(self, record: EnvironmentStepRecord, n_env_steps: int) -> Optional[int]:
        """
        Run the completion check and end the run if it says the task is done.


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

        The final permitted step is **always** checked regardless of *k*.

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


