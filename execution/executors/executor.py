"""
The one executor. Everything that used to be a subclass is now a pair of policies.

:class:`PolicyExecutor` owns the loop, the budget, the prompt assembly and the environment
plumbing.  What varies between arms lives in two collaborators:

- an :class:`~execution.executors.policies.action.ActionPolicy` — how a decision is made
- a :class:`~execution.executors.policies.history.HistoryPolicy` — what is remembered

Nine arms, three plus three plus one class.  The hierarchy this replaces had one class per
variant and a hook surface (``_query_vlm``, ``_pick_action``, ``_context_section``,
``_step_template``, ``_on_step_start``, ``_on_env_step``, …) whose only job was to let
those classes reach into a loop they could not otherwise change — and even then the two
axes could not be combined, because a variant that rewrote ``STEP_PROMPT`` to change the
response format silently dropped the ``[CONTEXT_SECTION]`` placeholder that carried the
history.

**A decision yields a list of actions.**  That is the change that removes the last reason
for a second loop: the sequence planner used to override ``_execute`` outright because one
of its calls produced several steps, and that override reimplemented budget accounting,
invalid handling and the completion check, each subtly differently.
"""

from __future__ import annotations

import re

from typing import Any, List, Optional, Type

from gameboy_worlds.interface import HighLevelAction

from execution.executors.base import MAX_CONSECUTIVE_INVALID, Executor
from execution.executors.policies import (AVAILABLE_ACTION_POLICIES,
                                          AVAILABLE_HISTORY_POLICIES)
from execution.report import EnvironmentStepRecord, StepRecord



#: Angle-bracket placeholders as they appear in the action list a controller advertises,
#: e.g. ``move(<up, down, right or left> <steps: int>)``. A model asked to fill one in often
#: keeps the brackets -- ``move(<up> 1)``.
_PLACEHOLDER_BRACKETS = re.compile(r"[<>]")


def _advertised_verbs(action_strings) -> set:
    """The call names a controller advertises, e.g. {"move", "interact", "openmenu"}.

    Read off the advertised strings rather than hardcoded, so this follows whatever the
    controller offers for the current game and state.
    """
    verbs = set()
    for text in (action_strings or {}).values():
        head = str(text).split("(", 1)[0].strip().lower()
        if head and head.isidentifier():
            verbs.add(head)
    return verbs


def normalise_action_string(action_str: str, verbs=()) -> str:
    """Repair the ways a model mis-renders an advertised call, and nothing else.

    Every failure below was observed on hardware in one 407-step state_wise run, which lost
    19.4% of its steps to them -- all of them a correctly *chosen* action written in the
    wrong shape:

        move(<up> 1)      brackets copied from the advertised format
        move(<right>, 1)  brackets, plus a comma between arguments
        Move right 1      prose: capitalised, no parentheses
        Interact          prose: no parentheses, no arguments
        openmenu<trainer> brackets used *as* the parentheses

    An action whose verb the controller does not advertise is returned untouched, so a
    genuinely unrecognised action still fails and is reported rather than silently rewritten.
    """
    text = _PLACEHOLDER_BRACKETS.sub(" ", action_str or "").strip()
    if not text:
        return ""
    verbs = set(verbs)

    if "(" in text:
        head, _, rest = text.partition("(")
        head = head.strip().lower()
        if head not in verbs:
            return action_str.strip()
        args = " ".join(rest.rsplit(")", 1)[0].replace(",", " ").split())
        return f"{head}({args})"

    # No parentheses: "Move right 1" / "Interact" / "openmenu trainer".
    parts = text.split()
    head = parts[0].strip().lower()
    if head not in verbs:
        return action_str.strip()
    args = " ".join(" ".join(parts[1:]).replace(",", " ").split())
    return f"{head}({args})"


class PolicyExecutor(Executor):
    """
    An executor assembled from an action policy and a history policy.
    """

    ACTION_POLICY = "single"
    HISTORY_POLICY = "none"

    STEP_PROMPT = """
Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][CONTEXT_SECTION][INSTRUCTION]
[RESPONSE_FORMAT]
[STOP]"""

    def __init__(self, env, task, max_steps, max_tool_calls,
                 action_policy=None, history_policy=None,
                 history_k: int = 5, **kwargs):
        action_policy = action_policy or self.ACTION_POLICY
        history_policy = history_policy or self.HISTORY_POLICY
        if isinstance(action_policy, str):
            action_policy = AVAILABLE_ACTION_POLICIES[action_policy]()
        if isinstance(history_policy, str):
            history_policy = AVAILABLE_HISTORY_POLICIES[history_policy](
                # Bound late: `self._vlm_call` is the recording entry point, so a history
                # policy's auxiliary calls land in the same log as everything else rather
                # than going out unrecorded.
                call=self._vlm_call, game=kwargs.get("game", ""), history_k=history_k,
            )
        self._action_policy = action_policy
        self._history_policy = history_policy
        # Set before super().__init__, which runs the whole episode.
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    @property
    def DONE_CHECK_REASONING_LABEL(self) -> str:  # noqa: N802 - shadows a base constant
        """
        Deferred to the action policy — see the base class attribute it overrides.
        """
        return self._action_policy.done_check_reasoning_label

    def _run_config(self, extra_kwargs):
        """
        The base configuration plus which pair of policies this arm actually is.
        """
        return {**super()._run_config(extra_kwargs),
                "action_policy": self._action_policy.name,
                "history_policy": self._history_policy.name}

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _take_action(self, action_class: Type[HighLevelAction], **kwargs) -> EnvironmentStepRecord:
        """Execute one high-level action and record it under the current call."""
        before_info = self._env.get_info()
        frame_before = before_info["core"]["current_frame"]
        obs, reward, terminated, truncated, info = self._env.step_high_level_action(
            action_class, **kwargs
        )
        if "previous_action_details" in info.get("core", {}):
            _, _, transition_states, action_success, _ = info["core"]["previous_action_details"]
        else:
            transition_states, action_success = [], -1

        record = EnvironmentStepRecord(
            frame_before=frame_before,
            frame_after=info["core"]["current_frame"],
            action_class=action_class,
            kwargs=kwargs,
            transition_states=transition_states,
            action_success=action_success,
            reward=reward,
            frame_changed=info["core"]["frame_changed"],
        )
        self._record_step(record)
        self._last_terminated = terminated
        self._last_truncated = truncated
        self._last_frame_changed = record.frame_changed
        return record

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _error_block(error_message: Optional[str]) -> str:
        return f"[ERROR] {error_message}\n\n" if error_message is not None else ""

    @staticmethod
    def _tool_result_block(tool_call_message: Optional[str]) -> str:
        return f"Tool result: {tool_call_message}\n\n" if tool_call_message is not None else ""

    def _action_list_block(self) -> str:
        return "\n".join(f"  {s}" for s in self._get_action_strings().values())

    def _tools_offered(self, tool_calls_exceeded: bool) -> bool:
        """
        Whether this step may use a tool at all.

        Two things have to agree: the budget is not spent, and tools exist. Every action
        policy can choose a tool.
        """
        return not tool_calls_exceeded and bool(self.available_tools)

    def _tools_block(self, tool_calls_exceeded: bool) -> str:
        if not self._tools_offered(tool_calls_exceeded):
            return ""
        lines = [f"Available tool calls (do not advance the game, "
                 f"{self._max_tool_calls} total budget):"]
        lines += [f"  {tool_class.verbalize()}" for tool_class in self.available_tools]
        return "\n".join(lines) + "\n\n"

    def _action_format(self, tool_calls_exceeded: bool) -> str:
        return self._action_policy.action_format(self._tools_offered(tool_calls_exceeded))

    def _build_prompt(self, tool_call_message: Optional[str], error_message: Optional[str],
                      tool_calls_exceeded: bool) -> str:
        return (
            self.STEP_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[TOOL_RESULT_BLOCK]", self._tool_result_block(tool_call_message))
            .replace("[ACTION_LIST]", self._action_list_block())
            .replace("[TOOLS_BLOCK]", self._tools_block(tool_calls_exceeded))
            .replace("[CONTEXT_SECTION]", self._history_policy.render())
            .replace("[INSTRUCTION]", self._action_policy.instruction())
            .replace("[RESPONSE_FORMAT]", self._action_policy.response_format())
            # Last: the response format may itself contain the placeholder.
            .replace("[ACTION_FORMAT]", self._action_format(tool_calls_exceeded))
        )

    def _try_parse_tool_call(self, action_str: str) -> Optional[tuple]:
        """``(tool_class, kwargs)`` if *action_str* is a tool call, else ``None``."""
        for tool_class in self.available_tools:
            try:
                kwargs = tool_class.string_to_kwargs(action_str)
                if kwargs is not None:
                    return tool_class, kwargs
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._history_policy.reset()

        tool_call_message: Optional[str] = None
        error_message: Optional[str] = None
        n_env_steps = 0
        consecutive_invalid = 0

        while n_env_steps < self._max_steps:
            frame = self._get_state()["core"]["current_frame"]
            tool_calls_exceeded = self._n_tool_calls >= self._max_tool_calls
            prompt = self._build_prompt(tool_call_message, error_message,
                                        tool_calls_exceeded)

            call_kwargs = {"texts": prompt, "images": [frame]}
            if self._action_policy.max_new_tokens is not None:
                call_kwargs["max_new_tokens"] = self._action_policy.max_new_tokens
            response = self._vlm_call(self._action_policy.tag, **call_kwargs)

            decision = self._action_policy.parse(response)
            if decision is None:
                self._record_invalid(str(response))
                error_message = self._action_policy.parse_error()
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            # Captured here, once per decision, because this is the single funnel every
            # response passes through. A sequence decision covers several steps with one
            # reasoning, which is what the completion check is then shown.
            self._last_reasoning = decision.reasoning or self._last_reasoning

            (outcome, n_env_steps, consecutive_invalid, error_message,
             tool_call_message) = self._run_decision(decision, n_env_steps,
                                                     consecutive_invalid)
            if outcome is not None:
                return outcome

        self.report.termination_reason = "max_steps"
        return 0

    def _run_decision(self, decision, n_env_steps: int, consecutive_invalid: int):
        """Dispatch one decision's actions, then let the history see what happened.

        :return: ``(outcome, n_env_steps, consecutive_invalid, error_message,
            tool_call_message)`` where a non-None outcome ends the episode.
        """
        records: List[EnvironmentStepRecord] = []
        steps: List[StepRecord] = []
        error_message: Optional[str] = None
        tool_call_message: Optional[str] = None
        aborted = False

        for action_str in decision.actions:
            if n_env_steps >= self._max_steps:
                break

            if self._tools_offered(self._n_tool_calls >= self._max_tool_calls):
                tool_result = self._try_parse_tool_call(action_str)
                if tool_result is not None:
                    tool_class, tool_kwargs = tool_result
                    steps.append(self._use_tool(tool_class, **tool_kwargs))
                    tool_call_message = str(steps[-1].result)
                    # A tool call costs a step of the budget even though it does not advance
                    # the emulator. It used to be free here, which meant this loop had no
                    # bound at all on a tool-heavy run: `max_tool_calls` stops the tool
                    # being *offered*, but until it is spent the loop could iterate without
                    # `n_env_steps` ever moving. The supervisor's episode budget is now
                    # counted the same way (`Supervisor` legs charge every recorded step),
                    # and the two must agree or the leg and the episode disagree about what
                    # a step is.
                    n_env_steps += 1
                    consecutive_invalid = 0
                    # The rest of a committed plan was written without the answer this tool
                    # just returned, so it is re-planned rather than run on.
                    aborted = True
                    break

            action_class, action_kwargs = self._env.string_to_high_level_action(
                normalise_action_string(action_str,
                                        _advertised_verbs(self._get_action_strings())))
            if action_class is None:
                self._record_invalid(f"Unrecognised action string: {action_str!r}",
                                     reason="unrecognised action")
                error_message = self._action_policy.unknown_action_error(action_str)
                n_env_steps += 1
                consecutive_invalid += 1
                aborted = True
                break

            record = self._take_action(action_class, **(action_kwargs or {}))
            records.append(record)
            steps.append(record)
            n_env_steps += 1
            consecutive_invalid = 0

            if self._last_terminated:
                self._finish_decision(steps)
                self.report.termination_reason = "terminated"
                return 1, n_env_steps, consecutive_invalid, error_message, tool_call_message
            if self._last_truncated:
                self._finish_decision(steps)
                self.report.termination_reason = "truncated"
                return 2, n_env_steps, consecutive_invalid, error_message, tool_call_message

            # A failed action ends the decision: the rest of a committed plan was written
            # on the assumption that this one worked. Low-level actions report success=0
            # by convention rather than as a failure, so they never trigger this.
            if (len(decision.actions) > 1
                    and not _is_low_level(action_class)
                    and record.action_success == 0):
                error_message = (f"Action '{action_str}' failed (blocked or invalid). "
                                 "Re-plan.")
                aborted = True
                break

        self._finish_decision(steps)

        if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
            self.report.termination_reason = "max_invalid"
            return -1, n_env_steps, consecutive_invalid, error_message, tool_call_message

        # The completion check runs at a decision boundary, and only when the decision ran
        # to the end. One rule, two cadences: a single-action policy is checked after every
        # step, a sequence once per committed plan — which is what each of them did
        # separately before. A half-executed plan is a state its planner never intended to
        # be judged in, so an aborted decision is not checked at all.
        if records and not aborted:
            outcome = self._maybe_self_terminate(records[-1], n_env_steps)
            if outcome is not None:
                return outcome, n_env_steps, consecutive_invalid, error_message, tool_call_message

        return None, n_env_steps, consecutive_invalid, error_message, tool_call_message

    def _finish_decision(self, steps: List[StepRecord]) -> None:
        """Show the history policy everything this decision did, once.

        Batched at the decision boundary rather than per step so a policy that summarises
        frames pays one round trip for a five-action plan instead of five.
        """
        if steps:
            self._history_policy.observe(steps)


def _is_low_level(action_class) -> bool:
    from gameboy_worlds.interface.action import LowLevelAction
    return issubclass(action_class, LowLevelAction)


def make_executor_class(action_name: str, history_name: str) -> type:
    """A named subclass for one ``<action>_<history>`` arm.

    Real subclasses rather than ``functools.partial`` because the arm's name has to survive
    into the report: :meth:`Executor._make_report` stamps ``__class__.__name__`` as
    ``executor_name``, and that string names the on-disk results directory. A partial would
    collapse all nine arms onto one name and one directory.
    """
    name = f"{action_name}_{history_name}"
    return type(name, (PolicyExecutor,), {
        "ACTION_POLICY": action_name,
        "HISTORY_POLICY": history_name,
        "__doc__": f"Executor arm: {action_name} action policy, {history_name} history.",
    })
