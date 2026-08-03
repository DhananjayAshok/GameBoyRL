"""
Executors that change how a single decision is reached.

Each variant here intervenes at the point of choosing one action — sampling several
candidates, scoring them, reflecting on the last one, or gating on confidence — via
:meth:`_query_vlm` or :meth:`_pick_action`.  The state carried between steps is
unchanged from :class:`~execution.executors.simple.SimpleExecutor`.

:class:`ActionValueEstimatorExecutor` overrides :meth:`_execute` outright instead of
using the base class hooks — see the note in :mod:`execution.executors.base`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from gameboy_worlds.interface.action import LowLevelAction

from execution.executors.base import MAX_CONSECUTIVE_INVALID
from execution.executors.simple import SimpleExecutor
from execution.report import EnvironmentStepRecord
from utils import parse_int


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
        """The 1-5 self-reported confidence, or None if the model did not give one.

        A method for the same reason as :meth:`_parse_action` — it is a subclass hook. The
        range is the one this executor's prompt asks for, so it is passed rather than baked
        into the parser.
        """
        return parse_int(response, "Confidence", 1, 5)

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
