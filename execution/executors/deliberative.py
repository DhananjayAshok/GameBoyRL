"""
Executors that change how a single decision is reached.

Each variant here intervenes at the point of choosing one action — scoring candidates,
reflecting on the last one, or arguing with itself — via :meth:`_query_vlm` or
:meth:`_pick_action`.  The state carried between steps is unchanged from
:class:`~execution.executors.simple.SimpleExecutor`.

**The call that decides the action must be the last one, and must be action-tagged.**
Whichever VLM call an executor makes last before returning from :meth:`_query_vlm` is the
one that owns the resulting step (see :meth:`~execution.executors.base.Executor._vlm_call`),
and consumers that reconstruct an action sequence filter on
:data:`~execution.report.ACTION_TAGS`. Two executors were removed for breaking that rather
than being reconciled with it: a ``SelfConsistencyExecutor`` logged several action calls
per step, and a ``ConfidenceGatedExecutor`` decided the action in a ``"rethink"`` call,
so every action it took was dropped from the dataset.

Auxiliary calls (``"reflection"``, ``"propose"``, ``"challenge"``) own no steps and are
free to be as numerous as they like, provided an action-tagged call comes last.

Every executor in this module uses the base class loop; none overrides :meth:`_execute`
— see the note in :mod:`execution.executors.base` for the one that does.
"""

from __future__ import annotations

from typing import List, Optional

from gameboy_worlds.interface.action import LowLevelAction

from execution.executors.simple import SimpleExecutor
from execution.report import EnvironmentStepRecord
from utils import parse_int


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

    def _context_section(self) -> str:
        return f"Current plan: {self._plan_summary}\n\n" if self._plan_summary else ""


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

    #: Replaces the inherited step prompt outright — this executor never asks for an
    #: action, only for scores. The unused ``[TOOL_RESULT_BLOCK]`` / ``[TOOLS_BLOCK]`` /
    #: ``[CONTEXT_SECTION]`` / ``[ACTION_FORMAT]`` slots are simply absent, and the base
    #: substitution chain leaves them alone.
    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

[ERROR_BLOCK]You are playing a GameBoy game. The current screen is shown in the image.

Score each available action on how useful it would be RIGHT NOW for making progress toward the task (1=useless, 5=very useful).

Actions:
[ACTION_LIST]

Respond with one line per action in exactly this format:
<action>: <score>
...
[STOP]"""

    def _query_vlm(self, prompt: str, frame) -> str:
        return self._vlm_call("score", texts=prompt, images=[frame], max_new_tokens=300)

    def _pick_action(self, vlm_output: str) -> Optional[str]:
        """The highest-scored action, or None if no line parsed.

        Deliberately does not set ``_last_reasoning``: scoring produces no reasoning, so
        the completion check runs on the frames and the action name alone.
        """
        best_score = -1
        best_action_str = None
        for line in vlm_output.splitlines():
            stripped = line.strip()
            if ":" not in stripped:
                continue
            # Split on the LAST colon: an action string may contain one, a score may not.
            action_part = stripped[:stripped.rfind(":")].strip()
            # parse_int rather than scanning for the first digit character: that read "10"
            # as 1, and accepted any digit anywhere in the line. It also range-checks
            # against the 1-5 scale the prompt asks for, so an out-of-range score is
            # discarded rather than allowed to win the max.
            score = parse_int(stripped, action_part, 1, 5)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_action_str = action_part
        return best_action_str

    def _on_parse_failure(self, vlm_output) -> str:
        self._record_invalid("Failed to parse action scores")
        return "Could not determine a valid action from scoring. Try again."

    def _on_unrecognised_action(self, action_str: str) -> str:
        self._record_invalid(f"Unrecognised scored action: {action_str!r}",
                             reason="unrecognised action")
        return (
            f"Highest-scored action '{action_str}' is not a recognised action string. "
            "Re-score using exact action strings from the list."
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

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        # Set before super(), which runs the whole episode — see the initialisation-order
        # warning in executors/base.py. It used to be created in _on_execute_start, which
        # happens to run early enough but leaves the attribute undefined on an instance
        # whose _execute never reaches that hook.
        self._last_proposal: str = ""
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

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
        self._last_proposal = ""
        super()._on_execute_start()

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
