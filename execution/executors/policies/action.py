"""
How one decision produces action(s).

An :class:`ActionPolicy` contributes two fragments to the step prompt and parses the reply
into a :class:`Decision`.  **It never touches the VLM.**  The executor makes exactly one
call on its behalf and hands it the text.

That is the whole point of the split.  Two executors were deleted for making the action
across several calls — a ``SelfConsistencyExecutor`` that logged one action call per sample
and a ``ConfidenceGatedExecutor`` that decided in a non-action-tagged ``"rethink"`` call —
and in both cases every action they took was dropped from the harvested dataset, silently.
A policy that cannot reach the VLM cannot reintroduce that: the "exactly one action-tagged
call per decision" rule is enforced by the shape of this interface rather than by anyone
remembering it.

The three policies differ only in what they ask for and how they read the answer:

- :class:`SingleActionPolicy`   reason, then name one action
- :class:`ScoredActionPolicy`   justify and score every action, take the best
- :class:`SequenceActionPolicy` commit to several actions at once
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from utils import parse_action_line, parse_int, parse_key_value


@dataclass(frozen=True)
class Decision:
    """What one VLM call decided.

    ``actions`` is a list even when it holds one element.  That is the change that lets a
    single loop drive every policy: the sequence planner used to need its own ``_execute``
    solely because a call produced several actions, and the loop that special-cased it had
    to reimplement budget accounting, invalid handling and the completion check.

    :param actions: Action strings to dispatch in order. Empty is not a valid decision —
        return ``None`` from :meth:`ActionPolicy.parse` instead, which the loop reads as a
        parse failure.
    :param reasoning: What the model said it was doing, for the completion check. ``None``
        when the policy's format asks for none.
    """

    actions: List[str]
    reasoning: Optional[str] = None


class ActionPolicy(Protocol):
    """The action-selection half of an executor."""

    #: Report tag for the deciding call. Must be in :data:`~execution.report.ACTION_TAGS`.
    tag: str
    #: Registry name, and half of the arm's ``<action>_<history>`` identity.
    name: str
    #: Whether the step prompt should offer passive tool calls alongside env actions.
    supports_tools: bool
    #: Token budget for the deciding call, or ``None`` for the executor's default.
    max_new_tokens: Optional[int]
    #: How :attr:`Decision.reasoning` is introduced to the completion check. A policy that
    #: reasons once per plan must not have that sentence presented as step-level reasoning.
    done_check_reasoning_label: str

    def instruction(self) -> str:
        """The line(s) telling the model what kind of answer to give."""

    def response_format(self) -> str:
        """The exact reply format, appended after the instruction."""

    def parse(self, response: str) -> Optional[Decision]:
        """Read the reply, or ``None`` if nothing usable came back."""

    def parse_error(self) -> str:
        """What to tell the model when :meth:`parse` returned ``None``."""

    def unknown_action_error(self, action: str) -> str:
        """What to tell the model when a parsed action is not a real action."""


class SingleActionPolicy:
    """Reason, then name one action. The reference behaviour."""

    tag = "action"
    name = "single"
    supports_tools = True
    max_new_tokens = None
    done_check_reasoning_label = "The reasoning given for that action was:"

    def instruction(self) -> str:
        return "Reason about the best next action, then respond in exactly this format:"

    def response_format(self) -> str:
        return "Reasoning: <your reasoning>\n[ACTION_FORMAT]"

    def parse(self, response: str) -> Optional[Decision]:
        action = parse_action_line(response)
        if action is None:
            return None
        return Decision(actions=[action],
                        reasoning=(parse_key_value(response, "Reasoning") or "").strip()
                        or None)

    def parse_error(self) -> str:
        return ("Your previous response could not be parsed. "
                "You must end your response with:\n"
                "  Action: <action>\n"
                "  [STOP]")

    def unknown_action_error(self, action: str) -> str:
        return (f"You tried to do '{action}' but that is not a recognised action. DO NOT "
                f"use '{action}' in your response. Choose exactly one from the listed "
                "actions.")


class ScoredActionPolicy:
    """Justify and score every available action, then take the highest.

    Forces systematic evaluation of all options rather than anchoring on the first
    plausible action. Ties go to the first in the list.

    ``supports_tools`` is False: the rubric scores actions on progress toward the task, and
    a passive tool call makes none by construction, so a tool would score 1 and never win.
    Offering tools under a rubric that cannot choose them is worse than not offering them.
    Revisit alongside a rubric that scores information-gathering on its own terms.
    """

    tag = "score"
    name = "scored"
    supports_tools = False
    max_new_tokens = 300
    done_check_reasoning_label = "The justification given for scoring that action highest was:"

    def instruction(self) -> str:
        return ("Score each available action on how useful it would be RIGHT NOW for "
                "making progress toward the task (1=useless, 5=very useful), and justify "
                "each score in one sentence.\n\nRespond with one line per action in "
                "exactly this format:")

    def response_format(self) -> str:
        return "<action>: <one-sentence reason for the score>: <score>\n..."

    def parse_score_line(self, line: str) -> Optional[Tuple[str, str, int]]:
        """Split one ``<action>: <reasoning>: <score>`` line into its three fields.

        Anchored at both ends rather than on a single separator, because only the outer
        two colons are reliable: the action is everything before the **first** and the
        score everything after the **last**, leaving the middle — reasoning, colons and
        all — untouched. An action invocation is ``name(args)`` and never contains a colon,
        so the first colon is always the action's terminator even though the *listed*
        verbalisation of that action carries one before its description.

        A line with only one colon is read as an action and a score with the justification
        omitted, rather than rejected — a missing reason costs the completion check some
        context, but the score it came with is still a usable vote.
        """
        stripped = line.strip()
        first, last = stripped.find(":"), stripped.rfind(":")
        if first == -1:
            return None
        action = stripped[:first].strip()
        reasoning = stripped[first + 1:last].strip() if last > first else ""
        # Scored through a synthetic keyed line so parse_int only ever sees the score
        # field. Handing it the whole line would search the justification for digits too,
        # and "reach route 1, best odds: 4" would score 1.
        score = parse_int(f"score: {stripped[last + 1:]}", "score", 1, 5)
        if not action or score is None:
            return None
        return action, reasoning, score

    def parse(self, response: str) -> Optional[Decision]:
        best_score, best_action, best_reasoning = -1, None, None
        for line in response.splitlines():
            parsed = self.parse_score_line(line)
            if parsed is None:
                continue
            action, reasoning, score = parsed
            if score > best_score:
                best_score, best_action, best_reasoning = score, action, reasoning or None
        if best_action is None:
            return None
        # Only the winner's justification: the losing lines explain a choice that was not
        # made, so folding them in would misdescribe the step to the completion check.
        return Decision(actions=[best_action], reasoning=best_reasoning)

    def parse_error(self) -> str:
        return ("Could not determine a valid action from scoring. Score every action "
                "again, one per line, in exactly this format:\n"
                "  <action>: <one-sentence reason for the score>: <score 1-5>")

    def unknown_action_error(self, action: str) -> str:
        return (f"Highest-scored action '{action}' is not a recognised action string. "
                "Re-score using exact action strings from the list.")


class SequenceActionPolicy:
    """Commit to a short sequence of actions, executed until exhausted or one fails.

    ``supports_tools`` is False: a passive tool call neither advances the game nor is
    unlimited, so one sitting in the middle of a committed plan is a different execution
    model than the plan assumes.
    """

    tag = "action"
    name = "sequence"
    supports_tools = False
    max_new_tokens = None
    # This policy reasons once per *plan* and then executes several actions from it, so the
    # reasoning the completion check sees is the plan's rather than the step's. Say so
    # rather than presenting it as step-level reasoning.
    done_check_reasoning_label = ("The reasoning given for the planned sequence this "
                                  "action came from was:")

    def instruction(self) -> str:
        return ("Plan a short sequence of actions (1-5) to make progress on the task. "
                "Respond in exactly this format:")

    def response_format(self) -> str:
        return "Reasoning: <your reasoning>\nAction: <ACTION1, ACTION2, ...>"

    def parse(self, response: str) -> Optional[Decision]:
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("action:"):
                continue
            body = stripped[len("action:"):].strip()
            actions = [part.strip() for part in body.split(",") if part.strip()]
            if not actions:
                return None
            return Decision(actions=actions,
                            reasoning=(parse_key_value(response, "Reasoning") or "").strip()
                            or None)
        return None

    def parse_error(self) -> str:
        return ("Your previous response could not be parsed. "
                "You must end your response with:\n"
                "  Action: ACTION1, ACTION2, ...\n"
                "  [STOP]")

    def unknown_action_error(self, action: str) -> str:
        return (f"'{action}' in the planned sequence is not a recognised action. "
                "Re-plan with valid actions.")


AVAILABLE_ACTION_POLICIES = {
    policy.name: policy
    for policy in (SingleActionPolicy, ScoredActionPolicy, SequenceActionPolicy)
}
""" Action policies by registry name — the first half of an arm's identity. """
