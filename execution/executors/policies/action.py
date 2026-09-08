"""
How one decision produces action(s).

"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from utils import parse_action_line, parse_int, parse_key_value


@dataclass(frozen=True)
class Decision:
    """What one VLM call decided.

    :param actions: Action strings to dispatch in order. Empty is not a valid decision —
        return ``None`` from :meth:`ActionPolicy.parse` instead, which the loop reads as a
        parse failure.
    :param reasoning: What the model said it was doing, for the completion check. ``None``
        when the policy's format asks for none.
    """

    actions: List[str]
    reasoning: Optional[str] = None


#: Markdown and list decoration the model wraps action names in when it is asked to
#: enumerate them: "2. **DOWN**", "- START". Stripped before the name is resolved.
_DECORATION = re.compile(r"^[\s\-*#>]*(?:\d+[.)]\s*)?|[\s*`]+$")

#: A parenthesised single token, as in "Arrow Keys (DOWN)". The controller describes the
#: pad as prose -- "Arrow Keys (UP for up, DOWN for down, ...)" -- and a model asked to
#: score every action by name echoes that phrasing back rather than the bare token. The
#: single-action policy never notices, because it emits one `Action:` line and writes the
#: token plainly; the scored policy must name all seven and fails on every directional one.
_PARENTHESISED = re.compile(r"\(([A-Za-z]+)\)")


def normalise_action_name(text: str) -> str:
    """Reduce a model-written action label to the bare action string.

    Leaves anything it does not recognise untouched, so a genuinely unknown action still
    reaches :meth:`unknown_action_error` and is reported rather than silently rewritten.
    """
    cleaned = _DECORATION.sub("", (text or "").strip()).strip()
    cleaned = cleaned.strip("*`").strip()
    inner = _PARENTHESISED.search(cleaned)
    if inner and len(cleaned.split()) > 1:
        return inner.group(1).strip()
    return cleaned


class ActionPolicy(Protocol):
    """The action-selection half of an executor."""

    #: Report tag for the deciding call. Must be in :data:`~execution.report.ACTION_TAGS`.
    tag: str
    #: Registry name, and half of the arm's ``<action>_<history>`` identity.
    name: str
    #: Token budget for the deciding call, or ``None`` for the executor's default.
    max_new_tokens: Optional[int]
    #: How :attr:`Decision.reasoning` is introduced to the completion check. A policy that
    #: reasons once per plan must not have that sentence presented as step-level reasoning.
    done_check_reasoning_label: str

    def instruction(self) -> str:
        """The line(s) telling the model what kind of answer to give."""

    def response_format(self) -> str:
        """The exact reply format, appended after the instruction."""

    def action_format(self, tools_offered: bool) -> str:
        """The ``Action:`` line spliced in for ``[ACTION_FORMAT]``.

        :param tools_offered: Whether a tool call is a legal answer this step.
        """

    def parse(self, response: str) -> Optional[Decision]:
        """Read the reply, or ``None`` if nothing usable came back."""

    def parse_error(self) -> str:
        """What to tell the model when :meth:`parse` returned ``None``."""

    def unknown_action_error(self, action: str) -> str:
        """What to tell the model when a parsed action is not a real action."""


def _one_action_format(tools_offered: bool) -> str:
    if tools_offered:
        return "Action: <one environment action OR one tool call>"
    return "Action: <one environment action>"


def _split_top_level(body: str) -> List[str]:
    """Split on commas that are not inside brackets, so ``f(a=1, b=2)`` stays whole."""
    parts, current, depth = [], "", 0
    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


class SingleActionPolicy:
    """Reason, then name one action. The reference behaviour."""

    tag = "action"
    name = "single"
    max_new_tokens = None
    done_check_reasoning_label = "The reasoning given for that action was:"

    def instruction(self) -> str:
        return "Reason about the best next action, then respond in exactly this format:"

    def response_format(self) -> str:
        return "Reasoning: <your reasoning>\n[ACTION_FORMAT]"

    def action_format(self, tools_offered: bool) -> str:
        return _one_action_format(tools_offered)

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

    Tool calls are scored under the same rubric, which asks about progress toward the task
    — something a passive tool makes none of by construction — so they will tend to score
    low until the rubric scores information-gathering on its own terms.
    """

    tag = "score"
    name = "scored"
    # One scored line per available action, and tool calls add lines on top of the action
    # list. Truncation here is silent — the parser just maxes over whatever lines arrived —
    # so the budget is set well clear of what the longest action list needs.
    max_new_tokens = 700
    done_check_reasoning_label = "The justification given for scoring that action highest was:"

    def instruction(self) -> str:
        return ("Score each available action on how useful it would be RIGHT NOW for "
                "making progress toward the task (1=useless, 5=very useful), and justify "
                "each score in one sentence.\n\nRespond with one line per action in "
                "exactly this format:")

    def response_format(self) -> str:
        return "<action>: <one-sentence reason for the score>: <score>\n..."

    def action_format(self, tools_offered: bool) -> str:
        return _one_action_format(tools_offered)

    def parse_score_line(self, line: str) -> Optional[Tuple[str, str, int]]:
        """
        Split one ``<action>: <reasoning>: <score>`` line into its three fields.
        """
        stripped = line.strip()
        first, last = stripped.find(":"), stripped.rfind(":")
        if first == -1:
            return None
        action = normalise_action_name(stripped[:first])
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
    """Commit to a short sequence of actions, executed until exhausted or one fails."""

    tag = "action"
    name = "sequence"
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
        return "Reasoning: <your reasoning>\n[ACTION_FORMAT]"

    def action_format(self, tools_offered: bool) -> str:
        if tools_offered:
            return ("Action: <ACTION1, ACTION2, ...> — a sequence of environment actions, "
                    "OR Action: <TOOL_CALL> — a single tool call on its own")
        return "Action: <ACTION1, ACTION2, ...>"

    def parse(self, response: str) -> Optional[Decision]:
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("action:"):
                continue
            actions = _split_top_level(stripped[len("action:"):].strip())
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
