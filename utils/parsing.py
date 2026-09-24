"""
Parsers for structured VLM/LLM responses.

Every function here reads a model's free-text reply and pulls a typed value out of it.
"""

from __future__ import annotations

import re
from typing import Optional

from utils.fundamental import depathify  # noqa: F401  (re-export; see below)
from utils.lm_inference import parse_key_value


# depathify lives in utils.fundamental, importable without the LM stack. Re-exported here to
# keep `from utils.parsing import depathify` working.


# A list item: "- foo", "* foo", "• foo", "1. foo", "1) foo", "1 foo".
# The bare "<digits> foo" form — a number with no punctuation after it — is accepted
_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)]?)\s+(.*)$")

_ABSENT = {"NONE", "N/A", "NA"}


def _is_absent(value: str) -> bool:
    """Whether an extracted item is a "no answer" token or punctuation rather than content.
    The alphanumeric test catches markdown residue such as a trailing ``**``.

    :return: Whether the item is absent.
    :rtype: bool
    """
    stripped = value.strip()
    return (not stripped
            or stripped.upper() in _ABSENT
            or not any(ch.isalnum() for ch in stripped))


def _heading_index(lines: list[str], marker: str) -> Optional[int]:
    """Index of the line carrying ``marker``, preferring a heading over a mention.

    :param lines: The lines of the model's reply.
    :type lines: list[str]
    :param marker: The heading to look for, in lowercase and with a trailing colon.
    :type marker: str
    :return: The index of the line with the heading, or the first line mentioning it
    :rtype: Optional[int]
    """
    mention = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower().lstrip("*#->•+ \t")
        if stripped.startswith(marker):
            return i
        if mention is None and marker in line.lower():
            mention = i
    return mention


def parse_list(text: str, key: Optional[str] = None) -> list[str]:
    """Items from a model's list answer, optionally scoped to a ``Key:`` heading.

    :param text: The model response.
    :type text: str
    :param key: Heading to scope the search to, without a trailing colon (``"Insights"``,
        not ``"Insights:"``). When ``None``, every item line in the reply is taken.
    :type key: str or None
    :return: The items as bare strings — no bullet, no numbering.
    :rtype: list[str]
    """
    body = text or ""

    lowered = body.lower()
    if lowered.count("response:") == 1:
        body = body[lowered.index("response:") + len("response:"):]

    lines = body.splitlines()
    items: list[str] = []
    start = 0

    if key is not None:
        index = _heading_index(lines, f"{key.lower()}:")
        if index is None:
            return []
        marker = f"{key.lower()}:"
        line = lines[index]
        # Whatever follows the heading on its own line is the first item.
        head = line[line.lower().index(marker) + len(marker):].strip()
        match = _ITEM_RE.match(head)
        head = match.group(1).strip() if match else head
        if not _is_absent(head):
            items.append(head)
        start = index + 1

    started = bool(items)
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        match = _ITEM_RE.match(stripped)
        if match is None:
            if started:
                break      # next section — stop
            continue       # preamble before the list — skip
        started = True
        value = match.group(1).strip()
        if not _is_absent(value):
            items.append(value)
    return items


def parse_int(text: str, key: str, lo: Optional[int] = None,
              hi: Optional[int] = None) -> Optional[int]:
    """The integer on a ``Key:`` line, or ``None`` if the model did not give one.

    :param text: The model response.
    :param key: The key whose value to read.
    :param lo: Inclusive lower bound, or ``None`` for unbounded.
    :param hi: Inclusive upper bound, or ``None`` for unbounded.
    :return: The integer, or ``None`` when the key is absent, answered ``N/A``/``none``/
        ``unknown``, unparseable, or out of range.
    """
    raw = (parse_key_value(text, key) or "").strip()
    if not raw or raw.lower().startswith(("n/a", "na", "none", "unknown")):
        return None

    def in_range(value: int) -> bool:
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    for token in raw.replace(",", " ").split():
        if token.isdigit() and in_range(int(token)):
            return int(token)

    if lo is None and hi is None:
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None

    for ch in raw:
        if ch.isdigit() and in_range(int(ch)):
            return int(ch)
    return None


def parse_action_line(text) -> Optional[str]:
    """
    The action on the first ``Action:`` line, or ``None`` if there is no usable one.

    :param text: The model response.
    :type text: str
    :return: The action, or ``None`` when the key is absent, answered ``N/A``/``none``/``unknown``, or unparseable.
    :rtype: Optional[str]
    """
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            value = stripped[len("action:"):].strip()
            return value or None
    return None



PLAN_SEPARATOR = "[STEP]"


def parse_steps(text: str) -> list[str]:
    """
    Split a ``[STEP]``-separated plan into steps, dropping empties.
    
    :param text: The model response.
    :type text: str
    :return: The steps, or ``[]`` when the model produced no usable plan.
    :rtype: list[str]
    """
    if not text:
        return []
    return [part.strip() for part in text.split(PLAN_SEPARATOR) if part.strip()]


def parse_plan(text: str) -> list[str]:
    """
    The steps of a ``Plan:`` answer, however the model chose to lay them out.
    :param text: The model response.
    :type text: str
    :return: The steps, or ``[]`` when the model produced no usable plan.
    :rtype: list[str]
    """
    body = text or ""

    lowered = body.lower()
    if lowered.count("response:") == 1:
        body = body[lowered.index("response:") + len("response:"):]

    lines = body.splitlines()
    marker = "plan:"
    index = _heading_index(lines, marker)
    if index is None:
        return []

    head = lines[index]
    rest = head[head.lower().index(marker) + len(marker):]
    block = "\n".join([rest, *lines[index + 1:]]).strip().lstrip("*# \t").strip()

    if not block or block.upper().startswith("NONE"):
        return []
    if PLAN_SEPARATOR in block:
        return parse_steps(block)
    steps = parse_list(text, "Plan")
    if steps:
        return steps
    return [] if _is_absent(block) else [" ".join(block.split())]
