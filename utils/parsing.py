"""
Parsers for structured VLM/LLM responses.

Every function here reads a model's free-text reply and pulls a typed value out of it.
There are four readers, one per shape of answer:

===================  =============================================================
:func:`parse_list`   a list of items, optionally under a ``Key:`` heading
:func:`parse_int`    an integer on a ``Key:`` line, optionally range-checked
:func:`parse_action_line`  the ``Action:`` an executor chose
:func:`parse_steps`  a ``[STEP]``-separated plan
===================  =============================================================

plus :func:`strip_stop` for the house end-of-response marker, and two more that live in
:mod:`utils.lm_inference` because they predate this module and are imported by name
throughout: :func:`~utils.lm_inference.parse_key_value` and
:func:`~utils.lm_inference.parse_yes_no`.

These four replace twelve near-duplicates that had accumulated across ``execution/``,
``vlm_scripts/`` and ``debug_scripts/`` — five list readers, three
integer readers and four action readers, each written for one prompt and none aware of the
others. The differences between them were accidental rather than intentional, so the
unified versions take the most permissive rule on every axis. That direction is deliberate:
every strictness in the old set was a silent data-loss path, where a model answering with
``*`` instead of ``-``, or a heading matched mid-line rather than at the start, read as
"the model returned nothing" rather than "the parser could not find it".

Two behaviours from the old set were dropped on purpose and must be supplied by callers:
lowercasing (``vlm_scripts/propose_tasks_zeroshot.py``) and ``"- "`` re-prefixing for
markdown output (``execution/supervisors/``). Both are caller concerns — one names
directories on disk, the other formats a prompt — and neither belongs in a parser.
"""

from __future__ import annotations

import re
from typing import Optional

from utils.lm_inference import parse_key_value


# A list item: "- foo", "* foo", "• foo", "1. foo", "1) foo", "1 foo".
# The bare "<digits> foo" form — a number with no punctuation after it — is accepted
# because models routinely drop the punctuation when asked for a numbered list. It is
# safe here only because scanning stops at the first non-item line, so a prose line that
# happens to open with a number cannot pull in the paragraph after it.
_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)]?)\s+(.*)$")

# The house "no answer" tokens. Matched exactly, not by prefix, so a real answer opening
# with the word "None" ("None of the doors opened, but…") is kept.
_ABSENT = {"NONE", "N/A", "NA"}


def strip_stop(text: str) -> str:
    """Cut a model response at [STOP], the house end-of-response marker."""
    idx = text.lower().find("[stop]")
    return text[:idx] if idx != -1 else text


def _is_absent(value: str) -> bool:
    """Whether an extracted item is a "no answer" token or punctuation rather than content.

    The alphanumeric test catches markdown residue: a heading written ``**Insights:**``
    leaves ``**`` trailing the colon, which is formatting, not the model's first item.
    """
    stripped = value.strip()
    return (not stripped
            or stripped.upper() in _ABSENT
            or not any(ch.isalnum() for ch in stripped))


def _heading_index(lines: list[str], marker: str) -> Optional[int]:
    """Index of the line carrying ``marker``, preferring a heading over a mention.

    A start-anchored match anywhere in the reply beats a mid-line one, because a mid-line
    match is as often narration ("Let me give you the Insights: below") as it is a real
    heading. Leading markdown is ignored before anchoring, so ``**Insights:**`` and
    ``## Insights:`` count as headings — a purely start-anchored test misses those, which
    is how a bolded heading used to read as "the model returned nothing".
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
    :param key: Heading to scope the search to, without a trailing colon (``"Insights"``,
        not ``"Insights:"``). When ``None``, every item line in the reply is taken.
    :return: The items as bare strings — no bullet, no numbering, no trailing ``[STOP]``.

    The rules, and why each is the permissive one:

    - **The reply is cut at ``[STOP]``** before anything else.
    - **When the reply contains exactly one ``Response:``, only the text after it is
      searched.** Models that narrate before answering repeat the key inside the narration.
      Borrowed from :func:`parse_key_value` so the two agree on what counts as an answer.
    - **The key is found by :func:`_heading_index`**, which prefers a real heading to a
      passing mention and tolerates markdown around it.
    - **Anything after the key on its own line is the first item**, so
      ``Insights: only one thing`` is a one-item list rather than an empty one.
    - **``-``, ``*``, ``•`` and numbered items all count.** Models vary between them freely
      and the choice carries no meaning.
    - **Leading prose is skipped, then scanning stops at the first non-item line.** The two
      halves are not symmetric on purpose: before the list starts, a non-item line is the
      preamble models habitually write ("Here are the paraphrases:") and skipping it costs
      nothing; once items are coming, a non-item line is the next section and stopping keeps
      one heading from absorbing another's items. Blank lines never terminate.
    - **Items that are empty or ``NONE``/``N/A`` are dropped**, so "the model explicitly
      said there is nothing here" and "the model listed nothing" both give ``[]``.

    **A missing key gives ``[]``**, not a whole-reply scan. Callers that would rather have
    the unscoped reading than nothing say so explicitly — ``parse_list(t, "Targets") or
    parse_list(t)`` — so the fallback is visible where it matters rather than built into
    every lookup.
    """
    body = strip_stop(text or "")

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

    **Always three-valued.** ``None`` means "the model did not answer", which is a
    different event from any number it could have given, and callers that want a default
    say so at the call site (``parse_int(...) or 1``). An earlier bounded reader returned
    its lower bound on a parse failure, which made a truncated reply indistinguishable
    from a genuine lowest value — the kind of thing that is invisible until someone asks
    why so many episodes came out at the floor.

    Digits are looked for in three passes, most trustworthy first: a whole token, then —
    only when unbounded — every digit in the value concatenated, then any single in-range
    digit. The last pass is what rescues an out-of-range ``11`` from a 1-10 scale as
    ``1``; it is lossy, but it is what the callers have always done.
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
    """The action on the first ``Action:`` line, or ``None`` if there is no usable one.

    The single reader of the ``Action:`` format, which is a cross-module contract: the
    executor chooses an action by it at run time, the frame tooling labels screenshots
    with it, and ``plan_replay`` reconstructs an episode from it.

    **The first match wins**, matching the executor — the code whose reading actually
    determined what happened. ``plan_replay`` previously took the *last* match, so on a
    reply containing more than one ``Action:`` line (a model that narrates a rejected
    action before committing) a replay silently diverged from the run it existed to
    reproduce.

    Non-string input gives ``None`` rather than raising, since callers map this over
    dataframe columns that may hold NaN.
    """
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            value = stripped[len("action:"):].replace("[STOP]", "").strip()
            return value or None
    return None


# Separator InfoPlanSupervisor joins plan steps with, and the `hint` CSV column stores the
# whole plan under. It lives here with parse_steps, the only code that splits on it;
# execution/supervisors/ re-exports the name so `from execution.supervisors import
# PLAN_SEPARATOR` keeps working.
PLAN_SEPARATOR = "[STEP]"


def parse_steps(text: str) -> list[str]:
    """Split a ``[STEP]``-separated plan into steps, dropping empties."""
    if not text:
        return []
    return [part.strip() for part in text.split(PLAN_SEPARATOR) if part.strip()]
