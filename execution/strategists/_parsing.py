"""
Filling strategist prompts in, and reading the replies back out.

Separated from the strategist itself so the fragile half — everything that depends on a
model having followed a format — can be exercised without a model. Every parser here is
written to degrade rather than raise: a reply that arrives truncated, unlabelled or in the
wrong case should cost one weak turn, not end the episode.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from utils import parse_key_value

from execution.strategists.prompts import EMPTY_MARKER, FIELD_SEPARATOR


def render(template: str, **substitutions: object) -> str:
    """
    Fill a prompt template's ``[TOKEN]`` slots.

    :param template: One of the templates in :mod:`execution.strategists.prompts`.
    :param substitutions: Slot name to value, lower-case; ``game="pokemon_red"`` fills
        ``[GAME]``. Values are stringified, so step budgets can be passed as ints.
    :return: The filled prompt.
    """
    filled = template
    for name, value in substitutions.items():
        filled = filled.replace(f"[{name.upper()}]", str(value))
    # Always last: the format-token slots are the same for every template, and leaving them
    # to callers meant a template that used [SEP] only in its example line rendered a literal
    # "[SEP]" into the prompt whenever a caller forgot it.
    filled = filled.replace("[SEP]", FIELD_SEPARATOR).replace("[EMPTY]", EMPTY_MARKER)
    return filled


def _block_value(text: str, key: str, stop_keys: Tuple[str, ...] = ()) -> Optional[str]:
    """
    Read a key's value, continuing across lines until the next known key.

    :func:`~utils.parse_key_value` stops at the end of the key's own line, which is right for
    a one-line answer and wrong for a model that was asked for a separated list and produced
    a bulleted one instead. That reply is well-formed enough to use, but line-oriented
    reading silently keeps only its first item — the worst outcome, since a short list looks
    like a valid short list.

    Continuation stops on the *named* following keys rather than on anything shaped like
    ``Word:``, because map facts legitimately contain colons ("Pallet Town: two houses") and
    a general rule truncates them.

    :param text: Raw model output.
    :param key: Key to read, matched case-insensitively.
    :param stop_keys: The other keys of this reply, any of which ends the value.
    :return: The value, newlines preserved, or ``None`` if the key is absent or empty.
    """
    marker = f"{key.lower()}:"
    stops = tuple(f"{stop.lower()}:" for stop in stop_keys)
    collected: List[str] = []
    capturing = False

    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not capturing:
            if lowered.startswith(marker):
                capturing = True
                collected.append(stripped[len(marker):].strip())
            continue
        if lowered.startswith("[stop]") or not stripped:
            break
        if any(lowered.startswith(stop) for stop in stops):
            break
        collected.append(stripped)

    value = "\n".join(part for part in collected if part).strip()
    return value or None


def _split_items(value: Optional[str]) -> List[str]:
    """
    Split a single-line multi-valued answer, dropping empties and the "none" marker.

    Also tolerates a model that answered with a bulleted list anyway: leading dashes are
    stripped per item, and newlines are treated as separators alongside
    :data:`~execution.strategists.prompts.FIELD_SEPARATOR`.
    """
    if not value:
        return []
    normalised = value.replace("\n", FIELD_SEPARATOR)
    items = []
    for raw in normalised.split(FIELD_SEPARATOR):
        item = raw.strip().lstrip("-*").strip()
        if not item:
            continue
        if item.lower().strip(".") == EMPTY_MARKER:
            continue
        items.append(item)
    return items


def parse_plan(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Read a planning reply.

    :param text: Raw model output.
    :return: ``(task, hint)``. ``task`` is ``None`` if no task line was found, which the
        caller must treat as a failed planning turn — inventing a fallback task here would
        hide a broken prompt behind plausible-looking play.
    """
    task = parse_key_value(text, "Task")
    hint = parse_key_value(text, "Hint")
    if hint and hint.lower().strip(".") == EMPTY_MARKER:
        hint = None
    return (task.strip() if task else None, hint.strip() if hint else None)


def parse_reflection(text: str) -> Tuple[List[str], Dict[str, str], str]:
    """
    Read a reflection reply.

    :param text: Raw model output.
    :return: ``(map_facts, progress_updates, lesson)``. Any field the model omitted comes
        back empty rather than raising: a reflection that yields nothing is a wasted turn,
        but a reflection that raises loses the attempt that prompted it too.
    """
    facts = _split_items(_block_value(text, "Map facts", ("Progress", "Lesson")))

    updates: Dict[str, str] = {}
    for item in _split_items(_block_value(text, "Progress", ("Map facts", "Lesson"))):
        if "=" not in item:
            # A belief with no value is not a belief. Dropped rather than stored as an empty
            # string, which would render as "has_pokemon: " and read as a definite "no".
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            updates[key] = value

    lesson = _block_value(text, "Lesson", ("Map facts", "Progress")) or ""
    return facts, updates, lesson.strip()


def parse_summary(text: str) -> str:
    """
    Read a compression reply.

    :param text: Raw model output.
    :return: The summary, or ``""`` if none was found. The empty string is meaningful: it is
        what tells :meth:`~execution.strategists.notebook.Notebook.apply_compression` to
        leave the log alone rather than fold it into nothing.
    """
    summary = _block_value(text, "Summary")
    if summary:
        return summary.strip()
    # A model that ignored the format but still wrote a summary is worth salvaging here,
    # because the alternative is discarding the history it was asked to condense.
    stripped = text.replace("[STOP]", "").strip()
    return stripped if stripped else ""


def parse_goal_check(text: str) -> bool:
    """
    Read a goal-check reply.

    :param text: Raw model output.
    :return: ``True`` only on an explicit yes. Anything unparseable is ``False``, because the
        cost of a missed positive is one more task and the cost of a false positive is
        declaring the run a success when it is not.
    """
    answer = parse_key_value(text, "Answer")
    if not answer:
        return False
    return answer.strip().lower().startswith("yes")
