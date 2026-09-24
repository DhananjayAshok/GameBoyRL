"""
The parametric document: what the model already believes about a game, in document form.

The second way an :class:`~execution.info_doc.InfoDocument` can come into existence. The
first is ``vlm_scripts/build_info.py``, which distils one out of real trajectories. This
module asks the model to write one from its own weights, given nothing but the game's name —
no frames, no playthroughs, no retrieval of any kind.

It is the control for the plan arm: same document schema, same relevance pass, same planner,
same budgets, differing only in where the entries came from.

**Nothing here is verified against the game.** The model will confabulate as confidently as
it recalls, which is why the document is written to disk and kept — a run whose plans came
from invented level geometry has to remain auditable.

**Task entries only.** The image section is left empty, so
``InfoSubgoalSupervisor._candidates`` yields nothing for it. Every entry has ``frame=None``,
which :func:`~execution.info_doc.resolve_frame` maps to ``None`` and the relevance call
handles with its no-frame prompt wording.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from execution.info_doc import (
    Entry,
    InfoDocument,
    Provenance,
    TASK_SECTION,
    dump_document,
    load_document,
)
from utils import log_info, log_warn, VLM

PARAMETRIC_SOURCE = "parametric"

PARAMETRIC_PROMPT = """
You are writing a knowledge base for the Game Boy game [GAME], from what you already know about it. You will not be shown any screenshots — write only from your own knowledge of this game.

Produce up to [N_CATEGORIES] categories of task a player might be asked to perform in this game. For each category, give:

- "category": a short name for this kind of task (e.g. "opening a locked door", "buying an item from a shop").
- "description": one or two sentences describing the situation this category covers and how a player recognises they are in it.
- "examples": 1-3 concrete instances of this kind of task in this game.
- "insights": 2-5 pieces of concrete, actionable advice for carrying out this kind of task. Name the actual button, the actual menu, the actual object wherever you can.

Requirements:

- Only write about [GAME]. Do NOT pad with advice that would apply to any game ("explore the area", "save often", "be careful"): a generic entry fires on every screen and displaces a useful one.
- If you are unsure whether a detail is true of THIS game, either leave it out or say plainly in the insight what is uncertain.
- If you do not know this game at all, return an empty array [] rather than inventing a plausible-sounding game.

Respond with ONLY a JSON array, no prose before or after it, in exactly this shape:

[
  {
    "category": "...",
    "description": "...",
    "examples": ["...", "..."],
    "insights": ["...", "..."]
  }
]
"""


def _extract_json_array(output: str) -> Optional[list]:
    """Pull the JSON array out of a reply, tolerating fenced or prefaced output.


    :return: The decoded list, or ``None`` if nothing parseable was found.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", output, re.DOTALL)
    candidate = fenced.group(1) if fenced else output

    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _entries_from_payload(payload: list, parameters: Optional[dict]) -> List[Entry]:
    """Turn the decoded JSON into entries, dropping items that carry no usable knowledge.

    An item needs a category and at least one insight to be worth keeping
    """
    entries = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        insights = [str(i).strip() for i in (item.get("insights") or []) if str(i).strip()]
        if not category or not insights:
            continue
        entries.append(Entry(
            category=category,
            description=str(item.get("description") or "").strip(),
            examples=[str(e).strip() for e in (item.get("examples") or []) if str(e).strip()],
            insights=insights,
            # No frame, no init_states: this document never saw the game run.
            frame=None,
        ))

    dropped = len(payload) - len(entries)
    if dropped:
        log_warn(f"[parametric] dropped {dropped} of {len(payload)} generated entries "
                 f"(missing category or insights).", parameters)
    return entries


def generate_parametric_document(
    game: str,
    vlm: VLM,
    n_categories: int = 10,
    max_new_tokens: int = 4000,
    parameters: Optional[dict] = None,
) -> InfoDocument:
    """
    Ask *vlm* to write a document for *game* from its own knowledge.


    :param game: The game identifier, as the benchmark names it.
    :param vlm: The model to ask. Not a supervisor call — no episode is running yet, so
        there is no :class:`~execution.report.SupervisorReport` for it to land on.
    :param n_categories: Upper bound on task categories requested.
    :param max_new_tokens: Token budget for the single generation call.
    :return: A document with ``provenance.source == "parametric"`` and an empty image section.
    """
    result = vlm.infer(
        texts=PARAMETRIC_PROMPT
            .replace("[GAME]", game)
            .replace("[N_CATEGORIES]", str(n_categories)),
        max_new_tokens=max_new_tokens,
    )
    output, meta = result["output"], result["meta"]

    payload = _extract_json_array(output)
    if payload is None:
        log_warn(f"[parametric] could not parse a JSON array from the reply for '{game}'; "
                 f"writing an empty document. First 200 chars: {output[:200]!r}", parameters)
        payload = []

    entries = _entries_from_payload(payload, parameters)
    document = InfoDocument(game=game)
    document.set_entries(TASK_SECTION, entries)
    document.provenance = Provenance(
        source=PARAMETRIC_SOURCE,
        # Neither exists for a document that was not distilled from a run.
        executor=None,
        model=getattr(vlm, "_model_name", None),
        trajectory_stem=None,
        built_at=datetime.now(timezone.utc).isoformat(),
        # What writing this document cost. Must be serialised: callers get a reloaded copy.
        input_tokens=meta["input_tokens"],
        output_tokens=meta["output_tokens"],
    )
    return document


def _warn_if_empty(document: InfoDocument, game: str, path: str, cached: bool,
                   parameters: Optional[dict]) -> None:
    """
    Say loudly that the parametric document carries no task entries.
    """
    if document.task_entries:
        return
    source = "the cached document" if cached else "the model's reply"
    remedy = ("Delete it and rerun to ask again."
              if cached else
              f"'{game}' is not in this model's priors. Use --mode retrieval if you need "
              f"knowledge for this game.")
    log_warn(
        f"[parametric] the document for '{game}' has NO task entries — {source} produced no "
        f"usable knowledge about this game. Every episode will plan WITHOUT insights, which "
        f"makes this run equivalent to the knowledge-free subgoal arm; it is not a "
        f"parametric result even though the CSV is named like one. {remedy} "
        f"The document is at {path}.",
        parameters,
    )


def load_or_generate_parametric_document(
    game: str,
    path: str,
    vlm: VLM,
    n_categories: int = 10,
    max_new_tokens: int = 4000,
    parameters: Optional[dict] = None,
) -> InfoDocument:
    """Return the cached document at *path*, generating and writing it if absent.

    :param path: Where the document is cached, from ``Paths.parametric_doc()``.
    """
    if os.path.exists(path):
        document = load_document(path, parameters=parameters)
        log_info(f"[parametric] reusing cached document for '{game}' — "
                 f"{len(document.task_entries)} task entries ({path})")
        _warn_if_empty(document, game=game, path=path, cached=True, parameters=parameters)
        return document

    log_info(f"[parametric] generating a document for '{game}' from model priors...")
    document = generate_parametric_document(
        game, vlm, n_categories=n_categories, max_new_tokens=max_new_tokens,
        parameters=parameters,
    )

    # Written even when empty: the reply is the evidence for why a run planned without
    # insights, and regenerating it to look at would cost another call and might not
    # reproduce.
    dump_document(document, path)
    log_info(f"[parametric] wrote {len(document.task_entries)} task entries to {path}")

    _warn_if_empty(document, game=game, path=path, cached=False, parameters=parameters)
    return load_document(path, parameters=parameters)
