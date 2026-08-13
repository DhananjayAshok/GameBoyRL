"""
The parametric document: what the model already believes about a game, in document form.

The second way an :class:`~execution.info_doc.InfoDocument` can come into existence. The
first is ``vlm_scripts/build_info.py``, which distils one out of real trajectories. This
module asks the model to write one from its own weights, given nothing but the game's name —
no frames, no playthroughs, no retrieval of any kind.

**Why it exists.** The plan arm's result has always been "planning from a distilled document
beats the no-knowledge baseline". That comparison cannot separate two very different claims:
that *distilling trajectories* worked, or that *any* game-specific text in the prompt worked.
A document written from the model's priors is the control that separates them — same
document schema, same relevance pass, same planner, same budgets, differing only in where the
entries came from. Where the parametric document scores like the built one, the pipeline that
produced the built one has not earned its cost.

**What it is not.** Nothing here is verified against the game. On a well-known title the model
has genuine knowledge; on an obscure one it will confabulate fluently and the document will
look exactly as confident either way. That is the finding rather than a bug, and it is the
reason the document is written to disk and kept: a run whose plans came from invented level
geometry has to remain auditable after the fact.

**Task entries only.** The schema's image section describes kinds of screen, identified from
frames the builder actually saw. A model that has seen no screens has nothing well-founded to
put there, and an invented image category is worse than an absent one — it would be matched
against a real screen by the relevance pass. So the image section is left empty, and
``InfoSubgoalSupervisor._candidates`` yields nothing for it without needing a special case.

Frames are likewise absent: every entry has ``frame=None``, which
:func:`~execution.info_doc.resolve_frame` maps to ``None`` and the relevance call handles by
switching to its no-frame prompt wording.
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

#: ``Provenance.source`` for a document from this module. Deliberately not ``"zeroshot"``:
#: that label already means the zero-shot *task proposal* vertical, whose documents are
#: distilled from real attempted trajectories. Confusing the two would make a document's
#: provenance block a lie about how it was produced.
PARAMETRIC_SOURCE = "parametric"

PARAMETRIC_PROMPT = """You are writing a knowledge base for the Game Boy game [GAME], from what you already know about it. You will not be shown any screenshots — write only from your own knowledge of this game.

Produce up to [N_CATEGORIES] categories of task a player might be asked to perform in this game. For each category, give:

- "category": a short name for this kind of task (e.g. "opening a locked door", "buying an item from a shop").
- "description": one or two sentences describing the situation this category covers and how a player recognises they are in it.
- "examples": 1-3 concrete instances of this kind of task in this game.
- "insights": 2-5 pieces of concrete, actionable advice for carrying out this kind of task. Name the actual button, the actual menu, the actual object wherever you can.

Requirements:

- Only write about [GAME]. Do NOT pad with advice that would apply to any game ("explore the area", "save often", "be careful"): a generic entry fires on every screen and displaces a useful one.
- If you are unsure whether a detail is true of THIS game, either leave it out or say plainly in the insight what is uncertain. A confidently wrong instruction is followed until the step limit; an absent one costs nothing.
- If you do not know this game at all, return an empty array [] rather than inventing a plausible-sounding game. An empty document is a real and useful answer.

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

    Parsed rather than trusted: models wrap JSON in ``` fences or precede it with a
    sentence often enough that a bare ``json.loads`` on the whole reply fails on
    well-formed content. The array is located by its outermost brackets, which is
    sufficient here because the requested shape has no array nested above the top level.

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

    An item needs a category and at least one insight to be worth keeping: the category is
    what the relevance pass judges and the insights are the only thing the planner ever
    reads, so an item missing either contributes nothing but a wasted VLM call per episode.
    Malformed items are skipped individually rather than failing the whole document — one
    bad element out of ten is not a reason to fall back to no knowledge at all.
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
    """Ask *vlm* to write a document for *game* from its own knowledge.

    One call. The reply is parsed strictly and a document that comes back empty stays empty
    — :meth:`InfoSubgoalSupervisor.write_plan` already degrades to running the task unplanned
    when nothing is selected, and that is the honest behaviour for a game the model does not
    know. Inventing entries to avoid an empty document would be exactly the failure this
    control exists to detect.

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
        # No executor and no trajectory_stem: neither exists for a document that was not
        # distilled from a run. Left None rather than filled with a plausible value, so the
        # provenance block cannot be mistaken for a built document's.
        executor=None,
        model=getattr(vlm, "_model_name", None),
        trajectory_stem=None,
        built_at=datetime.now(timezone.utc).isoformat(),
        # What writing this document cost. On the provenance rather than returned,
        # because load_or_generate_parametric_document hands back a reloaded copy — see
        # its docstring — so anything not serialised is lost on the generating run too.
        input_tokens=meta["input_tokens"],
        output_tokens=meta["output_tokens"],
    )
    return document


def load_or_generate_parametric_document(
    game: str,
    path: str,
    vlm: VLM,
    n_categories: int = 10,
    max_new_tokens: int = 4000,
    regenerate: bool = False,
    parameters: Optional[dict] = None,
) -> InfoDocument:
    """Return the cached document at *path*, generating and writing it if absent.

    Cached per (game, model) rather than regenerated per run so that two runs of the same
    command plan from the same document. Without that, every rerun silently redraws the
    knowledge and a change in score cannot be attributed to whatever was actually changed.

    Reloaded through :func:`~execution.info_doc.load_document` even on the generating run, so
    the freshly written and the cached paths return an identically populated object —
    ``Entry.source`` set from provenance, ``resolved_frame`` resolved — instead of differing
    in exactly the fields a reader forgets to check.

    :param path: Where the document is cached, from ``Paths.parametric_doc()``.
    :param regenerate: Overwrite an existing cached document instead of reusing it.
    """
    if os.path.exists(path) and not regenerate:
        document = load_document(path, parameters=parameters)
        log_info(f"[parametric] reusing cached document for '{game}' — "
                 f"{len(document.task_entries)} task entries ({path})")
        return document

    log_info(f"[parametric] generating a document for '{game}' from model priors...")
    document = generate_parametric_document(
        game, vlm, n_categories=n_categories, max_new_tokens=max_new_tokens,
        parameters=parameters,
    )
    dump_document(document, path)
    log_info(f"[parametric] wrote {len(document.task_entries)} task entries to {path}")

    if not document.task_entries:
        log_warn(f"[parametric] the document for '{game}' is EMPTY — the model produced no "
                 f"usable knowledge about this game. Every episode will run unplanned, which "
                 f"is the baseline with extra steps. Check {path}.", parameters)
    return load_document(path, parameters=parameters)
