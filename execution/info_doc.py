"""
The info document: a game's distilled knowledge, as JSON.

The document is the shared artifact of the context-engineering vertical. It has two
producers — vlm_scripts/build_info.py, which distils one from real trajectories, and
execution/parametric_doc.py, which has the model write one from its own priors — is read by
execution.supervisors.InfoPlanSupervisor at test time, and is rendered by
debug_scripts/info.py. This module owns its *shape* so none of them disagree about it.

Exactly two sections of entries, no task-agnostic "general knowledge" section — every insight
hangs off either a task category or an image category::

    {
      "schema": 1,
      "game": "<game>",
      "provenance": {"source": "curiosity", "executor": "history", ...},
      "frames_root": "proposed_tasks/<game>/<model>/.../info_<model>_<executor>",
      "task_entries":  [{"category": ..., "description": ..., "insights": [...], ...}],
      "image_entries": [...]
    }

``description`` / ``examples`` / ``insights`` are kept strictly separate and never collapsed
into prose: the merge decides identity from description + examples + frame and writes only
into insights, which is what makes that decision well-posed rather than a judgement over a
blob of text.

**Why JSON rather than the markdown this used to be.** The markdown was never the format the
model saw (:meth:`Entry.evidence_block` builds that) nor the one humans read
(``debug_scripts/info.py`` renders its own). It was storage only, and as storage it cost a
hand-written parser that had to stay the exact inverse of the renderer, plus three
representational holes — values could not contain a newline, list values could not contain a
comma, and an item reading exactly ``(none recorded)`` was silently dropped. Serialising to
JSON makes the round trip free and removes all three. :func:`render_document` survives as a
**one-way** markdown view for humans.

**Provenance is recorded, not inferred.** It used to be reconstructed at load time by parsing
the directory the document was found in, which meant every reader re-derived it and
``execution/`` had to import a top-level CLI script to do so. ``build_info`` knows all of it
at write time and now stamps it in.

**Frame paths.** Entries store a path relative to :attr:`InfoDocument.frames_root`, which is
itself relative to ``storage_dir``. That keeps a document portable — an absolute path would
pin the artifact to one machine — while staying unambiguous once entries from several
documents are unioned. :func:`load_document` resolves each entry's
:attr:`Entry.resolved_frame` so readers never have to know the scheme.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from utils.fundamental import file_makedir
from utils.parameter_handling import load_parameters

#: Bumped when the on-disk shape changes incompatibly, so a stale document is detected
#: rather than silently misread.
SCHEMA_VERSION = 1

TASK_SECTION = "Task Understanding"
IMAGE_SECTION = "Image Understanding"


@dataclass
class Provenance:
    """Where a document came from, recorded by its producer.

    Every field is written by ``build_info`` at the moment the document is built. Nothing
    here is ever re-derived from a path: that is the whole point of the block existing.

    :param source: The vertical — ``"curiosity"`` or ``"zeroshot"`` for a document distilled
        from trajectories, or ``"parametric"`` for one written from the model's own priors by
        ``execution/parametric_doc.py``. This alone is the provenance label.
    :param executor: Executor whose trajectories were distilled. ``None`` for a parametric
        document, which distilled nothing.
    :param model: Model that did the distilling, or the writing (the save name, not the
        full path).
    :param trajectory_stem: The input stem, relative to ``storage_dir``. The curiosity
        run name is a component of this, so it is not recorded a second time. ``None`` for
        a parametric document, which has no input.
    :param built_at: ISO-8601 UTC timestamp.
    """

    source: str = ""
    executor: Optional[str] = None
    model: Optional[str] = None
    trajectory_stem: Optional[str] = None
    built_at: Optional[str] = None

    @property
    def label(self) -> str:
        """The single string readers use to attribute an entry.

        Matches what :func:`utils.paths.source_label` reconstructs from directory names:
        the vertical the trajectories came from.
        """
        return self.source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "executor": self.executor,
            "model": self.model,
            "trajectory_stem": self.trajectory_stem,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Provenance":
        data = data or {}
        return cls(
            source=data.get("source", ""),
            executor=data.get("executor"),
            model=data.get("model"),
            trajectory_stem=data.get("trajectory_stem"),
            built_at=data.get("built_at"),
        )


@dataclass
class Entry:
    """One task or image category, with its evidence and its advice kept apart."""

    category: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    frame: Optional[str] = None          # relative to the document's frames_root
    example_frames: List[str] = field(default_factory=list)
    init_states: List[str] = field(default_factory=list)

    #: Provenance label, set by :func:`load_document` from the document's
    #: :class:`Provenance`. Not persisted per entry — it belongs to the document — but it
    #: is carried on the entry because a union mixes entries from several documents into
    #: one candidate list, after which the entry is all a reader has.
    source: Optional[str] = None

    #: Absolute path to :attr:`frame`, resolved by :func:`load_document`. Not persisted.
    #: ``None`` when the entry has no frame, or when the document was built in memory
    #: rather than loaded.
    resolved_frame: Optional[str] = None

    def evidence_block(self, include_frame_note: bool = False) -> str:
        """Description + Examples — the fields identity is judged from. Never Insights."""
        lines = [f"Category: {self.category}", f"Description: {self.description}"]
        if self.examples:
            lines.append("Examples:")
            lines.extend(f"- {e}" for e in self.examples)
        if include_frame_note and self.frame:
            lines.append("(a representative frame for this entry is shown as an image)")
        return "\n".join(lines)

    def insights_block(self) -> str:
        return "\n".join(f"- {i}" for i in self.insights)

    def to_dict(self) -> Dict[str, Any]:
        """Persisted fields only — ``source`` and ``resolved_frame`` are load-time."""
        return {
            "category": self.category,
            "description": self.description,
            "examples": list(self.examples),
            "insights": list(self.insights),
            "frame": self.frame,
            "example_frames": list(self.example_frames),
            "init_states": list(self.init_states),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entry":
        return cls(
            category=data.get("category", ""),
            description=data.get("description", ""),
            examples=list(data.get("examples") or []),
            insights=list(data.get("insights") or []),
            frame=data.get("frame"),
            example_frames=list(data.get("example_frames") or []),
            init_states=list(data.get("init_states") or []),
        )


@dataclass
class InfoDocument:
    game: str = ""
    task_entries: List[Entry] = field(default_factory=list)
    image_entries: List[Entry] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    #: Directory the entries' frame paths are relative to, itself relative to
    #: ``storage_dir``. Empty for a document built in memory that has not been written yet.
    frames_root: str = ""

    @staticmethod
    def _check_section(section: str) -> None:
        """Reject an unrecognised section name.

        Both accessors used to treat anything that was not :data:`TASK_SECTION` as the
        image section, so a typo silently read — or, through ``set_entries``, silently
        overwrote — the wrong list.
        """
        if section not in (TASK_SECTION, IMAGE_SECTION):
            raise ValueError(
                f"Unknown info-document section {section!r}. "
                f"Expected {TASK_SECTION!r} or {IMAGE_SECTION!r}."
            )

    def entries(self, section: str) -> List[Entry]:
        self._check_section(section)
        return self.task_entries if section == TASK_SECTION else self.image_entries

    def set_entries(self, section: str, entries: List[Entry]) -> None:
        self._check_section(section)
        if section == TASK_SECTION:
            self.task_entries = entries
        else:
            self.image_entries = entries

    @property
    def n_entries(self) -> int:
        return len(self.task_entries) + len(self.image_entries)

    def copy(self) -> "InfoDocument":
        """Deep-ish copy: new Entry objects with their own list fields."""
        def copied(entries: List[Entry]) -> List[Entry]:
            return [replace(e, examples=list(e.examples), insights=list(e.insights),
                            example_frames=list(e.example_frames),
                            init_states=list(e.init_states))
                    for e in entries]

        return InfoDocument(
            game=self.game,
            task_entries=copied(self.task_entries),
            image_entries=copied(self.image_entries),
            provenance=replace(self.provenance),
            frames_root=self.frames_root,
        )

    def to_dict(self) -> Dict[str, Any]:
        """The persisted shape. Inverse of :meth:`from_dict`."""
        return {
            "schema": SCHEMA_VERSION,
            "game": self.game,
            "provenance": self.provenance.to_dict(),
            "frames_root": self.frames_root,
            "task_entries": [e.to_dict() for e in self.task_entries],
            "image_entries": [e.to_dict() for e in self.image_entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfoDocument":
        """Inverse of :meth:`to_dict`.

        Does **not** populate :attr:`Entry.source` or :attr:`Entry.resolved_frame` — those
        are load-time concerns and belong to :func:`load_document`, which knows where the
        document came from. An in-memory document built during a merge has neither.
        """
        schema = data.get("schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"Info document has schema {schema}, this code understands "
                f"{SCHEMA_VERSION}. Rebuild it with vlm_scripts/build_info.py."
            )
        return cls(
            game=data.get("game", ""),
            task_entries=[Entry.from_dict(e) for e in data.get("task_entries") or []],
            image_entries=[Entry.from_dict(e) for e in data.get("image_entries") or []],
            provenance=Provenance.from_dict(data.get("provenance")),
            frames_root=data.get("frames_root", ""),
        )


# ---------------------------------------------------------------------------
# Markdown view (one-way — for humans, never parsed back)
# ---------------------------------------------------------------------------

_TASK_HEADING = "Task category:"
_IMAGE_HEADING = "Image category:"


def _render_entry(entry: Entry, heading: str) -> str:
    lines = [f"### {heading} {entry.category}", f"**Description:** {entry.description}"]
    if entry.frame:
        lines.append(f"**Frame:** {entry.frame}")
    if entry.init_states:
        lines.append(f"**Init states:** {', '.join(entry.init_states)}")
    lines.append("**Examples:**")
    lines.extend(f"- {e}" for e in entry.examples or ["(none recorded)"])
    if entry.example_frames:
        lines.append(f"**Example frames:** {', '.join(entry.example_frames)}")
    lines.append("**Insights:**")
    lines.extend(f"- {i}" for i in entry.insights or ["(none recorded)"])
    return "\n".join(lines)


def render_document(doc: InfoDocument) -> str:
    """Render a document as markdown, for reading.

    **One-way.** There is no parser for this text and there should not be one: JSON is the
    stored format, and a second representation that had to round-trip is exactly the
    coupling this module shed. Use it for display (``debug_scripts/info.py``) and for
    eyeballing a document; use :func:`load_document` / :func:`dump_document` for anything
    a program reads back.
    """
    out = [f"# Info: {doc.game}", ""]
    if doc.provenance.source:
        out += [f"*Source: {doc.provenance.label}*", ""]
    out += [f"## 1. {TASK_SECTION}", ""]
    for entry in doc.task_entries:
        out.append(_render_entry(entry, _TASK_HEADING))
        out.append("")
    out.append(f"## 2. {IMAGE_SECTION}")
    out.append("")
    for entry in doc.image_entries:
        out.append(_render_entry(entry, _IMAGE_HEADING))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Load / dump
# ---------------------------------------------------------------------------


def resolve_frame(doc: InfoDocument, frame: Optional[str],
                  parameters: Optional[dict] = None) -> Optional[str]:
    """Turn one of ``doc``'s frame paths into an absolute path that can be opened.

    Entry frame paths are relative to ``doc.frames_root``, which is relative to
    ``storage_dir``. Most readers do not need this — :func:`load_document` has already
    filled in :attr:`Entry.resolved_frame` — but a document built in memory has not been
    through that, so the resolution lives here rather than only inside the loader.

    :param doc: The document the path belongs to.
    :param frame: A stored frame path, or ``None``.
    :param parameters: Loaded parameters dict. If None, loads from config.
    :return: An absolute path, or ``None`` if ``frame`` was ``None``.
    """
    if not frame:
        return None
    if os.path.isabs(frame):
        return frame
    storage_dir = load_parameters(parameters)["storage_dir"]
    return os.path.join(storage_dir, doc.frames_root, frame)


def dump_document(doc: InfoDocument, path: str) -> None:
    """Write a document to ``path`` as JSON, atomically.

    Atomic because the merge tree writes a node per round and a partial file read back on
    resume would be indistinguishable from a completed one.
    """
    file_makedir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(doc.to_dict(), handle, indent=2)
    os.replace(tmp, path)


def load_document(path: str, parameters: Optional[dict] = None) -> InfoDocument:
    """
    Read a document and fill in the two load-time fields on every entry.

    :attr:`Entry.source` comes from the document's own recorded
    :attr:`Provenance.label` — not from the directory ``path`` sits in. It is copied onto
    each entry because a union mixes entries from several documents into one list, after
    which the entry is all a reader has to attribute it by.

    :attr:`Entry.resolved_frame` is the absolute path to the entry's frame, so no reader
    has to know that stored paths are relative to ``frames_root``.
    """
    with open(path, "r") as handle:
        doc = InfoDocument.from_dict(json.load(handle))

    parameters = load_parameters(parameters)
    label = doc.provenance.label
    for entry in doc.task_entries + doc.image_entries:
        entry.source = label or None
        entry.resolved_frame = resolve_frame(doc, entry.frame, parameters)
    return doc
