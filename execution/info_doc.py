"""
The info document: a game's distilled knowledge as plain markdown.

The document is the shared artifact of the context-engineering vertical. It is written by
vlm_scripts/build_info.py, read by execution.supervisors.InfoHintSupervisor at test time, and
rendered by debug_scripts/info.py. This module owns its *shape* so those three never disagree
about it.

Format (exactly two sections, no task-agnostic "general knowledge" section — every insight
hangs off either a task category or an image category)::

    # Info: <game>

    ## 1. Task Understanding

    ### Task category: <short name>
    **Description:** <what this category of task is>
    **Frame:** frames/<id>.png
    **Examples:**
    - <what was attempted and what happened>
    **Insights:**
    - <non-obvious insight>

    ## 2. Image Understanding

    ### Image category: <short name>
    ...

``Description`` / ``Examples`` / ``Insights`` are kept strictly separate and never collapsed
into prose: the merge decides identity from Description + Examples + Frame and writes only
into Insights, which is what makes that decision well-posed rather than a judgement over a
blob of text.

Round-tripping is exact for documents this module wrote (``parse_document(render_document(d))
== d``), which is what lets the merge tree checkpoint to disk as markdown rather than as an
opaque pickle.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import List, Optional

TASK_SECTION = "Task Understanding"
IMAGE_SECTION = "Image Understanding"

_TASK_HEADING = "Task category:"
_IMAGE_HEADING = "Image category:"


@dataclass
class Entry:
    """One task or image category, with its evidence and its advice kept apart."""

    category: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    frame: Optional[str] = None          # path relative to the info dir
    example_frames: List[str] = field(default_factory=list)
    init_states: List[str] = field(default_factory=list)
    source: Optional[str] = None         # provenance label, set at load time (§4.3)

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


@dataclass
class InfoDocument:
    game: str = ""
    task_entries: List[Entry] = field(default_factory=list)
    image_entries: List[Entry] = field(default_factory=list)

    def entries(self, section: str) -> List[Entry]:
        return self.task_entries if section == TASK_SECTION else self.image_entries

    def set_entries(self, section: str, entries: List[Entry]) -> None:
        if section == TASK_SECTION:
            self.task_entries = entries
        else:
            self.image_entries = entries

    @property
    def n_entries(self) -> int:
        return len(self.task_entries) + len(self.image_entries)

    def copy(self) -> "InfoDocument":
        return InfoDocument(
            game=self.game,
            task_entries=[replace(e, examples=list(e.examples), insights=list(e.insights),
                                  example_frames=list(e.example_frames),
                                  init_states=list(e.init_states))
                          for e in self.task_entries],
            image_entries=[replace(e, examples=list(e.examples), insights=list(e.insights),
                                   example_frames=list(e.example_frames),
                                   init_states=list(e.init_states))
                           for e in self.image_entries],
        )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


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
    """Render to the markdown of the module docstring. Inverse of :func:`parse_document`."""
    out = [f"# Info: {doc.game}", "", f"## 1. {TASK_SECTION}", ""]
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
# Parse
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(r"^\*\*(?P<name>[^:*]+):\*\*\s*(?P<value>.*)$")


def _split_csv(value: str) -> List[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def _parse_entries(block: str, heading: str) -> List[Entry]:
    entries: List[Entry] = []
    current: Optional[Entry] = None
    bullet_target: Optional[str] = None

    for raw in block.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("### "):
            title = stripped[4:].strip()
            if title.lower().startswith(heading.lower()):
                title = title[len(heading):].strip()
            current = Entry(category=title)
            entries.append(current)
            bullet_target = None
            continue

        if current is None:
            continue

        match = _FIELD_RE.match(stripped)
        if match:
            name = match.group("name").strip().lower()
            value = match.group("value").strip()
            bullet_target = None
            if name == "description":
                current.description = value
            elif name == "frame":
                current.frame = value or None
            elif name == "init states":
                current.init_states = _split_csv(value)
            elif name == "example frames":
                current.example_frames = _split_csv(value)
            elif name == "examples":
                bullet_target = "examples"
            elif name == "insights":
                bullet_target = "insights"
            continue

        if stripped.startswith("- ") and bullet_target:
            item = stripped[2:].strip()
            if item and item != "(none recorded)":
                getattr(current, bullet_target).append(item)

    return entries


def parse_document(text: str) -> InfoDocument:
    """Parse the markdown of the module docstring. Inverse of :func:`render_document`."""
    game = ""
    for line in text.splitlines():
        if line.startswith("# Info:"):
            game = line[len("# Info:"):].strip()
            break

    task_block, image_block = "", ""
    section = None
    buckets = {"task": [], "image": []}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            lowered = stripped.lower()
            if TASK_SECTION.lower() in lowered:
                section = "task"
                continue
            if IMAGE_SECTION.lower() in lowered:
                section = "image"
                continue
            section = None
            continue
        if section:
            buckets[section].append(line)

    task_block = "\n".join(buckets["task"])
    image_block = "\n".join(buckets["image"])

    return InfoDocument(
        game=game,
        task_entries=_parse_entries(task_block, _TASK_HEADING),
        image_entries=_parse_entries(image_block, _IMAGE_HEADING),
    )


def load_document(path: str, source: Optional[str] = None) -> InfoDocument:
    """
    Read and parse an info document, tagging every entry with a provenance label.

    Frame paths are rewritten to absolute so an entry stays loadable once several documents
    from different directories are unioned together (§4.3).
    """
    with open(path, "r") as handle:
        doc = parse_document(handle.read())

    root = os.path.dirname(os.path.abspath(path))
    for entry in doc.task_entries + doc.image_entries:
        entry.source = source
        if entry.frame and not os.path.isabs(entry.frame):
            entry.frame = os.path.join(root, entry.frame)
        entry.example_frames = [
            f if os.path.isabs(f) else os.path.join(root, f) for f in entry.example_frames
        ]
    return doc


def source_label(info_dir: str) -> str:
    """
    Provenance label for an info dir, reusing create_dataset's labelling.

    ``create_dataset._source_label`` already knows every naming quirk of these directory
    layouts (including the ``--extra curiosity`` collision it documents), and an info dir sits
    at exactly the same depth as a practice dir in both verticals — ``.../curiosity/<run>/
    info_<model>`` and ``.../<tasks>_<executor>_attempts/info_<model>`` — so it labels both
    correctly with no second scheme to keep in sync.
    """
    from create_dataset import _source_label

    return _source_label(os.path.normpath(info_dir))
