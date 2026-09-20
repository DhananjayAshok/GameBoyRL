"""
The knowledge base: short facts the strategist has learned, tagged by kind and by the
location they were learned in.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import file_makedir

KINDS = ("map", "npc", "item", "mechanic", "blocker", "objective")
DEFAULT_KIND = "mechanic"
RENDER_LIMIT = 3000

_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", (text or "").lower()).split())


def clean_kind(kind: Optional[str]) -> str:
    kind = (kind or "").strip().lower()
    return kind if kind in KINDS else DEFAULT_KIND


@dataclass
class Fact:
    id: int
    text: str
    kind: str
    location: Optional[str] = None
    first_seen_episode: Optional[int] = None
    last_seen_episode: Optional[int] = None
    uses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(
            id=int(data["id"]),
            text=data["text"],
            kind=clean_kind(data.get("kind")),
            location=data.get("location"),
            first_seen_episode=data.get("first_seen_episode"),
            last_seen_episode=data.get("last_seen_episode"),
            uses=int(data.get("uses", 0)),
        )

    def render(self, with_location: bool = False) -> str:
        where = f", {self.location}" if with_location and self.location else ""
        return f"- ({self.kind}{where}) {self.text}"


class KnowledgeBase:
    def __init__(self, facts: Optional[List[Fact]] = None) -> None:
        self.facts: List[Fact] = list(facts or [])
        self._next_id = max((fact.id for fact in self.facts), default=0) + 1

    def __len__(self) -> int:
        return len(self.facts)

    # -- adding ----------------------------------------------------------

    def duplicate_of(self, text: str) -> Optional[Fact]:
        wanted = normalise(text)
        if not wanted:
            return None
        for fact in self.facts:
            existing = normalise(fact.text)
            if existing == wanted:
                return fact
            if len(wanted) > 20 and (wanted in existing or existing in wanted):
                return fact
        return None

    def add(self, text: str, kind: Optional[str] = None, location: Optional[str] = None,
            episode: Optional[int] = None) -> Optional[Fact]:
        text = (text or "").strip()
        if not text:
            return None
        existing = self.duplicate_of(text)
        if existing is not None:
            existing.last_seen_episode = episode if episode is not None else existing.last_seen_episode
            existing.uses += 1
            return None
        fact = Fact(id=self._next_id, text=text, kind=clean_kind(kind), location=location,
                    first_seen_episode=episode, last_seen_episode=episode)
        self._next_id += 1
        self.facts.append(fact)
        return fact

    def add_many(self, entries: Iterable[Tuple[str, Optional[str]]], location: Optional[str] = None,
                 episode: Optional[int] = None) -> List[Fact]:
        added = []
        for text, kind in entries:
            fact = self.add(text, kind, location, episode)
            if fact is not None:
                added.append(fact)
        return added

    # -- reading ---------------------------------------------------------

    def render(self, location: Optional[str] = None, limit: int = RENDER_LIMIT) -> str:
        here = [f for f in self.facts if location and f.location == location]
        game_wide = [f for f in self.facts if f.location is None]
        seen = {id(f) for f in here} | {id(f) for f in game_wide}
        elsewhere = [f for f in self.facts if id(f) not in seen]
        sections = [
            (f"What you know about {location}:" if location else None, here, False),
            ("What you know about the game:", game_wide, False),
            ("What you know about other places:", elsewhere, True),
        ]
        lines: List[str] = []
        used = 0
        for heading, facts, with_location in sections:
            if not facts or heading is None:
                continue
            block = [heading] + [fact.render(with_location) for fact in facts]
            for line in block:
                if used + len(line) + 1 > limit:
                    lines.append("... (older facts not shown)")
                    return "\n".join(lines)
                lines.append(line)
                used += len(line) + 1
            lines.append("")
            used += 1
        rendered = "\n".join(lines).strip()
        return rendered or "You have not learned anything yet."

    def render_all(self) -> str:
        if not self.facts:
            return "(the knowledge base is empty)"
        return "\n".join(
            f"- ({fact.kind}, {fact.location or 'game-wide'}) {fact.text}" for fact in self.facts
        )

    def counts_by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for fact in self.facts:
            counts[fact.kind] = counts.get(fact.kind, 0) + 1
        return counts

    # -- reorganising ----------------------------------------------------

    def replace(self, entries: Iterable[Tuple[str, Optional[str], Optional[str]]],
                episode: Optional[int] = None) -> None:
        by_text = {normalise(fact.text): fact for fact in self.facts}
        rebuilt: List[Fact] = []
        next_id = 1
        for text, kind, location in entries:
            text = (text or "").strip()
            if not text:
                continue
            old = by_text.get(normalise(text))
            rebuilt.append(Fact(
                id=next_id,
                text=text,
                kind=clean_kind(kind),
                location=location,
                first_seen_episode=old.first_seen_episode if old else episode,
                last_seen_episode=episode,
                uses=old.uses if old else 0,
            ))
            next_id += 1
        self.facts = rebuilt
        self._next_id = next_id

    # -- persistence -----------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        return [fact.to_dict() for fact in self.facts]

    def save(self, path: str) -> str:
        file_makedir(path)
        with open(path, "w") as f:
            json.dump({"next_id": self._next_id, "facts": self.to_list()}, f, indent=2)
        return path

    def archive(self, path: str) -> Optional[str]:
        """Copy the current file aside as ``knowledge_<n>.json`` before it is rewritten."""
        if not os.path.exists(path):
            return None
        stem, extension = os.path.splitext(path)
        n = 1
        while os.path.exists(f"{stem}_{n}{extension}"):
            n += 1
        archived = f"{stem}_{n}{extension}"
        with open(path) as source, open(archived, "w") as target:
            target.write(source.read())
        return archived

    @classmethod
    def load(cls, path: str, missing_ok: bool = True) -> Optional["KnowledgeBase"]:
        if not os.path.exists(path):
            if missing_ok:
                return None
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        base = cls([Fact.from_dict(entry) for entry in data.get("facts", [])])
        base._next_id = max(base._next_id, int(data.get("next_id", 0)))
        return base
