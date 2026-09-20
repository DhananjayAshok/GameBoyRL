"""
Where the player is, and how they got there.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from utils import file_makedir

MAX_TRANSITIONS_RENDERED = 12


@dataclass
class Transition:
    origin: Optional[str]
    destination: str
    how: Optional[str] = None
    episode: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transition":
        return cls(origin=data.get("origin"), destination=data["destination"],
                   how=data.get("how"), episode=data.get("episode"))

    def render(self) -> str:
        how = f" ({self.how})" if self.how else ""
        return f"- {self.origin or 'somewhere'} -> {self.destination}{how}"


class LocationTracker:
    def __init__(self, current: Optional[str] = None, previous: Optional[str] = None,
                 transitions: Optional[List[Transition]] = None,
                 names: Optional[List[str]] = None) -> None:
        self.current = current
        self.previous = previous
        self.transitions: List[Transition] = list(transitions or [])
        self.names: List[str] = list(names or ([current] if current else []))

    # -- updating --------------------------------------------------------

    def canonical(self, name: str) -> Optional[str]:
        """The already-known spelling of ``name``, or ``None`` if it is new."""
        wanted = (name or "").strip().lower()
        if not wanted:
            return None
        return next((known for known in self.names if known.lower() == wanted), None)

    def update(self, name: Optional[str], changed: bool, how: Optional[str] = None,
               episode: Optional[int] = None) -> bool:
        """Record where the player is now. Returns whether the location actually moved."""
        name = (name or "").strip().strip('"\'')
        if not name:
            return False
        name = self.canonical(name) or name
        if name not in self.names:
            self.names.append(name)
        if self.current is None:
            self.current = name
            self.transitions.append(Transition(None, name, how, episode))
            return True
        if name == self.current:
            return False
        if not changed:
            # The model named a place but said nothing moved; trust the name, not the flag.
            changed = True
        self.previous = self.current
        self.current = name
        self.transitions.append(Transition(self.previous, name, how, episode))
        return changed

    # -- rendering -------------------------------------------------------

    def known_names(self) -> str:
        return ", ".join(self.names) if self.names else "(none yet)"

    def render_transitions(self, limit: int = MAX_TRANSITIONS_RENDERED) -> str:
        if not self.transitions:
            return "(no moves between places recorded yet)"
        return "\n".join(t.render() for t in self.transitions[-limit:])

    def neighbours(self, location: Optional[str] = None) -> List[str]:
        location = location or self.current
        out: List[str] = []
        for transition in self.transitions:
            if transition.origin == location and transition.destination not in out:
                out.append(transition.destination)
            elif transition.destination == location and transition.origin and transition.origin not in out:
                out.append(transition.origin)
        return out

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "previous": self.previous,
            "names": list(self.names),
            "transitions": [t.to_dict() for t in self.transitions],
        }

    def save(self, path: str) -> str:
        file_makedir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str, missing_ok: bool = True) -> Optional["LocationTracker"]:
        if not os.path.exists(path):
            if missing_ok:
                return None
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        return cls(
            current=data.get("current"),
            previous=data.get("previous"),
            transitions=[Transition.from_dict(entry) for entry in data.get("transitions", [])],
            names=data.get("names", []),
        )
