"""
The goal tree: a fixed ladder of long-horizon goals with discovered and synthesised
subgoals hanging off whichever rung is current.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from utils import file_makedir

#: The rungs every Pokemon title shares: eight badges in order, then the Elite Four.
#:
#: Deliberately unnamed. Naming the gyms, their badges or their towns would hand the agent
#: the progression knowledge the benchmark exists to test it on discovering, and for a title
#: set outside the region those names come from it would be wrong as well as unearned. The
#: ordinal is the only part that is true of the series rather than of one game.
BADGE_LADDER = [
    "Earn the first gym badge",
    "Earn the second gym badge",
    "Earn the third gym badge",
    "Earn the fourth gym badge",
    "Earn the fifth gym badge",
    "Earn the sixth gym badge",
    "Earn the seventh gym badge",
    "Earn the eighth gym badge",
    "Defeat the Elite Four",
]

STATUSES = ("pending", "active", "done", "abandoned")
SOURCES = ("ladder", "discovered", "synthesised")
OPEN_STATUSES = ("pending", "active")


@dataclass
class Goal:
    id: int
    text: str
    parent_id: Optional[int] = None
    status: str = "pending"
    source: str = "ladder"
    episodes: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            id=int(data["id"]),
            text=data["text"],
            parent_id=data.get("parent_id"),
            status=data.get("status", "pending"),
            source=data.get("source", "ladder"),
            episodes=list(data.get("episodes", [])),
            notes=list(data.get("notes", [])),
        )


class GoalTree:
    def __init__(self, goals: Optional[List[Goal]] = None) -> None:
        self.goals: List[Goal] = list(goals or [])
        self._next_id = max((goal.id for goal in self.goals), default=0) + 1

    @classmethod
    def ladder(cls, texts: Optional[List[str]] = None) -> "GoalTree":
        tree = cls()
        for text in texts or BADGE_LADDER:
            tree.add(text, parent_id=None, source="ladder")
        return tree

    # -- structure -------------------------------------------------------

    def add(
        self,
        text: str,
        parent_id: Optional[int] = None,
        source: str = "discovered",
        episode: Optional[int] = None,
    ) -> Goal:
        goal = Goal(
            id=self._next_id, text=text.strip(), parent_id=parent_id, source=source
        )
        if episode is not None:
            goal.episodes.append(episode)
        self._next_id += 1
        self.goals.append(goal)
        return goal

    def get(self, goal_id: int) -> Optional[Goal]:
        return next((goal for goal in self.goals if goal.id == goal_id), None)

    def children(self, goal_id: Optional[int]) -> List[Goal]:
        return [goal for goal in self.goals if goal.parent_id == goal_id]

    def roots(self) -> List[Goal]:
        return self.children(None)

    def descendants(self, goal_id: int) -> List[Goal]:
        out: List[Goal] = []
        for child in self.children(goal_id):
            out.append(child)
            out.extend(self.descendants(child.id))
        return out

    # -- the queue -------------------------------------------------------

    def current_ladder_goal(self) -> Optional[Goal]:
        return next(
            (goal for goal in self.roots() if goal.status in OPEN_STATUSES), None
        )

    def next_goal(self) -> Optional[Goal]:
        ladder_goal = self.current_ladder_goal()
        if ladder_goal is None:
            return None
        return self._first_open_leaf(ladder_goal)

    def _first_open_leaf(self, goal: Goal) -> Optional[Goal]:
        for child in self.children(goal.id):
            if child.status not in OPEN_STATUSES:
                continue
            found = self._first_open_leaf(child)
            if found is not None:
                return found
        return goal if goal.status in OPEN_STATUSES else None

    def open_subgoals(self, ladder_goal: Optional[Goal] = None) -> List[Goal]:
        ladder_goal = ladder_goal or self.current_ladder_goal()
        if ladder_goal is None:
            return []
        return [
            goal
            for goal in self.descendants(ladder_goal.id)
            if goal.status in OPEN_STATUSES
        ]

    def any_open(self) -> bool:
        return self.next_goal() is not None

    # -- mutation --------------------------------------------------------

    def mark(
        self,
        goal_id: int,
        status: str,
        note: Optional[str] = None,
        episode: Optional[int] = None,
    ) -> Optional[Goal]:
        goal = self.get(goal_id)
        if goal is None or status not in STATUSES:
            return None
        goal.status = status
        if note:
            goal.notes.append(note)
        if episode is not None and episode not in goal.episodes:
            goal.episodes.append(episode)
        return goal

    def touch(self, goal_id: int, episode: int) -> None:
        goal = self.get(goal_id)
        if goal is not None and episode not in goal.episodes:
            goal.episodes.append(episode)

    def abandon_matching(
        self, text: str, reason: str = "", episode: Optional[int] = None
    ) -> Optional[Goal]:
        wanted = _normalise(text)
        if not wanted:
            return None
        for goal in self.open_subgoals():
            normalised = _normalise(goal.text)
            if normalised == wanted or wanted in normalised or normalised in wanted:
                return self.mark(goal.id, "abandoned", reason, episode)
        return None

    # -- rendering -------------------------------------------------------

    def render_open(self, ladder_goal: Optional[Goal] = None) -> str:
        subgoals = self.open_subgoals(ladder_goal)
        if not subgoals:
            return "(no open subgoals)"
        return "\n".join(f"- {goal.text}" for goal in subgoals)

    def render_path(self, goal: Goal) -> str:
        chain = [goal.text]
        parent_id = goal.parent_id
        while parent_id is not None:
            parent = self.get(parent_id)
            if parent is None:
                break
            chain.append(parent.text)
            parent_id = parent.parent_id
        return " < ".join(chain)

    def ladder_status(self) -> List[str]:
        return [f"{goal.text} [{goal.status}]" for goal in self.roots()]

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_id": self._next_id,
            "goals": [goal.to_dict() for goal in self.goals],
        }

    def save(self, path: str) -> str:
        file_makedir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str, missing_ok: bool = True) -> Optional["GoalTree"]:
        if not os.path.exists(path):
            if missing_ok:
                return None
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        tree = cls([Goal.from_dict(entry) for entry in data.get("goals", [])])
        tree._next_id = max(tree._next_id, int(data.get("next_id", 0)))
        return tree


def _normalise(text: str) -> str:
    return " ".join(
        "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()
    )
