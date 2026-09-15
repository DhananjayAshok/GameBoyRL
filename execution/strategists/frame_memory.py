"""
Every screen the agent has ever stood on, and whether it is standing on one again.

The agent has no sense of place. Measured on a 864-step Pokemon Red run: 86% of its
time was spent on a screen it had already seen, and the notebook that was supposed to be
its world model held seven unanchored nouns -- "fence exists on east side", "signpost
exists" -- with no positions and no connections. Nothing in the loop could tell the
planner "you have been here before", so its tasks collapsed to compass directions and it
circled.

This is the smallest thing that fixes that: a set of every frame seen, and a flag when
the current one is a repeat. It does not build a map -- there is no topology here, no
notion of which screen adjoins which -- it only answers "is this new?". That single bit,
fed into the planning prompt and the executor's hint, is what lets a plan say "go
somewhere else" with any basis.

Exact-match only, deliberately. A Game Boy screen is 144x160 and deterministic given the
same position and game state, so an exact repeat is a real repeat and there are no
thresholds to tune or calibrate per game -- which the frame-difference work established
is otherwise a per-game calibration problem. Perceptual matching (patch projections,
embedding similarity) is the planned follow-up and would subsume this; it is not needed
to answer the question this asks.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

#: Frames whose repeat count is worth reporting to the planner. One prior visit is
#: ordinary -- walking back through a doorway revisits a screen -- so the message is held
#: until a screen has been stood on enough times to look like circling rather than
#: passing through.
REVISIT_THRESHOLD = 3

#: What the planner and the executor are told when the current screen is a repeat. Phrased
#: as a fact plus a direction, not a prohibition: "do not go here" leaves the model with no
#: alternative, and the failure being addressed is a model that already cannot name one.
SEEN_BEFORE_NOTE = (
    "[EXPLORATION] You have already stood on this exact screen {count} times, and it has "
    "not led anywhere new. Whatever you tried from here before did not open up the map. "
    "Do not repeat it: pick a direction or an exit you have NOT taken from this screen, "
    "and prefer anything that leads off the edge of what you can currently see."
)

#: Appended to the executor's hint on a repeated screen. Shorter than the planner's note
#: -- the executor acts on one screen at a time and gets the task text as well.
SEEN_BEFORE_HINT = (
    "This screen has been visited {count} times already. Head somewhere you have not been."
)


def frame_key(frame: np.ndarray) -> Optional[str]:
    """A stable identity for one screen, or ``None`` if it cannot be computed.

    Hashed rather than stored: a run produces thousands of 144x160 frames, which is tens
    of megabytes held for the length of an episode, against 16 bytes each this way.
    """
    # A missing frame is not a screen. Without this guard np.ascontiguousarray(None)
    # yields a 0-d object array that hashes happily, so a dropped frame would be counted
    # as a place the agent had stood.
    if frame is None or not hasattr(frame, "tobytes"):
        return None
    try:
        return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(),
                               digest_size=16).hexdigest()
    except Exception:  # noqa: BLE001 - a memory that cannot hash must not end a run
        return None


class FrameMemory:
    """Every frame seen this episode, counted by identity.

    Append-only within an episode and never compressed, unlike the notebook's attempt log:
    the whole value is knowing a screen was visited, and a summariser that dropped one
    would silently turn a revisit into a discovery.
    """

    def __init__(self) -> None:
        self._counts: dict = {}
        self._total = 0

    def add(self, frame: np.ndarray) -> int:
        """Record one frame. Returns how many times it has now been seen."""
        key = frame_key(frame)
        if key is None:
            return 0
        self._total += 1
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def add_steps(self, steps: Iterable) -> int:
        """Record the screen after every environment step in *steps*.

        Takes the frame AFTER each action, which is the screen the agent actually ends up
        looking at; the before-frame is the previous step's after-frame and would double
        every count.
        """
        added = 0
        for record in steps or ():
            frame = getattr(record, "frame_after", None)
            if frame is not None:
                self.add(frame)
                added += 1
        return added

    def times_seen(self, frame: np.ndarray) -> int:
        """How many times this exact screen has been recorded."""
        key = frame_key(frame)
        return 0 if key is None else self._counts.get(key, 0)

    def is_revisit(self, frame: np.ndarray) -> bool:
        """Whether this screen has been seen often enough to look like circling."""
        return self.times_seen(frame) >= REVISIT_THRESHOLD

    def note_for(self, frame: np.ndarray) -> str:
        """The planner's exploration note for this screen, or ``""`` if it is new."""
        count = self.times_seen(frame)
        return SEEN_BEFORE_NOTE.format(count=count) if count >= REVISIT_THRESHOLD else ""

    def hint_for(self, frame: np.ndarray) -> str:
        """The executor's hint addition for this screen, or ``""`` if it is new."""
        count = self.times_seen(frame)
        return SEEN_BEFORE_HINT.format(count=count) if count >= REVISIT_THRESHOLD else ""

    @property
    def n_frames(self) -> int:
        """Total frames recorded, repeats included."""
        return self._total

    @property
    def n_unique(self) -> int:
        """Distinct screens seen."""
        return len(self._counts)

    @property
    def revisit_rate(self) -> float:
        """Share of recorded frames that were not new. 0.0 when nothing is recorded."""
        return 0.0 if not self._total else 1.0 - (len(self._counts) / self._total)

    def to_dict(self) -> dict:
        """Summary for the run record. The counts themselves are not serialised --
        they are hashes, useless to a reader, and there are thousands of them."""
        return {"n_frames": self._total,
                "n_unique": self.n_unique,
                "revisit_rate": round(self.revisit_rate, 4)}
