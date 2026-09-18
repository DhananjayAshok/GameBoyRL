"""
Exploration driven by how novel a screen looks, rather than whether it is byte-identical.

:class:`~execution.strategists.frame_memory.FrameMemory` asks "exactly this screen
before?". :class:`CuriosityMemory` asks "anywhere that LOOKS like this before?", using the
random patch projection in :mod:`~execution.strategists.patch_embedding`. It answers the
same two questions to the same two callers -- a note for the planner and a line for the
executor's hint -- so it is a drop-in alternative and the loop does not change shape.

The reason to want the looser test: a Game Boy screen carries an animating sprite, a
blinking cursor and a scrolling text box, so the same doorway visited twice is often two
different byte strings. Exact matching then reports a discovery and the agent is
encouraged to explore a place it is already standing in. Measured on one Prism run, the
environment's own change-detector never fired once across 2,170 checks for the same
reason -- on that game nothing is ever byte-identical.

Two signals are kept rather than one, because they are fooled by different things:

* ``novelty`` -- distance from the nearest screen already seen. Blind to WHERE the
  matching content sat, so two different rooms with the same furniture look alike.
* ``patch overlap`` -- how many 16x16 tiles line up position-for-position with the nearest
  known screen. Catches "same room, text box open" and "same room, sprite mid-animation".
  It compares fixed positions, so a view shifted along a corridor scores low here even
  though it is the same place -- novelty is what covers that. It is also fooled by large
  uniform areas, where every tile matches everywhere.

A screen counts as somewhere the agent has been if EITHER fires. False "seen" costs one
redundant nudge to explore; false "new" leaves it circling unwarned, which is the failure
this exists to catch, so the asymmetry is deliberate.

Cost is bounded by construction: the nearest-neighbour search is one matmul against a
matrix of stored frame vectors, capped at :data:`MAX_KEYFRAMES` rows, and a screen is only
stored when it is novel -- so a run that circles stores almost nothing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from execution.strategists.patch_embedding import (NOVELTY_THRESHOLD,
                                                   OVERLAP_THRESHOLD,
                                                   embed_frame, frame_vector,
                                                   patch_overlap)

#: Distinct screens retained. A Pokemon game's reachable early map is a few hundred
#: screens, and the whole point is to notice returning to one, so the cap is generous --
#: at 32 floats per keyframe vector the memory cost is trivial, and the matmul over 4,000
#: rows is microseconds.
MAX_KEYFRAMES = 4000

#: Planner-facing note. States what was measured, then asks for a specific alternative:
#: the model this addresses is one that already cannot name somewhere else to go, so a
#: bare prohibition leaves it stuck.
SEEN_BEFORE_NOTE = (
    "[EXPLORATION] This screen is {pct:.0f}% the same as somewhere you have already been"
    "{detail}. Going over old ground has not opened up the map. Do not repeat what you "
    "tried here: pick an exit, a door or a direction you have NOT taken from this screen, "
    "and prefer whatever leads off the edge of what you can currently see."
)

#: Executor-facing line, appended to the planner's hint -- the only channel the executor
#: has, since it cannot see the notebook or this memory.
SEEN_BEFORE_HINT = (
    "This screen closely matches one visited before. Head somewhere you have not been."
)


class CuriosityMemory:
    """Screens seen this episode, compared by appearance.

    Interface-compatible with :class:`~execution.strategists.frame_memory.FrameMemory`
    where the strategist touches it: ``add``, ``add_steps``, ``note_for``, ``hint_for``,
    ``to_dict``, and the three summary properties.
    """

    def __init__(self, parser=None) -> None:
        self._parser = parser
        self._vectors: List[np.ndarray] = []      # one pooled vector per kept screen
        self._patches: List[np.ndarray] = []      # per-patch embeddings, same order
        self._matrix: Optional[np.ndarray] = None  # stacked vectors, rebuilt on append
        self._total = 0
        self._revisits = 0

    # ------------------------------------------------------------------ measuring

    def _nearest(self, patches: np.ndarray) -> Tuple[float, float]:
        """``(novelty, patch_overlap)`` of this screen against everything stored."""
        if self._matrix is None or not len(self._vectors):
            return 1.0, 0.0
        sims = self._matrix @ frame_vector(patches)
        best = int(np.argmax(sims))
        novelty = float(1.0 - np.clip(sims[best], -1.0, 1.0))
        return novelty, patch_overlap(patches, self._patches[best])

    def measure(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """``(novelty, overlap)`` for *frame*, or ``None`` if it cannot be embedded."""
        patches = embed_frame(frame, parser=self._parser)
        if patches is None:
            return None
        return self._nearest(patches)

    def is_revisit(self, frame: np.ndarray) -> bool:
        """Whether this screen looks like somewhere already visited."""
        measured = self.measure(frame)
        if measured is None:
            return False
        novelty, overlap = measured
        return novelty <= NOVELTY_THRESHOLD or overlap >= OVERLAP_THRESHOLD

    # ------------------------------------------------------------------ recording

    def add(self, frame: np.ndarray) -> float:
        """Record *frame*; return its novelty. Only novel screens are kept.

        Storing every frame would make the nearest-neighbour search grow without bound and
        fill the memory with thousands of copies of the screen the agent is stuck on --
        the exact case this is meant to detect.
        """
        patches = embed_frame(frame, parser=self._parser)
        if patches is None:
            return 0.0
        self._total += 1
        novelty, overlap = self._nearest(patches)
        if novelty <= NOVELTY_THRESHOLD or overlap >= OVERLAP_THRESHOLD:
            self._revisits += 1
            return novelty
        if len(self._vectors) < MAX_KEYFRAMES:
            self._vectors.append(frame_vector(patches))
            self._patches.append(patches)
            self._matrix = np.stack(self._vectors)
        return novelty

    def add_steps(self, steps) -> int:
        """Record the screen after every environment step in *steps*."""
        added = 0
        for record in steps or ():
            frame = getattr(record, "frame_after", None)
            if frame is not None:
                self.add(frame)
                added += 1
        return added

    # ------------------------------------------------------------------ reporting

    def note_for(self, frame: np.ndarray) -> str:
        """The planner's exploration note, or ``""`` when the screen looks new."""
        measured = self.measure(frame)
        if measured is None:
            return ""
        novelty, overlap = measured
        if novelty > NOVELTY_THRESHOLD and overlap < OVERLAP_THRESHOLD:
            return ""
        detail = (f", and {overlap:.0%} of its tiles line up exactly"
                  if overlap >= OVERLAP_THRESHOLD else "")
        return SEEN_BEFORE_NOTE.format(pct=100.0 * (1.0 - novelty), detail=detail)

    def hint_for(self, frame: np.ndarray) -> str:
        """The executor's hint addition, or ``""`` when the screen looks new."""
        return SEEN_BEFORE_HINT if self.is_revisit(frame) else ""

    @property
    def n_frames(self) -> int:
        """Frames measured, repeats included."""
        return self._total

    @property
    def n_unique(self) -> int:
        """Distinct-looking screens retained."""
        return len(self._vectors)

    @property
    def revisit_rate(self) -> float:
        """Share of measured frames that looked like somewhere already seen."""
        return 0.0 if not self._total else self._revisits / self._total

    def to_dict(self) -> dict:
        return {"mode": "curiosity",
                "n_frames": self._total,
                "n_unique": self.n_unique,
                "revisit_rate": round(self.revisit_rate, 4)}
