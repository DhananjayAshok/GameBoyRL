"""
Screens compared by what they look like, not by being byte-identical.

:mod:`~execution.strategists.frame_memory` answers "have I stood on this exact screen
before". That is exact and cheap and needs no calibration, but it is brittle in the one
place it matters: a Game Boy screen carries an animating sprite, a blinking cursor and a
scrolling text box, so standing in the same doorway twice can produce two different byte
strings. Exact matching then calls a revisit a discovery, and the agent is told to explore
a place it never left.

This module compares screens perceptually instead. It is deliberately not a learned
encoder -- nothing here trains, and there is no model to download or serve:

1. The frame is cut into 16x16 patches by the emulator's OWN parser
   (``StateParser.capture_grid_cells``), so the grid lands on the game's tile boundaries
   -- it carries a ``y_offset`` of -2 for exactly that reason, which naive slicing gets
   wrong.
2. Every patch is projected through ONE fixed random matrix (Johnson-Lindenstrauss: a
   random projection preserves relative distances in expectation, so cosine similarity in
   the projected space tracks similarity in pixel space). Seeded, so two runs of the same
   build embed a screen identically.
3. The patches are projected as a single batched matmul, not one at a time. At 90 patches
   per frame and thousands of frames per episode the loop version is the whole cost.

Two questions can then be asked of a screen, and they are different:

``novelty``       1 - the highest cosine similarity to any screen already seen. Near 0
                  means "I have been somewhere that looks like this"; near 1 means new.
``patch_overlap`` the fraction of the 16x16 patches that match the nearest known screen
                  position-for-position. This is the "partly matched" case: a text box
                  opened over a familiar room shares every tile except the bottom rows,
                  and a sprite animating in place changes one tile out of ninety. It is
                  comparison at fixed positions, so it does NOT recognise a view shifted
                  along a corridor -- that moves content between positions and is what
                  the pooled novelty score is for.

Both are reported rather than one collapsed score, because they fail differently: novelty
is fooled by two visually similar but distinct rooms, and patch overlap is fooled by large
uniform areas (grass, cave floor) where most patches match everywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

#: Side of one patch in pixels. 16 is the Game Boy's tile size and the emulator parser's
#: own default, so a patch is a tile and the grid aligns with what the game draws.
PATCH = 16

#: Width of the projected patch vector. 32 of 256 dimensions keeps enough structure for
#: cosine similarity to be meaningful while making the per-frame embedding small enough to
#: hold thousands of them.
PROJECTION_DIM = 32

#: Seed for the projection matrix. Fixed so that a screen embeds identically across runs
#: and across processes -- a novelty score that moved between runs would be unusable for
#: comparing arms.
PROJECTION_SEED = 20260917

#: Cosine similarity at or above which two projected patches are "the same patch".
PATCH_MATCH_THRESHOLD = 0.98

#: Novelty at or below which a screen is treated as somewhere the agent has effectively
#: been. Deliberately loose: the cost of a false "seen" is one redundant exploration
#: nudge, while the cost of a false "new" is the agent circling unwarned, which is the
#: failure this exists to catch.
NOVELTY_THRESHOLD = 0.08

#: Patch overlap at or above which a screen counts as a partial match even when its
#: novelty clears the bar -- the same place with a text box over it, or shifted by a tile.
OVERLAP_THRESHOLD = 0.85


def _projection(patch_pixels: int) -> np.ndarray:
    """The fixed random projection, built once per patch size and cached on the function.

    Gaussian entries scaled by 1/sqrt(dim): the Johnson-Lindenstrauss construction, which
    preserves pairwise distances in expectation without any training.
    """
    cache = _projection.__dict__.setdefault("_cache", {})
    if patch_pixels not in cache:
        rng = np.random.default_rng(PROJECTION_SEED)
        cache[patch_pixels] = (rng.standard_normal((patch_pixels, PROJECTION_DIM))
                               / np.sqrt(PROJECTION_DIM)).astype(np.float32)
    return cache[patch_pixels]


def split_patches(frame: np.ndarray, parser=None) -> Optional[np.ndarray]:
    """Cut *frame* into 16x16 patches, as ``(n_patches, PATCH*PATCH)``.

    Uses the emulator's ``capture_grid_cells`` when a parser is available, so the grid
    sits on the game's tile boundaries (it applies a y-offset for that). Falls back to
    plain slicing otherwise, which keeps this module testable and usable without an
    emulator; the fallback is off by the same couple of pixels every time, so it stays
    self-consistent even though it is not tile-aligned.
    """
    if frame is None or not hasattr(frame, "shape"):
        return None
    grey = np.asarray(frame)
    if grey.ndim == 3:
        grey = grey[:, :, 0]

    if parser is not None:
        try:
            cells = parser.capture_grid_cells(np.asarray(frame), grid_skip=PATCH)
            keep = [np.asarray(c) for _, c in sorted(cells.items())
                    if np.asarray(c).size == PATCH * PATCH]
            if keep:
                return np.stack([c.reshape(-1) for c in keep]).astype(np.float32)
        except Exception:  # noqa: BLE001 - fall through to slicing; never end a run
            pass

    rows, cols = grey.shape[0] // PATCH, grey.shape[1] // PATCH
    if rows == 0 or cols == 0:
        return None
    trimmed = grey[: rows * PATCH, : cols * PATCH]
    patches = (trimmed.reshape(rows, PATCH, cols, PATCH)
                      .transpose(0, 2, 1, 3)
                      .reshape(rows * cols, PATCH * PATCH))
    return patches.astype(np.float32)


def embed_frame(frame: np.ndarray, parser=None) -> Optional[np.ndarray]:
    """Per-patch embeddings for one frame, L2-normalised, as ``(n_patches, PROJECTION_DIM)``.

    One matmul for the whole frame -- the "take batches" part. Patches are centred before
    projection so that a uniformly brighter screen is not treated as a different place,
    and normalised after so that cosine similarity is a dot product.
    """
    patches = split_patches(frame, parser=parser)
    if patches is None or patches.size == 0:
        return None
    patches = patches - patches.mean(axis=1, keepdims=True)
    embedded = patches @ _projection(patches.shape[1])
    norms = np.linalg.norm(embedded, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embedded / norms


def frame_vector(patch_embeddings: np.ndarray) -> np.ndarray:
    """One vector for a whole screen: its patches laid end to end, renormalised.

    NOT the mean of the patches. Mean-pooling was tried first and is useless on real
    frames: a Game Boy screen is mostly background tiles, so two entirely different rooms
    average to nearly the same vector. Measured against the live emulator, a screen
    transition with a mean absolute pixel difference of 72.8 -- about as different as two
    frames of this game get -- scored a pooled novelty of 0.051, inside the threshold for
    "somewhere I have already been". It passed the unit tests only because those used
    random-noise frames, whose means genuinely do differ.

    Concatenating keeps each patch attached to its position, so the dot product of two
    screens is the average per-position patch similarity -- a continuous version of
    :func:`patch_overlap`, and discriminative on real screens.
    """
    flat = patch_embeddings.reshape(-1)
    norm = np.linalg.norm(flat)
    return flat if norm == 0 else flat / norm


def patch_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of patch positions where two frames agree, 0.0 if they are not comparable.

    Position-for-position, which is what makes this "partly matched" rather than "similar
    on average": a room with a dialogue box over it matches everywhere except the bottom
    rows, and that is visible here while the pooled vector blurs it away. The same
    property means a translated view scores LOW here even though it is the same place --
    novelty covers that case, and the two are reported separately for this reason.
    """
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    sims = np.einsum("ij,ij->i", a, b)
    return float((sims >= PATCH_MATCH_THRESHOLD).mean())
