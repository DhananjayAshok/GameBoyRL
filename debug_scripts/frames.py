"""
Frame-strip rendering shared by the curiosity / attempt / practice reports.

A "strip" is N frames sampled evenly from a trajectory, concatenated left-to-right with
a separator line between panels, each labelled with its index (top-left) and, when
available, the action taken from it (bottom-left). This is the same presentation as
show_trajectories.plot_transitions, generalised to accept RGB or greyscale frames and to
subsample long trajectories.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import parse_action_line


try:
    _FONT = ImageFont.load_default(size=14)
except TypeError:  # older Pillow
    _FONT = ImageFont.load_default()

SEPARATOR_WIDTH = 2


def to_pil(frame: np.ndarray) -> Image.Image:
    """Convert any frame the pipeline produces (2-D, HxWx1, or HxWx3) to an RGB image."""
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array).convert("RGB")


def sample_indices(total: int, n: int) -> list[int]:
    """Evenly spaced indices covering [0, total), always including first and last."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def action_label(high_level_action) -> str:
    """
    Best-effort short label for one entry of a trajectory's ``high_level_actions``.

    Entries are ``[action_class, kwargs]`` pairs — stored as **numpy object arrays**, not
    tuples, once the trajectory has been through the replay buffer — and the low-level
    button lives in ``kwargs["low_level_action"]``. Anything unexpected degrades to a
    truncated repr rather than raising: labels are decoration, not data.
    """
    try:
        pair = high_level_action
        if isinstance(pair, np.ndarray):
            pair = pair.tolist()
        if isinstance(pair, (tuple, list)) and len(pair) > 1:
            kwargs = pair[1]
            if isinstance(kwargs, dict) and "low_level_action" in kwargs:
                raw = str(kwargs["low_level_action"])
                return (
                    raw.replace("LowLevelActions.PRESS_BUTTON_", "")
                    .replace("LowLevelActions.PRESS_ARROW_", "")
                )
        return str(pair)[:12]
    except Exception:
        return ""


def strip(frames, out_path: str, labels=None, n: int = 5, overwrite: bool = False) -> str | None:
    """
    Render up to *n* frames side by side with separators and labels.

    :param frames: Sequence of frames (numpy arrays).
    :param out_path: Destination PNG path.
    :param labels: Optional per-frame bottom labels, indexed like *frames*.
    :param n: Number of frames to sample.
    :param overwrite: Re-render even if *out_path* already exists.
    :return: *out_path*, or None if there were no frames to draw.
    """
    if not overwrite and os.path.exists(out_path):
        return out_path
    frames = list(frames)
    indices = sample_indices(len(frames), n)
    if not indices:
        return None

    images = [to_pil(frames[i]) for i in indices]
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + SEPARATOR_WIDTH * (len(images) - 1)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for panel_i, (frame_i, image) in enumerate(zip(indices, images)):
        canvas.paste(image, (x, 0))
        draw.text((x + 3, 2), str(frame_i), fill=(255, 0, 0), font=_FONT)
        if labels is not None and frame_i < len(labels):
            text = str(labels[frame_i])
            if text:
                draw.text((x + 3, height - 16), text, fill=(255, 0, 0), font=_FONT)
        x += image.width
        if panel_i < len(images) - 1:
            draw.rectangle([x, 0, x + SEPARATOR_WIDTH - 1, height], fill=(0, 128, 255))
            x += SEPARATOR_WIDTH

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, "PNG")
    return out_path


def trajectory_strip(trajectory, out_path: str, n: int = 5, overwrite: bool = False) -> str | None:
    """
    Strip for one RL/attempt trajectory 5-tuple.

    ``(observations, actions, high_level_actions, rewards, init_state)`` — the shape used
    by both grouped_trajectories and success_trajectories.pkl.
    """
    observations = trajectory[0]
    high_level_actions = trajectory[2] if len(trajectory) > 2 else None
    labels = None
    if high_level_actions is not None:
        labels = [action_label(a) for a in high_level_actions]
    return strip(observations, out_path, labels=labels, n=n, overwrite=overwrite)


def call_log_strip(vlm_call_log, out_path: str, n: int = 5, overwrite: bool = False) -> str | None:
    """
    Strip for one practice episode's ``List[VLMCallRecord]``.

    Uses the first image of each action-tagged call, labelled with the parsed action from
    that call's response, so the strip reads as "what it saw -> what it chose".
    """
    frames, labels = [], []
    for record in vlm_call_log:
        images = getattr(record, "images", None)
        if not images:
            continue
        frames.append(images[0])
        # Truncated to fit the frame label; the width is a rendering concern, so it lives
        # here rather than in the parser.
        labels.append((parse_action_line(getattr(record, "response", "")) or "")[:16])
    return strip(frames, out_path, labels=labels, n=n, overwrite=overwrite)


