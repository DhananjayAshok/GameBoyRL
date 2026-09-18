from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

GRID_SKIP = 16
Y_OFFSET = 0
LEFT_COL = -4
TOP_ROW = 4
BOX_COLOR = (255, 0, 0)


def split_grid(frame: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    n_cols = frame.shape[1] // GRID_SKIP
    n_rows = (frame.shape[0] - Y_OFFSET) // GRID_SKIP
    cells = {}
    for col in range(n_cols):
        for row in range(n_rows):
            x0, y0 = col * GRID_SKIP, Y_OFFSET + row * GRID_SKIP
            cells[(LEFT_COL + col, TOP_ROW - row)] = frame[y0:y0 + GRID_SKIP, x0:x0 + GRID_SKIP]
    return cells


def cell_box(x: int, y: int) -> Tuple[int, int, int, int]:
    x0 = (x - LEFT_COL) * GRID_SKIP
    y0 = (TOP_ROW - y) * GRID_SKIP + Y_OFFSET
    return x0, y0, x0 + GRID_SKIP - 1, y0 + GRID_SKIP - 1


def draw_box(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    rgb = frame[:, :, 0] if frame.ndim == 3 and frame.shape[2] == 1 else frame
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=2)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), BOX_COLOR, 1)
    return Image.fromarray(rgb)


def draw_cell_box(frame: np.ndarray, cell: Tuple[int, int]) -> Image.Image:
    return draw_box(frame, *cell_box(*cell))


def recombine_grid(cells: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
    xs = sorted({x for x, _ in cells})
    ys = sorted({y for _, y in cells}, reverse=True)
    return np.concatenate(
        [np.concatenate([cells[(x, y)] for x in xs], axis=1) for y in ys], axis=0
    )
