"""Text results and the wire format shared by workers and clients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextLine:
    text: str
    score: float
    poly: np.ndarray  # (4, 2) corners, clockwise from top-left, in pixels

    @property
    def box(self) -> np.ndarray:
        """Axis-aligned x1, y1, x2, y2."""
        return np.array([*self.poly.min(axis=0), *self.poly.max(axis=0)], np.float32)

    @property
    def center(self) -> tuple[float, float]:
        c = self.poly.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def height(self) -> float:
        return float(self.poly[:, 1].max() - self.poly[:, 1].min())


def pack_lines(lines: list[TextLine]) -> tuple[list[np.ndarray], dict]:
    polys = np.array([ln.poly for ln in lines], np.float32).reshape(-1, 4, 2)
    scores = np.array([ln.score for ln in lines], np.float32)
    return [polys, scores], {"texts": [ln.text for ln in lines]}


def unpack_lines(arrays: list[np.ndarray], meta: dict) -> list[TextLine]:
    polys, scores = arrays[0].reshape(-1, 4, 2), arrays[1].reshape(-1)
    return [
        TextLine(text=t, score=float(s), poly=p.copy())
        for t, s, p in zip(meta.get("texts", []), scores, polys, strict=True)
    ]
