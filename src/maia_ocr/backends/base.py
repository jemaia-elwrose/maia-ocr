from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import TextLine


class OCRBackend(Protocol):
    name: str

    def read(self, image: np.ndarray) -> list[TextLine]:
        """Find and read every text line in a BGR image."""

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Read crops that each contain a single line of text (e.g. plates), in one batch."""


def full_poly(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
