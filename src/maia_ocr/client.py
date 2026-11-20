"""Use the ``ocr`` service from a pipeline."""

from __future__ import annotations

import numpy as np
from maia_infer import Client

from . import plates
from .types import TextLine, unpack_lines


class OCRClient:
    def __init__(self, client: Client | str = "tcp://localhost:5555", service: str = "ocr") -> None:
        self.client = Client(client) if isinstance(client, str) else client
        self.service = service

    def read(self, image: np.ndarray) -> list[TextLine]:
        res = self.client.infer(self.service, image, meta={"mode": "read"})
        return unpack_lines(res.arrays, res.meta)

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        res = self.client.infer(self.service, crop, meta={"mode": "recognize"})
        lines = unpack_lines(res.arrays, res.meta)
        return (lines[0].text, lines[0].score) if lines else ("", 0.0)

    def read_plate(self, crop: np.ndarray, formats: str | list[str] = "generic") -> plates.PlateReading:
        """Read a plate crop. One- and two-row plates both work: the crop is read in full
        and rows are joined top to bottom before the format correction. ``formats="uae"``
        uses the UAE parser (emirate, code, number; see :mod:`maia_ocr.uae`)."""
        lines = self.read(crop)
        if formats == "uae":
            from . import uae

            return uae.parse(lines)
        text = plates.merge_rows(lines) if lines else self.recognize(crop)[0]
        return plates.correct(text, formats)
