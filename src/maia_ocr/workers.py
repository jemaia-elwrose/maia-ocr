"""maia-infer worker, available by name after ``pip install maia-ocr``:

    maia-infer worker ocr                          # RapidOCR, CPU
    maia-infer worker ocr --opt device=cuda:0      # RapidOCR with onnxruntime-gpu
    maia-infer worker ocr --opt backend=paddle     # PaddleOCR on GPU

Request meta ``{"mode": "read"}`` (default) finds and reads all text in the image;
``{"mode": "recognize"}`` treats the image as one text line, e.g. a plate crop.
Crops in recognize mode are read together in one batch.
Reply: arrays = [polys (N, 4, 2), scores (N,)], meta = {"texts": [...], "backend": ...}.
"""

from __future__ import annotations

import logging
import time

from maia_infer import Result, Worker

from .backends.base import full_poly
from .types import TextLine, pack_lines

log = logging.getLogger("maia_ocr.workers")


class OCRWorker(Worker):
    service = "ocr"
    max_batch = 16
    max_wait_ms = 4

    def setup(self) -> None:
        from .backends import build

        t0 = time.perf_counter()
        name = self.options.get("backend", "rapid")
        opts = {"min_score": float(self.options.get("min_score", 0.5))}
        if "device" in self.options:
            opts["device"] = self.options["device"]
        self.backend = build(name, **opts)
        log.info("OCRWorker ready: %s (%.1fs)", self.backend.name, time.perf_counter() - t0)

    def predict_batch(self, requests):
        results: list[list[TextLine] | None] = [None] * len(requests)
        crops, crop_owner = [], []
        for i, r in enumerate(requests):
            image = r.arrays[0]
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 4:
                raise ValueError(f"expected an HxWx3 BGR image, got {image.shape}")
            if r.meta.get("mode", "read") == "recognize":
                crops.append(image)
                crop_owner.append(i)
            else:
                results[i] = self.backend.read(image)
        for i, (text, score) in zip(crop_owner, self.backend.recognize(crops), strict=True):
            results[i] = [TextLine(text, score, full_poly(requests[i].arrays[0]))] if text else []
        out = []
        for lines in results:
            arrays, meta = pack_lines(lines)
            meta["backend"] = self.backend.name
            out.append(Result(arrays=arrays, meta=meta))
        return out
