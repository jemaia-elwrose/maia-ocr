"""RapidOCR: PP-OCR models on ONNX Runtime. The PP-OCRv6 small models ship inside
the pip package, so nothing is downloaded; runs on CPU, or CUDA with onnxruntime-gpu."""

from __future__ import annotations

import logging

import numpy as np

from ..types import TextLine


class RapidBackend:
    name = "rapid-ppocrv6"

    def __init__(
        self, device: str = "cpu", min_score: float = 0.5, box_thresh: float = 0.5, use_angle_cls: bool = True
    ) -> None:
        from rapidocr import RapidOCR

        params = {
            "Global.log_level": "warning",
            "Global.text_score": min_score,
            "Global.use_cls": use_angle_cls,
            "Det.box_thresh": box_thresh,
        }
        if device.startswith("cuda"):
            params["EngineConfig.onnxruntime.use_cuda"] = True
            params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = int(device.partition(":")[2] or 0)
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        self.engine = RapidOCR(params=params)
        # RapidOCR creates its recognizer lazily on the first full read(); load it now so
        # recognize() works on a fresh engine and the first request isn't slow.
        if getattr(self.engine, "text_rec", None) is None and hasattr(self.engine, "_load_rec_model"):
            self.engine._load_rec_model()

    def read(self, image: np.ndarray) -> list[TextLine]:
        out = self.engine(image)
        if out.boxes is None or out.txts is None:
            return []
        return [
            TextLine(text=str(t), score=float(s), poly=np.asarray(b, np.float32).reshape(4, 2))
            for b, t, s in zip(out.boxes, out.txts, out.scores, strict=True)
        ]

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        from rapidocr.ch_ppocr_rec import TextRecInput

        recognizer = self.engine.text_rec or self.engine._load_rec_model()
        out = recognizer(TextRecInput(img=list(crops)))
        return [(str(t), float(s)) for t, s in zip(out.txts, out.scores, strict=True)]
