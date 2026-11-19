"""PaddleOCR 3.x on the PaddlePaddle framework (NVIDIA GPU). Install the framework first,
following paddlepaddle.org.cn (e.g. ``paddlepaddle-gpu`` for your CUDA), then
``maia-ocr[paddle]``. Models (PP-OCRv6 medium by default) download on first use."""

from __future__ import annotations

import numpy as np

from ..types import TextLine


class PaddleBackend:
    def __init__(
        self,
        device: str = "gpu:0",
        det_model: str | None = None,
        rec_model: str | None = None,
        min_score: float = 0.5,
        use_textline_orientation: bool = True,
    ) -> None:
        from paddleocr import PaddleOCR, TextRecognition

        device = "gpu:" + (device.partition(":")[2] or "0") if device.startswith(("cuda", "gpu")) else device
        self.min_score = min_score
        self.pipeline = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,  # document helpers: not useful on camera frames
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            text_rec_score_thresh=min_score,
            device=device,
        )
        self.recognizer = (
            TextRecognition(model_name=rec_model, device=device) if rec_model else TextRecognition(device=device)
        )
        self.name = f"paddle-{rec_model or 'default'}"

    def read(self, image: np.ndarray) -> list[TextLine]:
        lines = []
        for res in self.pipeline.predict(image):
            for poly, text, score in zip(res["rec_polys"], res["rec_texts"], res["rec_scores"], strict=True):
                lines.append(TextLine(str(text), float(score), np.asarray(poly, np.float32).reshape(4, 2)))
        return lines

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        results = self.recognizer.predict(input=list(crops), batch_size=len(crops))
        return [(str(r["rec_text"]), float(r["rec_score"])) for r in results]
