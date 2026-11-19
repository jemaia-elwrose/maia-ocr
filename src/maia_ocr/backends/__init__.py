"""OCR backends. Both run PaddlePaddle's PP-OCR models (Apache-2.0).

====== ================================= =============================== =====================
name   runtime                           models                          install
====== ================================= =============================== =====================
rapid  ONNX Runtime (CPU, optional CUDA) PP-OCRv6 small, bundled in pip  included
paddle PaddlePaddle (NVIDIA GPU)         PP-OCRv6 medium, downloaded     ``maia-ocr[paddle]``
====== ================================= =============================== =====================
"""

from __future__ import annotations

from .base import OCRBackend


def build(name: str = "rapid", **options) -> OCRBackend:
    name = name.lower()
    if name == "rapid":
        from .rapid import RapidBackend

        return RapidBackend(**options)
    if name == "paddle":
        from .paddle import PaddleBackend

        return PaddleBackend(**options)
    raise ValueError(f"unknown OCR backend '{name}' (rapid | paddle)")
