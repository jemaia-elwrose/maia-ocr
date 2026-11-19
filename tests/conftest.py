import cv2
import numpy as np
import pytest


def render(*rows, scale=1.4, pad=18):
    """White image with each row of text in black; rows stacked top to bottom."""
    sizes = [cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0] for r in rows]
    w = max(s[0] for s in sizes) + 2 * pad
    line_h = max(s[1] for s in sizes) + pad
    img = np.full((line_h * len(rows) + pad, w, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        cv2.putText(img, r, (pad, pad + (i + 1) * line_h - pad // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    return img


@pytest.fixture(scope="session")
def rapid():
    from maia_ocr.backends import build

    return build("rapid")
