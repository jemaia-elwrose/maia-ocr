"""maia-ocr read | plate | container

    maia-ocr read sign.jpg --save annotated.jpg
    maia-ocr plate plate_crop.jpg --format uae
    maia-ocr container gate_camera.jpg

Runs the model in-process by default; pass ``--endpoint`` to use a running ``ocr`` service.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from . import containers, plates
from .text import full_text
from .types import TextLine


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return image


def _draw(image: np.ndarray, lines: list[TextLine]) -> np.ndarray:
    out = image.copy()
    for ln in lines:
        cv2.polylines(out, [ln.poly.astype(np.int32)], True, (80, 200, 120), 2)
        x, y = (int(v) for v in ln.poly[0])
        cv2.putText(
            out,
            f"{ln.text} {ln.score:.2f}",
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 200, 120),
            1,
            cv2.LINE_AA,
        )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="maia-ocr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", default=os.getenv("MAIA_OCR_BACKEND", "rapid"), help="rapid | paddle")
    ap.add_argument("--device", help="cpu | cuda:0 (default: backend's choice)")
    ap.add_argument("--endpoint", help="use a running 'ocr' service, e.g. tcp://localhost:5555")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="all text in an image, in reading order")
    r.add_argument("image")
    r.add_argument("--save", help="write an annotated copy")
    p = sub.add_parser("plate", help="read a number-plate crop and fit it to a format")
    p.add_argument("image")
    p.add_argument(
        "--format", default="generic", help=f"one of {sorted(plates.FORMATS)} or a template like 'LL DD LLL'"
    )
    c = sub.add_parser("container", help="find ISO 6346 container codes (check digit verified)")
    c.add_argument("image")
    args = ap.parse_args(argv)

    if args.endpoint:
        from .client import OCRClient

        svc = OCRClient(args.endpoint)
        read = svc.read
    else:
        from .backends import build

        backend = build(args.backend, **({"device": args.device} if args.device else {}))
        read = backend.read

    image = _read_image(args.image)
    lines = read(image)
    if args.cmd == "read":
        print(
            json.dumps(
                {
                    "text": full_text(lines),
                    "lines": [
                        {"text": ln.text, "score": round(ln.score, 4), "box": [round(float(v), 1) for v in ln.box]}
                        for ln in lines
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if args.save:
            cv2.imwrite(args.save, _draw(image, lines))
    elif args.cmd == "plate":
        fmt = args.format if args.format in plates.FORMATS else [args.format]
        reading = plates.correct(plates.merge_rows(lines), fmt)
        print(json.dumps(reading.__dict__, ensure_ascii=False))
    else:
        print(json.dumps({"containers": containers.find(lines), "text": full_text(lines)}, indent=2))


if __name__ == "__main__":
    main()
