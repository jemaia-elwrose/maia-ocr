"""Test the ocr service through the broker: synthetic checks first, then your own images.

    maia-infer broker &
    maia-infer worker ocr &                       # or: --opt backend=paddle
    python examples/broker_test.py                # synthetic checks only
    python examples/broker_test.py plate.jpg container.jpg --plate-format uae --save out/

Synthetic checks render text with OpenCV, so they need no downloads:
  1. read      - a sign with two lines comes back in reading order
  2. container - a container door with a valid ISO 6346 code is found (and a fake one is not)
  3. plate     - a UAE-style plate parses into emirate, code and number
  4. batching  - 32 recognize calls from 8 threads, versus one at a time
                 (about 1x on CPU; the gain shows on GPU or with several workers)

For each image you pass, it prints the text, the plate reading and any container
codes, and with --save writes an annotated copy.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from maia_infer import Client

from maia_ocr import containers, full_text, plates, uae
from maia_ocr.client import OCRClient

FONT = cv2.FONT_HERSHEY_DUPLEX


def canvas(w, h):
    return np.full((h, w, 3), 245, np.uint8)


def put(img, text, x, y, scale, thick=2):
    cv2.putText(img, text, (x, y), FONT, scale, (20, 20, 20), thick, cv2.LINE_AA)


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'} {name:<10} {detail}")
    return ok


def synthetic(ocr: OCRClient, client_factory) -> bool:
    ok = True

    img = canvas(640, 220)
    put(img, "LOADING BAY 4", 30, 90, 1.6, 3)
    put(img, "AUTHORISED STAFF ONLY", 30, 170, 1.1)
    text = full_text(ocr.read(img))
    ok &= check("read", "LOADING" in text and "STAFF" in text and text.index("LOADING") < text.index("STAFF"), text)

    code10 = "MSCU123456"
    real = f"{code10}{containers.check_digit(code10)}"
    fake = f"{code10}{(containers.check_digit(code10) + 1) % 10}"
    img = canvas(720, 260)
    put(img, f"{real[:4]} {real[4:10]} {real[10]}", 30, 80, 1.6, 3)
    put(img, "45G1  MAX.GR. 32.500 KG", 30, 150, 0.9)
    put(img, f"TEST {fake}", 30, 220, 0.9)
    found = containers.find(ocr.read(img))
    ok &= check("container", found == [real], f"found {found}, expected [{real}] (and not {fake})")

    img = canvas(600, 160)
    cv2.rectangle(img, (5, 5), (595, 155), (20, 20, 20), 3)
    put(img, "DUBAI", 40, 70, 0.9)
    put(img, "K", 60, 135, 1.8, 4)
    put(img, "35071", 220, 125, 2.8, 6)
    r = ocr.read_plate(img, "uae")
    want = ("Dubai", "K", "35071")
    got = tuple(r.fields.get(k) for k in ("emirate", "code", "number"))
    ok &= check("plate", got == want, f"{got} key={r.fields.get('key')} raw={r.raw!r}")

    crop = canvas(260, 60)
    put(crop, "AB 1234", 10, 45, 1.3, 3)
    n, threads = 32, 8
    t0 = time.perf_counter()
    for _ in range(n):
        ocr.recognize(crop)
    serial = time.perf_counter() - t0

    def work(k):
        c = OCRClient(client_factory())
        for _ in range(k):
            c.recognize(crop)

    pool = [threading.Thread(target=work, args=(n // threads,)) for _ in range(threads)]
    t0 = time.perf_counter()
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    parallel = time.perf_counter() - t0
    text, score = ocr.recognize(crop)
    ok &= check(
        "batching",
        text.replace(" ", "") == "AB1234",
        f"{n} crops: {serial * 1000 / n:.1f} ms each serial, {parallel * 1000 / n:.1f} ms each from {threads} "
        f"threads ({serial / parallel:.1f}x); read {text!r} ({score:.2f})",
    )
    return ok


def draw(image, lines):
    out = image.copy()
    for ln in lines:
        cv2.polylines(out, [ln.poly.astype(np.int32)], True, (80, 200, 120), 2)
        x, y = ln.poly[:, 0].min(), ln.poly[:, 1].min()
        cv2.putText(out, ln.text, (int(x), max(14, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--endpoint", default="tcp://localhost:5555")
    ap.add_argument("--plate-format", default="uae", help=f"uae, {sorted(plates.FORMATS)} or a template")
    ap.add_argument("--save", help="folder for annotated copies")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--debug", action="store_true", help="print every OCR box (text, height, centre)")
    args = ap.parse_args()

    client = Client(args.endpoint, timeout_ms=20000)  # the first call loads the models
    if not client.ping():
        sys.exit(f"no broker at {args.endpoint}: start `maia-infer broker` and `maia-infer worker ocr`")
    services = client.stats().get("services", {})
    print(f"broker up; services: {sorted(services) if isinstance(services, dict) else services}")
    ocr = OCRClient(client)

    ok = True
    if not args.skip_synthetic:
        print("synthetic checks")
        ok = synthetic(ocr, lambda: Client(args.endpoint, timeout_ms=20000))

    fmt = args.plate_format if args.plate_format in plates.FORMATS else [args.plate_format]
    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            print(f"\n{path}: cannot read")
            continue
        t0 = time.perf_counter()
        lines = ocr.read(image)
        ms = (time.perf_counter() - t0) * 1000
        reading = uae.parse(lines) if fmt == "uae" else plates.correct(plates.merge_rows(lines), fmt)
        print(f"\n{path}: {len(lines)} lines in {ms:.0f} ms")
        print(f"  text:       {full_text(lines)}")
        print(f"  plate:      {reading.text!r} valid={reading.valid} {reading.fields or ''}")
        print(f"  containers: {containers.find(lines)}")
        if args.debug:
            for ln in lines:
                cx, cy = ln.center
                print(f"    {ln.text!r:<24} h={ln.height:6.1f} centre=({cx:.0f}, {cy:.0f}) score={ln.score:.2f}")
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            out = Path(args.save) / f"{Path(path).stem}_ocr.jpg"
            cv2.imwrite(str(out), draw(image, lines))
            print(f"  saved:      {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
