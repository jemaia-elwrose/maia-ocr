"""Read number plates from a traffic stream: YOLO finds vehicles, OCR reads the lower
part of each vehicle, the plate format fixes OCR slips, and a per-track vote gives one
stable plate per vehicle.

    maia-infer broker &  maia-infer worker yolo &  maia-infer worker ocr &
    python examples/plates_from_video.py --source traffic.mp4 --format uae --show

For production, swap the "lower half of the vehicle" heuristic for a dedicated
plate detector (another maia-infer worker) and use your pipeline's tracker.
"""

import argparse

import cv2
from maia_infer import Client, detections

from maia_ocr import PlateVoter, plates
from maia_ocr.client import OCRClient


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--endpoint", default="tcp://localhost:5555")
    ap.add_argument("--format", default="generic", help=f"{sorted(plates.FORMATS)} or a template")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    fmt = args.format if args.format in plates.FORMATS else [args.format]

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    client = Client(args.endpoint)
    ocr, voter = OCRClient(client), PlateVoter(min_votes=3)
    tracks, next_id = {}, 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vehicles = detections(client.infer("yolo", frame, meta={"classes": ["car", "truck", "bus"]}))
        new_tracks = {}
        for v in vehicles:
            box = v["box"]
            match = max(((iou(box, b), t) for t, b in tracks.items()), default=(0, None))
            tid = match[1] if match[0] > 0.3 else next_id
            next_id += tid == next_id
            new_tracks[tid] = box

            x1, y1, x2, y2 = (int(c) for c in box)
            crop = frame[y1 + (y2 - y1) // 2 : y2, x1:x2]  # plates sit in the lower half
            if crop.size and min(crop.shape[:2]) > 16:
                reading = ocr.read_plate(crop, fmt)
                plate = (
                    voter.update(tid, reading.text, 1.0 if reading.valid else 0.3)
                    if reading.valid
                    else voter.decision(tid)
                )
            else:
                plate = voter.decision(tid)
            color = (80, 200, 120) if plate else (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid} {plate or '...'}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        tracks = new_tracks
        if args.show:
            cv2.imshow("maia-ocr", frame)
            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":
    main()
