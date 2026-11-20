# Changelog

## 0.1.1
- `uae.parse`: UAE plates by layout (emirate, code, number), tested on real OCR output.
  Used by `read_plate(crop, "uae")` and `maia-ocr plate --format uae`.
- PlateVoter keeps `?` for unreadable characters and fills them from other frames.
- Rows never join text more than 2x apart in height (tilted plates).
- Container search anchors codes at word starts (no false positives from weight markings).
- `examples/broker_test.py`: end-to-end check through the broker.

## 0.1.0
- First release: `ocr` maia-infer worker (read and batched recognize modes), RapidOCR
  (PP-OCRv6, bundled models) and PaddleOCR backends, plate formats with position-aware
  correction, two-row plates, PlateVoter, ISO 6346 container codes, reading order, CLI, Docker.
