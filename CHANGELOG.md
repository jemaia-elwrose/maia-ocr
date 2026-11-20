# Changelog

## 0.1.1
- `uae.parse`: UAE plates by layout (emirate, code, number), tested on real OCR output.
  Used by `read_plate(crop, "uae")` and `maia-ocr plate --format uae`.
- PlateVoter keeps `?` for unreadable characters and fills them from other frames.
- Rows never join text more than 2x apart in height (tilted plates).
- Container search anchors codes at word starts (no false positives from weight markings).
- `examples/broker_test.py`: end-to-end check through the broker.
- Letter-code emirates (Dubai, Ajman, RAK, UAQ, Fujairah) ignore stray digits as codes;
  a lone `0`/`O`/`Q` box beside the number becomes an ambiguous code (paddle reads `D` that way).
- Letter codes are accepted down to 1/4 of the number's height (Dubai's code letter is small);
  numeric codes still need 0.4.
- `examples/broker_test.py --debug` prints every OCR box.
- Docker: `DATA_DIR` for the CLI mount and a clearer error; paddle GPU image (`--profile gpu`,
  keeps PaddleX's opencv-contrib-python, which its dependency check requires by name).

## 0.1.0
- First release: `ocr` maia-infer worker (read and batched recognize modes), RapidOCR
  (PP-OCRv6, bundled models) and PaddleOCR backends, plate formats with position-aware
  correction, two-row plates, PlateVoter, ISO 6346 container codes, reading order, CLI, Docker.
