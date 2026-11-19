# CPU image: RapidOCR (PP-OCRv6 small, models bundled in the pip package). Runs the broker too.
#   maia-infer broker            maia-infer worker ocr
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 MAIA_BROKER=tcp://broker:5556
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
 && (pip uninstall -y opencv-python opencv-contrib-python || true) \
 && pip install --force-reinstall --no-deps "opencv-python-headless>=4.8" \
 && python -c "import cv2, numpy as np; from maia_ocr.backends import build; \
b = build('rapid'); print('ok: maia-ocr', __import__('maia_ocr').__version__, '| opencv', cv2.__version__, \
'| smoke', b.recognize([np.full((32, 96, 3), 255, np.uint8)]))"
CMD ["maia-infer", "worker", "ocr"]
