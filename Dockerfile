# Bhutan document + liveness verification prototype.
#
# Two things this image has to get right beyond a plain "pip install":
#   1. pytesseract is only a wrapper — the actual OCR engine (the `tesseract`
#      binary) is a system package and must be installed separately.
#   2. opencv-contrib-python's wheel links against libGL/libglib, which the
#      slim Python base images do not ship.
#
# The ~37MB SFace recognition model is baked in at build time rather than
# downloaded on first request, so a running container makes no external
# calls — the same property the README claims for the local setup.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing app.py/vision.py doesn't re-run the install.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

# Pre-fetch the face-recognition weights that vision.py would otherwise pull
# on first use. Kept as a separate layer so it survives source edits.
RUN curl -fsSL -o backend/models/face_recognition_sface.onnx \
      "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
    && python -c "import pathlib,sys; p=pathlib.Path('backend/models/face_recognition_sface.onnx'); s=p.stat().st_size; print(f'sface model: {s} bytes'); sys.exit(0 if s > 1_000_000 else 1)"

# Drop privileges. The models dir stays writable so vision.py's download
# fallback still works if the baked-in file is ever missing.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/backend/models
USER appuser

# app.py does `import vision` and resolves the frontend as ../frontend,
# so the process has to start from inside backend/.
WORKDIR /app/backend

# Render (and most container hosts) inject the port to bind on via $PORT and
# route external traffic to it. Locally nothing sets it, so 8000 stays the
# default and docker-compose's 127.0.0.1:8000 mapping keeps working.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/api/health\", timeout=4).status==200 else 1)"

# Shell form on purpose: exec form would pass the literal string "$PORT" to
# uvicorn instead of expanding it.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
