FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ----
# Use CPU wheels for torch/vision (more reliable on slim)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown==5.2.0

# ---- Model download (ensure shared link is public) ----
ARG MODEL_DRIVE_ID=17c8ySOwxuQZsWRAFplqly4wbDaQHCQPC
ENV MODEL_DRIVE_ID=${MODEL_DRIVE_ID}
RUN gdown --fuzzy "https://drive.google.com/uc?id=${MODEL_DRIVE_ID}" -O /app/spoilage_model.pth || echo "model download skipped"

# ---- App ----
COPY server.py /app/server.py

ENV MODEL_PATH=/app/spoilage_model.pth
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

# ---- Start ----
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
