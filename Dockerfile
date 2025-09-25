FROM python:3.11-slim

# System deps for torch/vision + PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown==5.2.0

# Download model from Google Drive (make sure file is shared: Anyone with the link)
ARG MODEL_DRIVE_ID=17c8ySOwxuQZsWRAFplqly4wbDaQHCQPC
ENV MODEL_DRIVE_ID=${MODEL_DRIVE_ID}
RUN gdown --fuzzy "https://drive.google.com/uc?id=17c8ySOwxuQZsWRAFplqly4wbDaQHCQPC" -O /app/spoilage_model.pth

# App
COPY server.py /app/server.py

# App env
ENV MODEL_PATH=/app/spoilage_model.pth
ENV IMG_SIZE=224
ENV CLASS_NAMES="fresh,spoiled"
ENV PYTHONUNBUFFERED=1

# Ports
EXPOSE 7860

# Run (bind to dynamic $PORT if provided, else fallback 7860)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
