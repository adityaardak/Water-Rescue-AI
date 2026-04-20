FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    HF_HOME=/home/appuser/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/huggingface /tmp/matplotlib /tmp/ultralytics \
    && chown -R appuser:appuser /home/appuser /tmp/matplotlib /tmp/ultralytics /app

COPY requirements-docker.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements-docker.txt

COPY app.py .
COPY yolo11n.pt* ./

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
