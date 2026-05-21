# =====================================================================
# Production Dockerfile for Packaging Voice AI Backend
# =====================================================================
FROM python:3.10-slim-bullseye

# Set environment variables for optimal container behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies required for STT, VAD, and TTS engines:
# - libsndfile1: Required by soundfile for reading/resampling WAV audio.
# - espeak-ng: Required by Kokoro TTS phonemizer engine.
# - ffmpeg: Required by Faster-Whisper to load and preprocess audio segments.
# - build-essential & portaudio19-dev: Required to build pyaudio if wheels aren't loaded.
# - curl: Required for startup health check queries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    espeak-ng \
    ffmpeg \
    build-essential \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list first to optimize Docker layer caching
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ONNX voice models (Kokoro & Piper) directly into the image
# This ensures a fully self-contained deployment that runs offline instantly
COPY models/ /app/models/

# Copy the rest of the application codebase
COPY app/ /app/app/
COPY index.html /app/index.html

# Expose port 8000 for websocket/HTTP server
EXPOSE 8000

# Run the backend server with dynamic reload disabled in production for max performance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
