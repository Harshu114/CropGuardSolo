# CropGuard API Dockerfile for Hugging Face Spaces / GPU deployment
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code.
# losses.py and inference_utils.py are REQUIRED: model_server.py imports
# dfl_decode / box_xywh_to_xyxy from losses and nms from inference_utils. They
# were missing from this COPY list, so the image failed at import time.
COPY model.py .
COPY losses.py .
COPY inference_utils.py .
COPY model_server.py .
COPY main.py .

# v5 checkpoint (11 classes). Baked in so the image runs standalone; mount or
# override with MODEL_PATH for a different checkpoint.
COPY cropguard_model_v5.pt /app/cropguard_model_v5.pt
ENV MODEL_PATH=/app/cropguard_model_v5.pt

EXPOSE 8000

CMD ["python3", "main.py"]