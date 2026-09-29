FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY predict.py config.json ./
COPY LICENSE-DINOv3.md README.md ./
COPY weights ./weights
RUN python -c "from pathlib import Path; expected = {f'model_{i}.safetensors' for i in range(5)}; actual = {p.name for p in Path('weights').iterdir() if p.is_file()}; assert actual == expected, (expected, actual)"
ENTRYPOINT ["python", "predict.py"]
