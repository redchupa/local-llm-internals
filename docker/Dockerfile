# Reproducible CUDA + PyTorch + C++/CMake environment for RTX 5070 Ti (16GB VRAM)
# Usage:
#   docker build -t local-llm-internals:dev -f docker/Dockerfile .
#   docker run --gpus all -it --rm -v "${PWD}:/workspace" local-llm-internals:dev

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    gdb \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu128 torch && \
    pip install numpy pytest transformers peft trl bitsandbytes accelerate

WORKDIR /workspace

CMD ["/bin/bash"]
