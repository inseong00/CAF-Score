FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

# Basic settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install basic packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Working directory
WORKDIR /workspace

# Copy CAF-Score repository
COPY . /workspace/CAF-Score
WORKDIR /workspace/CAF-Score

# ================================
# Environment 1: Install caf_qwen3 packages
# (Docker uses a single environment, choose the required one)
# ================================
RUN pip install \
    accelerate>=1.10.1 \
    audioread>=3.0.1 \
    av>=15.1.0 \
    einops>=0.8.1 \
    fastapi>=0.119.0 \
    huggingface-hub>=0.35.3 \
    librosa>=0.11.0 \
    numba>=0.61.2 \
    numpy>=2.2.6 \
    pyyaml>=6.0.2 \
    qwen-omni-utils>=0.0.8 \
    ray>=2.50.1 \
    safetensors>=0.6.2 \
    scikit-learn>=1.7.2 \
    scipy>=1.15.3 \
    sentence-transformers>=5.1.2 \
    soundfile>=0.13.1 \
    timm>=1.0.24 \
    tokenizers>=0.22.1 \
    torch>=2.7.0 \
    torchaudio>=2.7.0 \
    torchvision>=0.22.0 \
    tqdm>=4.67.1 \
    xformers>=0.0.30 \
    nnaudio>=0.3.4

# vLLM (install official version)
RUN pip install vllm==0.13.0

# transformers (version supporting Qwen3-Omni)
RUN pip install transformers==4.57.3

# flash-attention
RUN pip install flash-attn --no-build-isolation

CMD ["/bin/bash"]