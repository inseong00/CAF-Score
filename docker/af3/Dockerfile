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
WORKDIR /workspace/CAF-Score

# Copy CAF-Score repository
COPY . .

# Install pip packages
RUN pip install \
    accelerate>=1.12.0 \
    audioread>=3.1.0 \
    einops>=0.8.1 \
    huggingface-hub>=1.2.1 \
    librosa>=0.11.0 \
    numba>=0.63.1 \
    numpy>=2.1.2 \
    pandas>=2.3.3 \
    peft>=0.18.0 \
    pillow>=11.3.0 \
    pyyaml>=6.0.3 \
    safetensors>=0.7.0 \
    scikit-learn>=1.7.2 \
    scipy>=1.15.3 \
    soundfile>=0.13.1 \
    timm>=1.0.24 \
    tokenizers>=0.22.1 \
    torch>=2.5.1 \
    torchaudio>=2.5.1 \
    torchvision>=0.20.1 \
    tqdm>=4.67.1 \
    "transformers>=5.0.0.dev0" \
    nnaudio>=0.3.4

CMD ["/bin/bash"]