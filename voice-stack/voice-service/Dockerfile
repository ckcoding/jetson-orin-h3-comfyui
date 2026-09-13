FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8

# 安装系统依赖 + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    git wget curl vim less \
    libsndfile1 libsndfile1-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-dev libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m ensurepip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# 升级 pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

WORKDIR /opt

# 安装 Jetson 专用 torch (从 ComfyUI 目录挂载的 wheel)
COPY torch-2.2.0-cp310-cp310-linux_aarch64.whl /tmp/torch.whl
COPY torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl /tmp/torchvision.whl
RUN pip install --no-cache-dir /tmp/torch.whl /tmp/torchvision.whl \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && rm /tmp/torch.whl /tmp/torchvision.whl

# 安装 torchaudio (匹配 2.2.0)
RUN pip install --no-cache-dir torchaudio==2.2.0 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 安装 sherpa-onnx (SenseVoice ASR)
RUN pip install --no-cache-dir sherpa-onnx \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 安装 CosyVoice 依赖
RUN pip install --no-cache-dir \
    fastapi uvicorn \
    hyperpyyaml modelscope \
    librosa soundfile \
    inflect pyworld \
    conformer diffusers \
    omegaconf hydra-core \
    transformers x-transformers \
    wetext numpy==1.26.4 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 拷贝 CosyVoice 源码并安装
COPY CosyVoice /opt/CosyVoice
RUN cd /opt/CosyVoice && pip install --no-cache-dir -e . \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com || true

EXPOSE 9880

CMD ["python", "/opt/voice-service/server.py"]
