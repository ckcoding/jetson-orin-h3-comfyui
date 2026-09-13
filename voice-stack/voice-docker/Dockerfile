FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 音频/系统依赖 (torch 需要 openmpi, 容器 overlay 内安装, 不影响 host)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev openmpi-bin \
    libsndfile1 libsndfile1-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl wget vim less \
    && rm -rf /var/lib/apt/lists/*

# voice-venv 的 python 解释器是 host 的 /opt/m/ComfyUI/python/python/bin/python3.10 (挂载进来)
# venv 激活时需要 runtime 库路径, 由启动脚本设置

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 9880

ENTRYPOINT ["/opt/entrypoint.sh"]
