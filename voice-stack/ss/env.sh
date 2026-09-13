#!/bin/bash
# speech-to-speech (HuggingFace) 运行环境 — Jetson Orin / JetPack 5 / Ubuntu 20.04
# 用法: source /opt/m/ss/env.sh

# /opt/m 需 remount 为 exec 才能执行 venv 里的二进制
mount -o remount,exec /opt/m 2>/dev/null

export SS_HOME=/opt/m/ss
export SS_VENV=/opt/m/ss-venv                 # 复用 ComfyUI 自带的 python3.10 建的 venv
export SS_DATA=/opt/update/speech-to-speech   # 日志/pid（/opt/update 空间大）
export PATH="${SS_VENV}/bin:${PATH}"

# Jetson 运行时库：openblas(torch 依赖) + CUDA 11.4 + 系统库
export LD_LIBRARY_PATH="/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu/openblas-pthread:/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu:/opt/m/ComfyUI/runtime/usr/local/cuda-11.4/targets/sbsa-linux/lib:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 模型/数据目录：权重放 /opt/update（/opt/m 只剩几百 MB）
export HF_HOME=/opt/update/hf
export MODELSCOPE_CACHE=/opt/m/.modelscope
export NLTK_DATA=/opt/other/nltk_data

# 本机系统 CA 不完整，python 走 certifi 才能访问 github / huggingface
export SSL_CERT_FILE="${SS_VENV}/lib/python3.10/site-packages/certifi/cacert.pem"
export REQUESTS_CA_BUNDLE="${SSL_CERT_FILE}"
export CURL_CA_BUNDLE="${SSL_CERT_FILE}"

# 线程/日志
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${SS_DATA}" 2>/dev/null

# torch.hub / silero VAD 缓存：/root/.cache 是只读的，改到 /opt/update
export XDG_CACHE_HOME="${SS_DATA}/cache"
export TORCH_HOME="${SS_DATA}/cache/torch"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" 2>/dev/null
