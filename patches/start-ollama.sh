#!/bin/bash
# Ollama (JetPack5) 启动脚本 — Jetson Orin 车载平台
# 用法: bash start-ollama.sh   (API: http://<board>:8080)
export HOME=/opt/update/ollama/home
export OLLAMA_MODELS=/opt/update/ollama/models
export OLLAMA_HOST=0.0.0.0:8080
export OLLAMA_DEBUG=1
export JETSON_JETPACK=5
export OLLAMA_LIBRARY_PATH=/opt/update/ollama/lib/ollama:/opt/update/ollama/lib/ollama/cuda_jetpack5
export LD_LIBRARY_PATH=/opt/update/ollama/lib/ollama/cuda_jetpack5:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH

# /opt/update 若为 noexec 挂载需要先 remount (平台默认)
mount -o remount,exec /opt/update 2>/dev/null

pkill -x ollama 2>/dev/null; sleep 2
exec /opt/update/ollama/bin/ollama serve
