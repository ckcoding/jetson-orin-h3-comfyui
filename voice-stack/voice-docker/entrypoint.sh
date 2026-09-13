#!/bin/bash
# Voice Service Docker 入口 v2
# 容器内运行: llama-server (GPU) + voice-server (ASR+TTS+WS)

export LD_LIBRARY_PATH="/var/opt/llama.cpp/lib:/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu/openblas-pthread:/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu:/opt/m/ComfyUI/runtime/usr/local/cuda-11.4/targets/sbsa-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/opt/m/ComfyUI/python/python/lib:/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}"

export COSYVOICE_MODEL_DIR=/opt/m/models/cosyvoice/models/damo--cosyvoice-300m/snapshots/master
export SENSEVOICE_MODEL_DIR=/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1

mkdir -p /tmp/mpl

case "${1:-all}" in

  # 只跑语音服务 (LLM 由外部提供)
  voice)
    exec /opt/m/voice-venv/bin/python /opt/m/voice-service/server.py
    ;;

  # 只跑 LLM
  llm)
    exec /var/opt/llama.cpp/bin/llama-server \
      -m /opt/m/models/qwen_gguf2/models/Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots/master/qwen2.5-3b-instruct-q4_k_m.gguf \
      --host 0.0.0.0 --port 8080 -ngl 99 -c 2048 -t 8
    ;;

  # 全栈: LLM + 语音 (默认)
  all|*)
    echo "[entrypoint] 启动 llama-server (GPU)..."
    /var/opt/llama.cpp/bin/llama-server \
      -m /opt/m/models/qwen_gguf2/models/Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots/master/qwen2.5-3b-instruct-q4_k_m.gguf \
      --host 0.0.0.0 --port 8080 -ngl 99 -c 2048 -t 8 \
      > /tmp/llama-server.log 2>&1 &
    LLAMA_PID=$!

    echo "[entrypoint] 等待 LLM 就绪..."
    for i in $(seq 1 30); do
      if wget -q -O- http://localhost:8080/health 2>/dev/null | grep -q ok; then
        echo "[entrypoint] LLM 就绪"; break
      fi
      sleep 2
    done

    echo "[entrypoint] 启动 voice-server (ASR+TTS+WS)..."
    trap "kill $LLAMA_PID 2>/dev/null" EXIT
    /opt/m/voice-venv/bin/python /opt/m/voice-service/server.py
    ;;
esac
