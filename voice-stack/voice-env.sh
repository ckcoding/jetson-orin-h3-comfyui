#!/usr/bin/env bash
# Voice Service 环境配置 (独立于 ComfyUI)
# 用法: source /opt/m/voice-env.sh

if ! mount -o remount,exec /opt/m 2>/dev/null; then
    echo "[voice-env] WARN: could not remount /opt/m with exec (may already be rw,exec)" >&2
fi

export VOICE_VENV=/opt/m/voice-venv
export PATH="${VOICE_VENV}/bin:${PATH}"

# Runtime 库 (与 comfyui-env.sh 相同来源: /opt/m/ComfyUI/runtime 下解包的 deb)
export LD_LIBRARY_PATH="/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu/openblas-pthread:/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu:/opt/m/ComfyUI/runtime/usr/local/cuda-11.4/targets/sbsa-linux/lib:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 模型路径
export COSYVOICE_MODEL_DIR=/opt/m/models/cosyvoice/models/damo--cosyvoice-300m/snapshots/master
export SENSEVOICE_MODEL_DIR=/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

# 离线模式 (模型已在本地)
export HF_HUB_OFFLINE=0
export MODELSCOPE_CACHE=/opt/m/.modelscope
