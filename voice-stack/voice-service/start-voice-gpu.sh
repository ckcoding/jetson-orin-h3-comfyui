#!/bin/bash
# GPU 版语音服务启动器 (host-direct, 用 voice-venv 内的 NVIDIA GPU torch)
# 与 start-voice.sh 区别：额外挂入真实 NVIDIA 驱动库 + libgomp LD_PRELOAD，
# 让 CosyVoice 真正跑在 Orin GPU 上(fp16)。CPU 版无这些库也能跑(自动回退)。
set -a
source /opt/m/voice-env.sh

# /opt/m0 在重启后会被重置为 noexec，必须重新挂 exec 才能加载里面的 .so
mount -o remount,exec /opt/m0 2>/dev/null || true

# 真实 GPU 驱动库 (host /usr/lib 里的 libcuda 等, 容器/venv 中是 stub)
export LD_LIBRARY_PATH="/opt/m0/nvlibs:${LD_LIBRARY_PATH}"

# 解决 sklearn/numba 自带 libgomp 的 static TLS 冲突
LIBGOMP=$(ls /opt/m/voice-venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp*.so* 2>/dev/null | head -1)
[ -n "$LIBGOMP" ] && export LD_PRELOAD="$LIBGOMP"

# CosyVoice GPU 开关 (fp16 默认开; jit/trt 需要时置 1)
export CV_GPU=1
export CV_FP16=1
export CV_JIT=${CV_JIT:-0}
export CV_TRT=${CV_TRT:-0}

echo "[start-voice-gpu] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[start-voice-gpu] LD_PRELOAD=$LD_PRELOAD"
echo "[start-voice-gpu] CV_GPU=$CV_GPU CV_FP16=$CV_FP16 CV_JIT=$CV_JIT CV_TRT=$CV_TRT"
exec /opt/m/voice-venv/bin/python /opt/m/voice-service/server.py
