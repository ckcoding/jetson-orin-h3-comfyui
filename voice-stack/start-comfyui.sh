#!/usr/bin/env bash
set -euo pipefail

source /opt/m/comfyui-env.sh
cd /opt/m/ComfyUI

if ss -lntp 2>/dev/null | grep -q ':8188'; then
    echo "[start] ComfyUI is already listening on port 8188"
    ss -lntp | grep ':8188'
    exit 0
fi

log_file=/opt/m/ComfyUI/comfyui.log
pid_file=/opt/m/ComfyUI/comfyui.pid
nohup python main.py --listen 0.0.0.0 --port 8188 --enable-dynamic-vram --vram-headroom 1 --cache-lru 1 --disable-pinned-memory >"${log_file}" 2>&1 &
comfy_pid=$!
echo "${comfy_pid}" >"${pid_file}"
echo "[start] launched PID ${comfy_pid}; log: ${log_file}"

for _ in $(seq 1 60); do
    if ss -lntp 2>/dev/null | grep -q ':8188'; then
        echo "[start] ComfyUI is listening on port 8188"
        ss -lntp | grep ':8188'
        exit 0
    fi
    if ! kill -0 "${comfy_pid}" 2>/dev/null; then
        echo "[start] ERROR: ComfyUI exited during startup" >&2
        tail -n 100 "${log_file}" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "[start] ERROR: timed out waiting for port 8188" >&2
tail -n 100 "${log_file}" >&2 || true
exit 1
