#!/bin/bash
# 启动 llama-server (Qwen 27B)
# 用法: bash start_llm.sh

export LD_LIBRARY_PATH=/var/opt/llama.cpp/lib:$LD_LIBRARY_PATH

MODEL="/opt/update/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
LOG="/tmp/llama.log"
PID_FILE="/tmp/llama.pid"

# 杀掉旧的
pkill -f llama-server 2>/dev/null
sleep 2

echo "=== 启动 llama-server ==="
nohup /var/opt/llama.cpp/bin/llama-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 35 \
    -c 2048 \
    -t 8 \
    --mlock \
    > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "PID=$(cat $PID_FILE)"
echo "日志: $LOG"

# 等待启动
for i in $(seq 1 60); do
    if ss -lntp 2>/dev/null | grep -q ':8080'; then
        echo "✅ llama-server 启动成功 (等待 ${i}s)"
        echo "API: http://0.0.0.0:8080"
        exit 0
    fi
    if ! kill -0 $(cat $PID_FILE) 2>/dev/null; then
        echo "❌ llama-server 启动失败"
        tail -20 "$LOG"
        exit 1
    fi
    sleep 1
done
echo "❌ 超时"
tail -20 "$LOG"
exit 1
