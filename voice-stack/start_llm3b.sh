#!/bin/bash
export LD_LIBRARY_PATH=/var/opt/llama.cpp/lib:$LD_LIBRARY_PATH
pkill -9 -f llama-server 2>/dev/null
sleep 2
rm -f /tmp/llama3b.log
MODEL=/opt/m/models/qwen_gguf2/models/Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots/master/qwen2.5-3b-instruct-q4_k_m.gguf
nohup /var/opt/llama.cpp/bin/llama-server -m "$MODEL" --host 0.0.0.0 --port 8080 -ngl 99 -c 2048 -t 8 > /tmp/llama3b.log 2>&1 &
echo $! > /tmp/llama3b.pid
echo "Started PID=$(cat /tmp/llama3b.pid)"
