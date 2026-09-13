#!/bin/bash
export LD_LIBRARY_PATH=/var/opt/llama.cpp/lib:$LD_LIBRARY_PATH
pkill -9 -f llama-server 2>/dev/null
sleep 2
rm -f /tmp/llama7b.log
nohup /var/opt/llama.cpp/bin/llama-server -m /opt/m/models/qwen7b.gguf --host 0.0.0.0 --port 8080 -ngl 0 -c 2048 -t 11 > /tmp/llama7b.log 2>&1 &
echo $! > /tmp/llama7b.pid
echo "Started PID=$(cat /tmp/llama7b.pid)"
