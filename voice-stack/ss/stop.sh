#!/bin/bash
# 停止 speech-to-speech 服务
PIDFILE=/opt/update/speech-to-speech/ss.pid
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        for i in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$PID" 2>/dev/null
        echo "已停止 pid=$PID"
    fi
    rm -f "$PIDFILE"
else
    pkill -f "speech-to-speech serve" && echo "已按进程名停止" || echo "未在运行"
fi
