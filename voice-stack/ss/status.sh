#!/bin/bash
# 查看 speech-to-speech 服务状态
PIDFILE=/opt/update/speech-to-speech/ss.pid
LOGFILE=/opt/update/speech-to-speech/ss.log
PORT="${SS_PORT:-8765}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "状态: 运行中 (pid $(cat "$PIDFILE"))"
else
    echo "状态: 未运行"
fi
echo "监听:"; ss -lntp 2>/dev/null | grep -E ":(8765|8080)" || true
echo "健康检查:"; curl -s -m 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "(无 curl，跳过)"
echo "--- 日志尾部 ---"; tail -n 20 "$LOGFILE" 2>/dev/null
