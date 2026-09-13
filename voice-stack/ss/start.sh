#!/bin/bash
# 启动 speech-to-speech Realtime 服务 (OpenAI Realtime 协议, WS: /v1/realtime)
# 用法: bash /opt/m/ss/start.sh
set -e
source /opt/m/ss/env.sh

PIDFILE="${SS_DATA}/ss.pid"
LOGFILE="${SS_DATA}/ss.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "已在运行 pid=$(cat "$PIDFILE")"
    exit 0
fi

# 上游 LLM：本机 llama-server (OpenAI 兼容)
LLM_BASE="${SS_LLM_BASE:-http://127.0.0.1:8080/v1}"
LLM_MODEL="${SS_LLM_MODEL:-/opt/m/models/qwen_gguf2/models/Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots/master/qwen2.5-3b-instruct-q4_k_m.gguf}"
PORT="${SS_PORT:-8765}"

nohup "${SS_VENV}/bin/speech-to-speech" serve \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --num_pipelines "${SS_NUM_PIPELINES:-1}" \
    --stt faster-whisper \
    --faster_whisper_stt_model_name "${SS_STT_MODEL:-small}" \
    --faster_whisper_stt_device cpu \
    --faster_whisper_stt_compute_type int8 \
    --faster_whisper_stt_gen_language auto \
    --llm_backend chat-completions \
    --model_name "${LLM_MODEL}" \
    --responses_api_base_url "${LLM_BASE}" \
    --responses_api_api_key "${SS_LLM_KEY:-sk-local-noauth}" \
    --responses_api_stream \
    --tts kokoro \
    --kokoro_device "${SS_TTS_DEVICE:-cpu}" \
    --init_chat_prompt "你是一个语音助手，回答要简短口语化，一两句话即可。" \
    --log_level INFO \
    >> "${LOGFILE}" 2>&1 &

echo $! > "${PIDFILE}"
sleep 2
echo "已启动 pid=$(cat "$PIDFILE")  日志: ${LOGFILE}"
echo "接口: ws://$(hostname -I | awk '{print $1}'):${PORT}/v1/realtime"
