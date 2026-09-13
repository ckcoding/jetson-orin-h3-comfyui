#!/bin/bash
# Voice Service 启动脚本 (ASR + TTS)
source /opt/m/voice-env.sh
exec /opt/m/voice-venv/bin/python /opt/m/voice-service/server.py
