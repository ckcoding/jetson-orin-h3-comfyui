#!/bin/bash
source /opt/m/voice-env.sh 2>/dev/null
export LD_PRELOAD=/opt/m/voice-venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
export MPLCONFIGDIR=/tmp/matplotlib
cd /opt/m
exec /opt/m/voice-venv/bin/python /opt/m/f5_ws_server.py >> /opt/m0/f5_ws.log 2>&1
