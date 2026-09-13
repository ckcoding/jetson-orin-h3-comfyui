#!/bin/bash
# speech-to-speech 容器管理脚本 —— NVIDIA Jetson Orin / JetPack 5.1.2
#
#   bash /opt/m/ss/ss-docker.sh {start|stop|restart|status|logs|shell|check|update}
#
# 说明：容器自身只提供 CUDA 11.4 + cuDNN 8.6.0 运行时和入口脚本，
#       模型与 python 环境以卷挂载方式来自宿主机。
#       STT/TTS 默认走 9880 voice-service（经 voice_adapter 转 OpenAI 协议），
#       adapter 由本脚本随容器一起拉起/停止。
set -e

NAME="speech-to-speech"
IMAGE="speech-to-speech:v1"
SS_DATA="/opt/update/speech-to-speech"

# ---- 协议适配层（把 9880 的 SenseVoice/melo 包装成 OpenAI 兼容接口）----
ADAPTER="/opt/m/ss/voice_adapter.py"
ADAPTER_PORT="${ADAPTER_PORT:-9881}"
ADAPTER_LOG="/opt/m/ss/adapter.log"
ADAPTER_PIDFILE="/opt/m/ss/adapter.pid"
SS_PY="${SS_PY:-/opt/m/ss-venv/bin/python}"

# 并发默认 1 路（单人通话 / 单路常驻场景）。
#   1 路 → 单会话独占，无争抢，延迟最低，内存最省  ← 默认
#   4 路 → 首包中位 42s，占用 ~9G    （稳，余量大）
#   6 路 → 首包中位 63s，占用 ~12G   （能力上限且安全，宿主余 ~8G）
#   8 路 → 首包中位 82s，占用 17-19G （能跑通但内存不释放，宿主剩 <4G，有 OOM 风险）
# 注意：池里每个 unit 是独立一整套 VAD+STT+TTS，内存随路数线性增长。
#       接入 voice_adapter（SS_STT/TTS_BACKEND=openai）后模型不再走池内，
#       实测空闲占用仅 ~500MB，上表数字对应的开销此时已不适用。
# 调整：SS_NUM_PIPELINES=N bash ss-docker.sh start    （必须 start 重建，restart 不生效）

# Tegra GPU 设备节点
GPU_DEVICES=(
  --device /dev/nvhost-gpu --device /dev/nvhost-dbg-gpu --device /dev/nvhost-prof-gpu
  --device /dev/nvhost-tsg-gpu --device /dev/nvhost-ctxsw-gpu --device /dev/nvhost-sched-gpu
  --device /dev/nvhost-nvsched-gpu --device /dev/nvhost-power-gpu
  --device /dev/nvmap --device /dev/nvidia0 --device /dev/nvidiactl
)

# Tegra 驱动库（libcuda 依赖链）
DRIVER_LIBS=(
  libcuda.so.1 libnvcucompat.so libnvos.so libnvrm_chip.so libnvrm_gpu.so
  libnvrm_host1x.so libnvrm_mem.so libnvrm_sync.so libnvsciipc.so libnvsocsys.so
  libnvtegrahv.so libnvrm_gpusched.so libnvrm_interop_gpu.so libnvrm_stream.so
  libnvrm_surface.so
)
MOUNTS=()
for lib in "${DRIVER_LIBS[@]}"; do
  [ -f "/usr/lib/$lib" ] && MOUNTS+=(-v "/usr/lib/$lib:/lib/aarch64-linux-gnu/$lib:ro")
done

# ---------------------------------------------------------------- adapter
# 适配层推荐交给 systemd 管理（voice-adapter.service：Restart=always + 开机自启）。
# unit 存在时本脚本只做委托，避免脚本与 systemd 同时管一个进程（会互相拉黑/端口打架）。
# 安装：cp voice-adapter.service /etc/systemd/system/ && systemctl enable --now voice-adapter
ADAPTER_UNIT="voice-adapter.service"

adapter_by_systemd() {
  [ -f "/etc/systemd/system/${ADAPTER_UNIT}" ] && command -v systemctl >/dev/null 2>&1
}

# 等 9881 健康（端口通 ≠ 起来，所以直接打 /health）
wait_adapter() {
  for _ in $(seq 1 20); do
    if wget -q -O- "http://127.0.0.1:${ADAPTER_PORT}/health" >/dev/null 2>&1; then
      echo "adapter 就绪"
      return 0
    fi
    sleep 1
  done
  echo "adapter 启动失败，日志（tail）："
  if adapter_by_systemd; then journalctl -u "$ADAPTER_UNIT" -n 20 --no-pager
  else tail -20 "$ADAPTER_LOG" 2>/dev/null; fi
  return 1
}

start_adapter() {
  echo "=== 启动协议适配层（adapter 127.0.0.1:${ADAPTER_PORT}）==="

  if adapter_by_systemd; then
    systemctl restart "$ADAPTER_UNIT"
    wait_adapter
    return $?
  fi

  if [ -f "$ADAPTER_PIDFILE" ]; then
    kill "$(cat "$ADAPTER_PIDFILE")" 2>/dev/null || true
    rm -f "$ADAPTER_PIDFILE"
  fi
  pkill -f "voice_adap[t]er\.py" 2>/dev/null || true
  sleep 1

  # TTS_ENGINE 必须用 fanchen：melo 在本机 9880 上合成语速约 11 字/秒
  # （正常约 4 字/秒），听感为快进，且 SenseVoice 无法识别其输出；fanchen 原生 16kHz、语速正常。
  ADAPTER_PORT="$ADAPTER_PORT" TTS_ENGINE="${SS_TTS_ENGINE:-fanchen}" \
    TTS_FALLBACK_ENGINES="${SS_TTS_FALLBACK:-melo}" \
    TTS_OUT_RATE="${SS_TTS_RATE:-16000}" ADAPTER_VERBOSE="${ADAPTER_VERBOSE:-}" \
    setsid "$SS_PY" "$ADAPTER" > "$ADAPTER_LOG" 2>&1 < /dev/null &
  echo $! > "$ADAPTER_PIDFILE"
  wait_adapter
}

stop_adapter() {
  if adapter_by_systemd; then
    systemctl stop "$ADAPTER_UNIT"
    echo "已停止 adapter（systemd）"
    return 0
  fi
  if [ -f "$ADAPTER_PIDFILE" ]; then
    kill "$(cat "$ADAPTER_PIDFILE")" 2>/dev/null || true
    rm -f "$ADAPTER_PIDFILE"
  fi
  pkill -f "voice_adap[t]er\.py" 2>/dev/null || true
  echo "已停止 adapter"
}

# 只用在本机已存在的上游服务时才需要 adapter
needs_adapter() {
  [ "${SS_STT_BACKEND:-openai}" = "openai" ] || [ "${SS_TTS_BACKEND:-openai}" = "openai" ]
}

# 等待真正就绪：端口通了不代表模型加载完（uvicorn 先起，模型后加载）。
# 以日志出现 "OpenAI Realtime API starting" 为准，多路时尤其重要。
wait_ready() {
  local want="${SS_NUM_PIPELINES:-1}" port="${SS_PORT:-8765}"
  local deadline=$(( $(date +%s) + ${SS_READY_TIMEOUT:-1800} ))
  echo "=== 等待就绪（${want} 路）==="
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
      echo "容器已退出，日志："; docker logs --tail 40 "$NAME"; return 1
    fi
    # /v1/pool 里 unit 初始状态就是 idle，不代表模型加载完，不能用它判就绪。
    # 这一行日志在全部 handler 加载并 warmup 之后才打印，才是真正的就绪标志。
    if docker logs "$NAME" 2>&1 | grep -q "OpenAI Realtime API starting"; then
      echo "已就绪（pool size ${want}）→ ws://0.0.0.0:${port}/v1/realtime"
      return 0
    fi
    sleep 3
  done
  echo "超时未就绪，查看：bash $0 logs"
  return 1
}

case "${1:-start}" in

start)
  echo "=== 启动 $NAME（${SS_NUM_PIPELINES:-1} 路并发）==="
  mkdir -p "${SS_DATA}"

  # 上游健康检查：9880 不在的话 STT/TTS 会全挂
  if needs_adapter; then
    if ! wget -q -O- "http://127.0.0.1:9880/health" >/dev/null 2>&1; then
      echo "警告：上游 voice-service(9880) 无响应，STT/TTS 会失败"
    fi
    start_adapter
  fi

  docker rm -f "$NAME" >/dev/null 2>&1 || true

  ENV_ARGS=(
    -e SS_PORT="${SS_PORT:-8765}"
    -e SS_LLM_BASE="${SS_LLM_BASE:-http://127.0.0.1:8080/v1}"
    -e SS_NUM_PIPELINES="${SS_NUM_PIPELINES:-1}"
    -e SS_USE_GPU="${SS_USE_GPU:-0}"
    -e SS_STT_BACKEND="${SS_STT_BACKEND:-openai}"
    -e SS_TTS_BACKEND="${SS_TTS_BACKEND:-openai}"
    -e SS_ADAPTER_BASE="${SS_ADAPTER_BASE:-http://127.0.0.1:${ADAPTER_PORT}/v1}"
    -e SS_TTS_VOICE="${SS_TTS_VOICE:-default}"
    -e SS_TTS_RATE="${SS_TTS_RATE:-16000}"
    -e MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
  )
  # 仅在显式设置时才覆盖（否则用 entrypoint 里按后端选的默认值）
  for v in SS_STT_MODEL SS_TTS_MODEL SS_STT_LANG SS_STT_TIMEOUT SS_TTS_TIMEOUT \
           SS_SYSTEM_PROMPT SS_LOG_LEVEL SS_TTS_DEVICE SS_STT_DEVICE; do
    [ -n "${!v:-}" ] && ENV_ARGS+=(-e "$v=${!v}")
  done

  docker run -d \
    --name "$NAME" \
    --privileged \
    --network host \
    --restart unless-stopped \
    "${GPU_DEVICES[@]}" \
    "${MOUNTS[@]}" \
    "${ENV_ARGS[@]}" \
    -v /opt/m:/opt/m \
    -v /opt/update:/opt/update \
    -v /opt/other/nltk_data:/opt/other/nltk_data \
    "$IMAGE" serve

  wait_ready
  ;;

stop)
  docker stop "$NAME" >/dev/null 2>&1 && echo "已停止 $NAME" || echo "$NAME 未运行"
  stop_adapter
  ;;

restart)
  # 必须重建容器：docker restart 会沿用旧的环境变量，
  # 改 SS_NUM_PIPELINES / SS_STT_BACKEND 等参数不会生效。
  # 只想不改配置快速重启，直接 docker restart $NAME。
  echo "重建容器以应用当前配置（SS_NUM_PIPELINES=${SS_NUM_PIPELINES:-1}）"
  bash "$0" start
  ;;

status)
  docker ps -a --filter "name=^${NAME}$" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" || true
  echo "--- 监听端口 ---"
  ss -lntp 2>/dev/null | grep -E ":(8765|8080|9880|${ADAPTER_PORT})" || echo "(无监听)"
  echo "--- 协议适配层 ---"
  if wget -q -O- "http://127.0.0.1:${ADAPTER_PORT}/health" 2>/dev/null; then
    echo
  else
    echo "adapter 无响应（STT/TTS 走 openai 后端时会失败）"
  fi
  echo "--- 流水线池 ---"
  wget -q -O- "http://127.0.0.1:${SS_PORT:-8765}/v1/pool" 2>/dev/null || echo "(无响应)"
  echo
  echo "--- 用量 ---"
  wget -q -O- "http://127.0.0.1:${SS_PORT:-8765}/v1/usage" 2>/dev/null || echo "(无响应)"
  echo
  ;;

logs)
  docker logs -f --tail 80 "$NAME"
  ;;

adapter-logs)
  tail -f "$ADAPTER_LOG"
  ;;

shell)
  docker exec -it "$NAME" bash
  ;;

check)
  docker exec "$NAME" /opt/ss-entrypoint.sh check
  ;;

bench)
  # 分阶段测速：定位 STT / LLM / TTS 各自耗时
  "$SS_PY" /opt/m/bench_stages.py
  ;;

update)
  echo "=== 重建镜像 ==="
  cd /opt/update/ss-docker && docker build -t "$IMAGE" .
  bash "$0" start
  ;;

*)
  echo "用法: $0 {start|stop|restart|status|logs|adapter-logs|shell|check|bench|update}"
  ;;
esac
