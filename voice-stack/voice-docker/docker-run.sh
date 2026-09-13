#!/bin/bash
# Voice Service Docker 运行脚本 (Jetson GPU)
# 用法: ./docker-run.sh {start|stop|logs|shell|status}

IMAGE=voice-service:v2
NAME=voice-service

# Jetson GPU 设备
GPU_DEVICES=(
  --device /dev/nvhost-as-gpu --device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu
  --device /dev/nvhost-gpu --device /dev/nvhost-dbg-gpu --device /dev/nvhost-prof-gpu
  --device /dev/nvhost-tsg-gpu --device /dev/nvhost-ctxsw-gpu --device /dev/nvhost-sched-gpu
  --device /dev/nvhost-nvsched-gpu --device /dev/nvhost-power-gpu
  --device /dev/nvmap --device /dev/nvidia0 --device /dev/nvidiactl
)

# Tegra 驱动库 (libcuda 依赖链 + 动态加载的 libnvcucompat)
DRIVER_LIBS=(libcuda.so.1 libnvcucompat.so libnvos.so libnvrm_chip.so libnvrm_gpu.so
  libnvrm_host1x.so libnvrm_mem.so libnvrm_sync.so libnvsciipc.so libnvsocsys.so
  libnvtegrahv.so libnvrm_gpusched.so libnvrm_interop_gpu.so libnvrm_stream.so
  libnvrm_surface.so)
MOUNTS=()
for lib in "${DRIVER_LIBS[@]}"; do
  [ -f "/usr/lib/$lib" ] && MOUNTS+=(-v "/usr/lib/$lib:/lib/aarch64-linux-gnu/$lib:ro")
done
# host 独有的库 (runtime 目录缺的)
[ -f /usr/lib/aarch64-linux-gnu/libevent-2.1.so.7 ] && MOUNTS+=(-v /usr/lib/aarch64-linux-gnu/libevent-2.1.so.7:/lib/aarch64-linux-gnu/libevent-2.1.so.7:ro)
[ -f /usr/lib/aarch64-linux-gnu/libevent-2.1.so.7.0.0 ] && MOUNTS+=(-v /usr/lib/aarch64-linux-gnu/libevent-2.1.so.7.0.0:/lib/aarch64-linux-gnu/libevent-2.1.so.7.0.0:ro)

case "${1:-start}" in

start)
    echo "=== 启动 $NAME (含 llama-server + 语音全栈) ==="
    # 停掉 host 上的服务避免 GPU 上下文冲突 (关键!)
    systemctl stop voice-llm 2>/dev/null
    systemctl stop voice-service 2>/dev/null

    docker rm -f $NAME 2>/dev/null
    docker run -d \
      --name $NAME \
      --privileged \
      --network host \
      --restart unless-stopped \
      "${GPU_DEVICES[@]}" \
      "${MOUNTS[@]}" \
      -v /opt/m:/opt/m \
      -v /var/opt/llama.cpp:/var/opt/llama.cpp \
      voice-service:v2 all

    echo "=== 容器已启动 ==="
    docker ps | grep $NAME
    echo "LLM API:  http://0.0.0.0:8080"
    echo "Voice WS: ws://0.0.0.0:9880/ws"
    ;;

stop)
    docker rm -f $NAME 2>/dev/null
    systemctl start voice-llm 2>/dev/null   # 恢复 host LLM
    systemctl start voice-service 2>/dev/null  # 恢复 host 语音
    echo "已切回 host 服务"
    ;;

logs)
    docker logs -f $NAME
    ;;

shell)
    docker exec -it $NAME bash
    ;;

status)
    docker ps -a | grep $NAME
    docker logs --tail 20 $NAME 2>/dev/null
    ;;
*)
    echo "用法: $0 {start|stop|logs|shell|status}"
    ;;
esac
