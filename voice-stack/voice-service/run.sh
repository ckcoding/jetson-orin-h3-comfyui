#!/bin/bash
# 启动语音服务容器 (Jetson GPU)
IMAGE_NAME="voice-service"
CONTAINER_NAME="voice-service"

# GPU 设备
GPU_DEVICES="
--device /dev/nvhost-as-gpu
--device /dev/nvhost-ctrl
--device /dev/nvhost-ctrl-gpu
--device /dev/nvhost-gpu
--device /dev/nvhost-dbg-gpu
--device /dev/nvhost-prof-gpu
--device /dev/nvhost-tsg-gpu
--device /dev/nvhost-ctxsw-gpu
--device /dev/nvhost-sched-gpu
--device /dev/nvhost-nvsched-gpu
--device /dev/nvhost-power-gpu
--device /dev/nvmap
--device /dev/nvidia0
--device /dev/nvidiactl
"

# Tegra 驱动库 (libcuda 依赖链)
TEGRA_LIBS=""
for lib in libcuda.so.1 libnvos.so libnvrm_chip.so libnvrm_gpu.so libnvrm_host1x.so libnvrm_mem.so libnvrm_sync.so libnvsciipc.so libnvsocsys.so libnvtegrahv.so libnvrm_gpusched.so libnvrm_interop_gpu.so libnvrm_stream.so libnvrm_surface.so; do
    [ -f "/usr/lib/$lib" ] && TEGRA_LIBS="$TEGRA_LIBS -v /usr/lib/$lib:/usr/lib/$lib:ro"
done

docker run -d \
    --name $CONTAINER_NAME \
    --network host \
    --restart unless-stopped \
    $GPU_DEVICES \
    $TEGRA_LIBS \
    -v /usr/local/cuda-11.4:/usr/local/cuda:ro \
    -v /usr/lib/aarch64-linux-gnu/libcudnn.so.8:/usr/lib/aarch64-linux-gnu/libcudnn.so.8:ro \
    -v /usr/lib/aarch64-linux-gnu/libcudnn.so.8.3.3:/usr/lib/aarch64-linux-gnu/libcudnn.so.8.3.3:ro \
    -v /opt/m/models:/opt/models:ro \
    -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    $IMAGE_NAME \
    python3 /opt/voice-service/server.py

echo "=== 容器已启动 ==="
docker ps | grep $CONTAINER_NAME
echo "服务地址: http://0.0.0.0:9880"
