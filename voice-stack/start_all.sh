#!/bin/bash
# 开机初始化脚本：remount exec + 启动 LLM 服务
# 用法: bash /opt/m/start_all.sh

echo "=== 1. remount exec ==="
mount -o remount,exec /opt/m 2>/dev/null && echo "OK: /opt/m 已可执行" || echo "FAIL: /opt/m"
mount -o remount,exec /opt/update 2>/dev/null && echo "OK: /opt/update 已可执行" || echo "SKIP: /opt/update"

echo ""
echo "=== 2. CA 证书（TLS 修复） ==="
if [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
    cp /opt/other/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt 2>/dev/null \
        && echo "OK: CA 证书已恢复" || echo "WARN: CA 证书恢复失败"
else
    echo "CA 证书正常"
fi

echo ""
echo "=== 3. 启动 Docker ==="
modprobe dm_mod 2>/dev/null
systemctl is-active docker >/dev/null 2>&1 && echo "Docker 已在运行" || {
    systemctl reset-failed docker 2>/dev/null
    systemctl start docker && sleep 3
    systemctl is-active docker >/dev/null 2>&1 && echo "OK: Docker 已启动" || echo "FAIL: Docker"
}

echo ""
echo "=== 4. 检查 llama-server ==="
if ss -lntp | grep -q 8080; then
    echo "llama-server 已在运行"
else
    echo "启动 llama-server..."
    bash /opt/m/start_llm3b.sh
    sleep 8
    ss -lntp | grep -q 8080 && echo "OK: llama-server 已启动 (端口8080)" || echo "FAIL: llama-server 启动失败, 查看 /tmp/llama3b.log"
fi

echo ""
echo "=== 完成 ==="
echo "LLM API:  http://192.168.8.111:8080/v1/chat/completions"
docker info 2>/dev/null | grep -q "Storage Driver" && echo "Docker:   运行中 ($(docker info 2>/dev/null | grep 'Storage Driver' | awk '{print $3}'))"
