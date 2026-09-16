# Ollama (JetPack 5 专用构建) 在 Jetson Orin 上运行 Qwen3.8-27B — 部署实录

> 目标：在 TTTech MotionWise 车载安全平台（只读根、noexec 挂载、无编译器、无外网）上，
> 以 **GPU 全层 offload** 运行 27B 级 LLM。
> 实测：Qwen3.8-27B-Uncensored Q4_K_M，**66/66 层进 Orin iGPU，7.58 tok/s**。

## 0. 为什么是 Ollama jetpack5 构建

设备自带的 llama.cpp（0.1.2-dev 定制版）**架构级不支持** Qwen3.8 的混合层
（Gated Delta Net 线性注意力，初始化即挂死，-ngl 0 纯 CPU 也一样）。
Ollama v0.34.0 起官方提供 `ollama-linux-arm64-jetpack5.tar.zst`，
内含针对 **JetPack 5 / CUDA 11.4 / sm_87 (Orin)** 编译的 `libggml-cuda.so`。

⚠️ 注意资产结构：jetpack5 包**只含 CUDA 附加层**（`lib/ollama/cuda_jetpack5/`），
不含二进制。必须先装基础包 `ollama-linux-arm64.tar.zst`，再叠加 jetpack5 层。

## 1. 安装（双包按序叠加）

```bash
# 顺序不能反: 先 base 后 jetpack5 (bin/ollama 需要能感知 cuda_jetpack5 目录)
mkdir -p /opt/update/ollama && cd /opt/update/ollama
tar zxf ollama-linux-arm64.tar.zst          # base: bin/ollama + lib/ollama/{cuda_v12,cuda_v13}
tar xf ollama-linux-arm64-jetpack5.tar.zst  # addon: lib/ollama/cuda_jetpack5/  (覆盖叠加)
chmod +x bin/ollama lib/ollama/cuda_jetpack5/*
```

## 2. 平台特有的坑（逐个排掉）

| 坑 | 现象 | 解法 |
|---|---|---|
| `/opt/update` 挂载 **noexec** | `./bin/ollama` → Permission denied（尽管 rwx） | `mount -o remount,exec /opt/update` |
| **只读根** `/` | 启动即死：`mkdir /root/.ollama: read-only file system`（生成 ed25519 密钥） | `export HOME=/opt/update/ollama/home`（重定向密钥/状态目录） |
| GPU 发现死活不认 | 反复探 `cuda_v12/v13` 后回落 CPU | **`export JETSON_JETPACK=5`**（日志明示 "set JETSON_JETPACK to override"） |
| 驱动库路径 | ggml-cuda 加载 libcuda 失败 | `export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH`（Jetson 驱动 libcuda/NVML 所在） |
| 模型目录 | 默认 `~/.ollama`（只读盘） | `export OLLAMA_MODELS=/opt/update/ollama/models` |

## 3. 启动

```bash
#!/bin/bash
# start-ollama.sh
export HOME=/opt/update/ollama/home
export OLLAMA_MODELS=/opt/update/ollama/models
export OLLAMA_HOST=0.0.0.0:8080
export OLLAMA_DEBUG=1
export JETSON_JETPACK=5
export OLLAMA_LIBRARY_PATH=/opt/update/ollama/lib/ollama:/opt/update/ollama/lib/ollama/cuda_jetpack5
export LD_LIBRARY_PATH=/opt/update/ollama/lib/ollama/cuda_jetpack5:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH
exec /opt/update/ollama/bin/ollama serve
```

验证 GPU 识别（DEBUG 日志必须出现这一行）：

```
inference compute id=0 library=CUDA compute=8.7 name=CUDA0 description=Orin
    libdirs=ollama,cuda_jetpack5 driver=11.4 type=iGPU total="24.3 GiB"
```

## 4. 导入 GGUF（绕过 create 的 quantize 校验）

社区量化 GGUF（如 `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`）会被
`ollama create` 的 llama-quantize 校验拒绝（"without compatibility patches"）。
该 GGUF 的 blob 其实已被 create 拷进 `models/blobs/sha256-<模型哈希>`，
只需**手工构造 manifest 注册**（零拷贝）：

```bash
MODEL_HASH=sha256-4c5e2db039e9325ac7724c8846c71356a24ad1cdfa28002d73ecb6be645f9675  # = GGUF 文件 sha256
CFG_CONTENT='{"model_format":"gguf","model_family":"qwen35","model_type":"27B","file_type":"Q4_K_M","architecture":"arm64","os":"linux","rootfs":{"type":"layers","diff_ids":["'"$MODEL_HASH"'"]},"config":{}}'
echo -n "$CFG_CONTENT" > /tmp/cfg.json
CFG_HASH=sha256-$(sha256sum /tmp/cfg.json | cut -d' ' -f1)
CFG_SIZE=$(stat -c %s /tmp/cfg.json)
cp /tmp/cfg.json $OLLAMA_MODELS/blobs/$CFG_HASH

mkdir -p $OLLAMA_MODELS/manifests/registry.ollama.ai/library/qwen38
printf '{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"%s","size":%s},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"%s","size":16810714528}]}' \
  "$CFG_HASH" "$CFG_SIZE" "$MODEL_HASH" \
  > $OLLAMA_MODELS/manifests/registry.ollama.ai/library/qwen38/latest

ollama cp qwen38 qwen3.8-27b   # 建议重命名对齐真实型号
ollama list   # → qwen3.8-27b:latest  16 GB
```

## 5. 实测性能（Orin iGPU，66/66 层 CUDA offload）

| 项 | 值 |
|---|---|
| 模型缓冲 | CUDA0 14.7 GiB + KV 2 GiB（ctx 2048），host RSS 仅 1.85 GB |
| 生成速度 | **7.58 tok/s**（132 ms/tok，三轮方差为零） |
| 提示词处理 | ~250 tok/s |
| 端到端 | 400 tok 回复 ≈ 53 s |
| OpenAI 兼容 | `/v1/chat/completions` ✓（Qwen3.8 带思考链，响应含 `reasoning` 字段） |

## 6. 已知限制

- CPU 模式（-ngl 0）在本 llama.cpp 旧版可用但仅 ~1-2 tok/s，不推荐
- 24G 统一内存：模型 14.7G + KV 2G + 系统 ~4G ≈ 贴边，ctx 别开太大；
  与 ComfyUI 同时常驻时注意内存余量
- llama.cpp 定制旧版（0.1.2-dev）对 Qwen3.8 混合架构支持不完整，勿混用


## 7. 延伸：Qwen3.6-35B-A3B (MoE) 的注册

unsloth 官方 GGUF（UD-IQ4_XS 17.73G）实测 **31.73 tok/s**（66 层 offload，A3B 架构
每 token 仅读 ~3G 激活权重，突破密集模型带宽墙 4.2 倍）。完整数据见 BENCHMARKS.md。

要点：
- GGUF 头解析：`general.architecture = qwen35moe`（733 tensors）
- create 的 quantize 校验同样拒绝 → 手工注册时 `model_family` 必须填 **qwen35moe**
- blob 用**硬链接**复用已校验文件（同分区零拷贝）：

```bash
ln /opt/update/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf    $OLLAMA_MODELS/blobs/sha256-649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3
# config blob: model_family=qwen35moe, manifest 同第 4 节 (layer size=17730509792)
```

- 下载务必带校验：`aria2c --checksum=sha-256=<HF API tree 接口的 lfs.oid>`，
  否则可能得到**全零空壳**（大小正确、内容全零，双端 sha256 互验无法发现）
- 24G 内存互斥：qwen36(17.7G) 与 z-image 生成峰值(~12G) 不可同时驻留，
  Ollama 5 分钟闲置自动卸载


## 8. 持久化（systemd 常驻 + 模型永不卸载）

`patches/ollama.service` → `/etc/systemd/system/ollama.service`：

- `OLLAMA_KEEP_ALIVE=-1`：模型加载后**常驻显存不卸载**
- `Restart=always`：崩溃 5 秒自动拉起
- `ExecStartPre` 自带 noexec remount
- `systemctl enable --now ollama` 开机自启

⚠️ 常驻代价：模型载入后占据 ~17G，z-image 生成前需先让位
（`curl -d '{"model":"qwen3.8-27b","keep_alive":0}'` 或 `systemctl restart ollama`）。


## 9. ComfyUI 同样 systemd 托管（OOM 自愈）

35B 模型载入时曾 OOM-kill 掉裸跑的 ComfyUI（无自愈机制）。
`patches/comfyui.service`：`Restart=always` + `OOMScoreAdjust=-500`（OOM 时优先保护），
`systemctl enable --now comfyui` 后被杀 10 秒自动复活。


## 10. 激活视觉（看图）能力

Qwen3.6/3.8 官方均为多模态模型，但 GGUF 只含语言部分——需要额外的
**mmproj 视觉投影器**（独立 GGUF，约 0.9G）注册为 `projector` 层：

```bash
# mmproj 传入服务器后:
MH=sha256-$(sha256sum mmproj-F16.gguf | cut -d' ' -f1)
ln mmproj-F16.gguf $OLLAMA_MODELS/blobs/$MH
# manifest.layers 追加:
# {"mediaType":"application/vnd.ollama.image.projector","digest":"$MH","size":<文件字节数>}
```

调用（消息里带 base64 图片）:
```json
{"model":"qwen3.8-27b","messages":[{"role":"user","content":"描述这张图","images":["<base64>"]}]}
```

⚠️ **mmproj 必须与文本模型配对**（视觉投影输出维度 = LLM hidden size）：
- Qwen3.8-27B（hidden 5120）→ unsloth/Qwen3.8-27B-GGUF 的 mmproj-F16 ✓
- Qwen3.6-35B-A3B（hidden 2048）→ unsloth 仓库的 mmproj **是错配的**（5120，报
  "mismatch n_embd 2048 vs 5120"）——用 **huihui-ai MTP 版仓库的 mmproj-model-f16** ✓
- 错配 mmproj 会导致模型加载失败（连带文本能力一起 404），此时移除 projector 层即恢复

实测：两台模型均正确识别 z-image 生成的纸鹤图（自产自检闭环 ✓）。


## 11. 平台日志降噪（GMSL 刷屏过滤）

车载平台的 GMSL 相机链路驱动在未接相机时每秒刷 2 行错误（日增 ~70MB 日志），
会淹没真实错误。驱动本身属安全平台组件**不可停**，用 rsyslog 层过滤：

```bash
# /etc/rsyslog.d/10-drop-gmsl.conf
:msg, contains, 'GMSL Link' stop
:msg, contains, 'max_gmsl_dp_ser' stop
```
`systemctl restart rsyslog` 后 kern.log 增长归零（驱动照常运行）。
