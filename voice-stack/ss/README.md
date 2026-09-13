# speech-to-speech 部署说明（192.168.8.111）

HuggingFace `speech-to-speech` v1.0.0 容器化部署到 NVIDIA Jetson 工控机上的记录与运维手册。

---

## 1. 目标机环境

| 项 | 值 |
|---|---|
| 地址 | `192.168.8.111`（root / nvidia） |
| 主机名 | `S11L-IPD` |
| 架构 | aarch64（NVIDIA Tegra，Jetson Orin 级） |
| 系统 | Ubuntu 20.04.4 LTS，内核 `5.10.104-rt63-tegra` |
| 平台 | **JetPack 5.1.2 / L4T r35.4.1**，CUDA 11.4，cuDNN 8.6.0 |
| Docker | 20.10.7（`overlay2`，含 `nvidia` runtime，Root Dir `/opt/other/docker`） |
| 分区 | `/` 5G（已满 95%）、`/opt/other` 20G（剩 6.2G）、`/opt/update` 40G（剩 22G） |

## 2. 交付物

| 位置 | 说明 |
|---|---|
| 镜像 `speech-to-speech:v1` | 基于本机已有的 `nvcr.io/nvidia/l4t-jetpack:r35.4.1` |
| 容器 `speech-to-speech` | `--network host`，监听 **8765**，Realtime WS 路径 `/v1/realtime` |
| `/opt/m/ss/ss-docker.sh` | 容器管理脚本（start/stop/restart/status/logs/check/update） |
| `/opt/m/ss/realtime_client.py` | 端到端验收客户端 |
| `/opt/update/ss-docker/` | 构建上下文（Dockerfile、entrypoint、源码快照） |
| `/opt/ss-shim/torchaudio` | 镜像内 torchaudio 兼容层 |

## 3. 常用命令

```bash
bash /opt/m/ss/ss-docker.sh start      # 启动（等就绪，约 1-3 分钟）
bash /opt/m/ss/ss-docker.sh status     # 状态 + 端口 + 健康
bash /opt/m/ss/ss-docker.sh logs       # 跟踪日志
bash /opt/m/ss/ss-docker.sh restart    # 重启
bash /opt/m/ss/ss-docker.sh stop       # 停止

# 端到端验收（在宿主机上跑，用同一个 venv）
/opt/m/ss-venv/bin/python /opt/m/ss/realtime_client.py \
    --wav /opt/m/CosyVoice/asset/zero_shot_prompt.wav \
    --out /opt/update/speech-to-speech/out.wav --wait 60
```

## 4. 运行配置

```
STT  faster-whisper / small / CPU / int8 / 自动语种
LLM  chat-completions -> http://127.0.0.1:8080/v1  (本机 llama-server, Qwen2.5-3B-Q4_K_M)
TTS  kokoro / hexgrad-Kokoro-82M / CPU / 音色 zf_xiaobei
协议 OpenAI Realtime，服务端口 8765
```

可用环境变量覆盖（改 `ss-docker.sh` 的 `-e` 或直接 `docker run`）：

`SS_PORT` `SS_LLM_BASE` `SS_LLM_KEY` `SS_STT_MODEL` `SS_NUM_PIPELINES` `SS_USE_GPU` `SS_SYSTEM_PROMPT`

> **上游 LLM 依赖**：本服务不自带 LLM，复用本机已在跑的 `llama-server`（8080，来自 `voice-service` 容器）。
> 若该服务停止，本服务会报 LLM 连接失败。要指向别处，改 `SS_LLM_BASE`。

## 5. 本次部署踩到的坑（都已解决）

### 5.1 官方 `Dockerfile.arm64` 不能直接用
它基于 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`，要求 **JetPack 6 / L4T r36**。
本机是 JetPack 5，用它构建出的镜像无法使用 GPU。
→ 改用本机已有的 `nvcr.io/nvidia/l4t-jetpack:r35.4.1`（层已在本地，几乎不占额外磁盘）。

### 5.2 宿主机 cuDNN 版本错误 —— 这是必须容器化的根本原因
Jetson 版 `torch 2.2.0` 按 **cuDNN 8.6.0** 编译，而宿主机系统只有 **8.3.3**：

```
RuntimeError: cuDNN version incompatibility:
  PyTorch was compiled against (8, 6, 0) but found runtime version (8, 3, 3)
```

`l4t-jetpack` 镜像里正好是 8.6.0。**原生（非容器）跑任何 CUDA 算子都会炸**，所以之前的裸机部署方案走不通。

### 5.3 torchaudio ABI 不兼容
PyPI 上 aarch64 的 `torchaudio` wheel 与 Jetson 版 torch 符号不匹配：

```
OSError: libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
```

NVIDIA 的 JetPack 5 wheel 源只提供 torch，不提供 torchaudio；设备上编译源码也不现实。
→ 自写了一个轻量 **torchaudio 兼容层**（`/opt/ss-shim/torchaudio`），用设备上已可用的
`soxr` + `soundfile` 实现项目实际用到的接口（`functional.resample`、`transforms.Resample`、
`load/save`、`sox_effects`）。全项目只有 `VAD/vad_handler.py` 和 silero-vad 用到 torchaudio。

### 5.4 缺 `libmpi_cxx.so.40` 等动态库
Jetson 版 `libtorch` 动态依赖 MPI/BLAS。
→ 镜像内 `apt install libopenmpi-dev openmpi-bin libopenblas0-pthread` 解决。

### 5.5 `libgomp` 静态 TLS 冲突
`transformers -> kokoro` 链路会 dlopen scikit-learn 自带的 libgomp，报
`cannot allocate memory in static TLS block`。
→ 入口脚本启动时 `LD_PRELOAD` 该 libgomp。

### 5.6 Kokoro 强制走 CUDA 导致崩溃循环
`--kokoro_device cpu` 会被忽略：`KPipeline` 只要发现 CUDA 可用就切 GPU，在旧 JetPack 上
报 `PTX JIT compiler library not found` 并不断重启。
另外这台机器的 GPU 已被 `llama-server` 占用，house 惯例也提示过 GPU 上下文冲突。
→ 入口脚本默认 `CUDA_VISIBLE_DEVICES=""`；需要 GPU 时用 `-e SS_USE_GPU=1` 打开。

### 5.7 缺失的 Python 依赖
原 venv 少 `addict`（misaki）、`spacy`（misaki.en）、`dlinfo`/`segments`/`csvw`（phonemizer）
→ 已补装进 `/opt/m/ss-venv`。

### 5.8 NLTK 语料被网络拦截
```
[nltk_data] Error loading punkt_tab: Security Violation
[nltk_data]   [pathsec.urlopen]: SSRF attempt to restricted IP 198.18.1.115
```
本机 DNS 走 fake-IP（198.18.0.0/15），nltk 下载被安全组件拦掉。
→ 从 GitHub `nltk/nltk_data` 取 `punkt_tab` 与 `averaged_perceptron_tagger_eng`
手动解压到 `/opt/other/nltk_data/`。

### 5.9 验收客户端事件名过时
项目用的是新版 Realtime 事件名 `response.output_audio.delta` /
`response.output_audio_transcript.delta`，旧客户端只认 `response.audio.delta`，
导致"有音频但报 FAIL"。→ 已修正并兼容新旧两种命名。

### 5.10 `docker exec` 在本机不可用
```
OCI runtime exec failed: read init-p: connection reset by peer
```
docker 20.10.7 与该内核的已知问题，对既有 `voice-service` 容器同样如此。影响：

- `docker exec` 进不去容器 → 改用 `docker run --rm -it --network host -v ... speech-to-speech:v1 shell`，
  或直接在宿主机用 `/opt/m/ss-venv/bin/python`。
- **Dockerfile 里不能定义 `HEALTHCHECK`**（执行健康检查同样走 exec，会永久显示 unhealthy）。

健康判据统一走宿主侧的 `bash /opt/m/ss/ss-docker.sh status`，它直接请求 `/v1/pool`：

```json
{"size":1,"in_use":0,"units":[{"index":0,"state":"idle","session_id":null}]}
```

`/v1/usage` 可以看到累计 tokens、音频时长、错误计数。

## 6. 已知限制

- **CPU 推理**：STT/TTS 都在 CPU 上。热态下从说完话到首包音频约需数秒；语种首次切换会重载
  Kokoro 语言包（更慢）。若需提速可评估 `-e SS_USE_GPU=1`，但需先确认与 `llama-server`
  共存时不会出现 GPU 上下文冲突。
- **单会话**：`SS_NUM_PIPELINES=1`，同一时间只支持 1 路实时会话；并发接入会收到
  `1008 All session slots are in use`。提高并发请调大该变量（CPU 负载同步上升）。
- **磁盘**：`/opt/other` 仅剩 6.2G，`/` 已用 95%。当前方案镜像增量很小（复用本地基础镜像层），
  但后续如需构建新镜像请留意空间。
- **依赖宿主挂载**：venv、模型权重通过卷挂载提供（`/opt/m`、`/opt/update`），
  镜像本身只负责 CUDA/cuDNN 运行时与入口。这是为规避磁盘限制而做的取舍。

## 7. 与既有服务的关系

| 服务 | 端口 | 说明 |
|---|---|---|
| `voice-service`（已有） | 9880 | 自研中文语音服务：SenseVoice ASR + melo TTS + WS |
| `llama-server`（已有） | 8080 | Qwen2.5-3B-Instruct GGUF，**本服务复用为上游 LLM** |
| `speech-to-speech`（本次） | 8765 | HuggingFace 流水线，OpenAI Realtime 协议 |

三者互不冲突：本服务不占 GPU、端口独立。
