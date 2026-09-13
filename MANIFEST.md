# 系统全量归档说明（MANIFEST）

本仓库包含 192.168.8.111（Jetson Orin 车载开发板）的**全量代码/配置归档**。
模型权重文件不在内（按需从 HuggingFace 下载，见 README）。

## 归档构成

| 路径 | 内容 | 大小 |
|---|---|---|
| `comfyui-build/ComfyUI/` | ComfyUI 0.34.0 完整可运行源码树（含本仓库 patches 的落地版、comfy_extras H3 节点、custom_nodes）| 55M |
| `voice-stack/` | 语音助手全栈：f5_ws_server、f5_cached_ref、voice-service、ss、CosyVoice、MeloTTS、torchaudio shim、42 个脚本 | 103M |
| `patches/` | ops.py / model_management 补丁 + torch22_compat_h3ops 桥接 + 启动脚本 | 小 |
| `system-tarballs/` | 系统数据卷分卷归档（见下）| 3.2G |
| `bigfiles/` | 超过 89M 的单体大文件分卷（见下）| 1.2G |

## system-tarballs/ 内容

| 包 | 原始内容 | 大小 |
|---|---|---|
| `sys-extras.tgz` | `/opt/update/{test_py, ai_env, backup, speech-to-speech}` + `/opt/m/tools`（离线 git 工具链+CA）+ `/opt/m0/{gpu(含 JetPack torch2.1 pylibs), link_share, nvlibs}` | 2.0G |
| `sys-configs.tgz` | `/etc/systemd/system`（f5ws/voice-* 等自启单元）、fstab、pip.conf、F5 参考文本 | 12K |

重组：`bash system-tarballs/rejoin-tarballs.sh` 后 `tar xzf` 即可。

## bigfiles/ 内容（>89M 单体文件分卷）

| 文件 | 用途 |
|---|---|
| `opt/m0/gpu/pylibs/torch/lib/libtorch_cpu.so / libtorch_cuda.so` | JetPack torch 2.1.0 运行库 |
| `opt/m0/gpu/whl/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl` | NVIDIA 官方 JetPack torch 2.1 轮子 |

重组：`bash bigfiles/rejoin.sh`（需先 `cd` 到仓库根，按 `bigfiles.list` 还原绝对路径）。

## 排除项（未归档）

- **模型权重**（按需下载）：H3 DiT 20.97G、Qwen3-VL TE 15.7G、双 VAE 5.5G、Turbo LoRA、z-image 4.7G、Qwen2.5-3B、F5/CosyVoice 权重
- **venv**（voice-venv 2.6G / ss-venv 2.1G）——依赖清单见 `voice-stack/`（pip_*.txt 在 m0 归档）
- **Docker 镜像层**（12G）——`voice-stack/voice-docker/Dockerfile` 可重建
- **/usr 系统包**、**/var**、**docker 镜像层**、**车机地图（map 4.1G）**
- 单文件 >89M 的其余二进制已按分卷收录，无遗漏

## 仓库体积

约 3.1G（分卷均为 89M 以内，直接 `git clone` 即可取回全部）。
