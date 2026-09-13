# 在 Jetson Orin 车载开发板上运行 MiniMax H3（音画同步视频生成）

> 在 NVIDIA Jetson Orin（24GB 统一内存 / aarch64）+ TTTech MotionWise 车载安全平台上，
> 让 ComfyUI 完整跑通 **MiniMax H3 文本→视频+同步音频** 生成。
> 没有官方 ARM64 支持、没有可用 CUDA 扩展、没有 flash attention——全部用软件层补齐。

![status](https://img.shields.io/badge/status-verified_working-brightgreen) ![board](https://img.shields.io/badge/board-Jetson_Orin_24GB-76b900) ![stack](https://img.shields.io/badge/stack-ComfyUI_0.34.0_/_torch_2.2.0_aarch64-blue)

---

## 1. 硬件与系统背景

| 项 | 值 |
|---|---|
| 板卡 | Jetson Orin（p3663-0001），11 核 Cortex-A78AE，**24GB 统一内存** |
| 平台 | TTTech MotionWise 车载安全底座（`tegra_virt_storage*` 虚拟盘供盘） |
| 系统 | Ubuntu 20.04 基座，L4T r35.4.1，内核 5.10.104-rt63-tegra（**PREEMPT RT**） |
| Python | 3.10.20（venv 定制） |
| PyTorch | **2.2.0 定制 aarch64 构建**（无 flash attention / 无 memory-efficient SDPA） |
| ComfyUI | 0.34.0（含 `comfy_kitchen` 量化运行时、`comfy_quant` 布局） |
| 外网 | **无**（全离线环境，模型/轮子全靠 LAN 传输） |

## 2. 要解决的五座大山

| # | 问题 | 根因 | 后果 |
|---|---|---|---|
| 1 | `import torchaudio` 即崩 | 官方 torchaudio 2.2.0 轮子的 C++ 扩展与定制 torch 2.2.0 **ABI 不兼容**（`undefined symbol: torch::autograd::Node::name()`），抛 `OSError` 穿透上游 try/except | 整个 ComfyUI 启动即死 |
| 2 | `torch.ops.comfy_kitchen.*` 全部缺失 | 设备上的 comfy_kitchen CUDA 扩展只编译了 2 个算子（int8_linear、rms_rope），而 H3 的 int8/nvfp4/awq 前向需要 ~50 个 | 量化模型前向 AttributeError |
| 3 | nvfp4 Linear dtype 失配 | 上层包装传 dtype 整数 code，eager 实现期望 `torch.dtype`；fp32 激活 × bf16 权重 | TE 编码崩溃 |
| 4 | TE 加载即 OOM | ComfyUI cast 路径对 15.7G nvfp4 文本编码器做**全量反量化**（膨胀 ~4 倍 ≈ 60G） | 内核 OOM-kill |
| 5 | 采样第一步即 OOM | torch 2.2 aarch64 **没编译 flash/mem-efficient attention** → math 模式物化整张注意力矩阵（40 头 × 15k² ≈ 25GB/层） | 内核 OOM-kill |

## 3. 解法总览（六层补丁栈）

```
┌─ L6  启动参数: --disable-pinned-memory（权重页可回收）
├─ L5  model_management: AIMDO cast 缓冲预订 16G → 2G
├─ L4  ops.py: SDPA 分块化（查询轴分块 + fp32 精确 softmax，数学等价）
├─ L3  ops.py: QuantizedTensor 免全量反量化（透传给 layout 算子按需反量化）
├─ L2  torch22_compat_h3ops.py: 49 个 comfy_kitchen 算子 eager 纯 torch 桥接
│      + dtype-code 适配 + nvfp4-linear dtype 均一化
└─ L1  torchaudio-shim: 纯 torch/torch+scipy 重实现 torchaudio API 子集
```

### 3.1 torchaudio 纯软件垫片（`torchaudio-shim/`）

卸载 ABI 不兼容的官方轮子，按 ComfyUI 实际用到的 API 面重实现：

- `functional.resample` —— 窗口化 sinc 重采样（支持 `sinc_interp_hann` / `sinc_interp_kaiser`，含 `lowpass_filter_width`/`rolloff`/`beta`）
- `functional.bass_biquad / treble_biquad / equalizer_biquad` —— RBJ 均衡器（scipy.signal.lfilter）
- `transforms.MelSpectrogram` —— torch.stft + slaney/htk mel 滤波器组（含 slaney 归一化），nn.Module 带 buffer 支持 `.to(device)`
- `load / save / info` —— soundfile 后端
- 附带 dist-info 让 `importlib.metadata.version("torchaudio")` 返回 `2.2.0`

### 3.2 comfy_kitchen 全量 eager 桥接（`patches/torch22_compat_h3ops.py`）

把 `comfy_kitchen.backends.eager` 里的**纯 Python 实现**按原名注册到 `torch.ops.comfy_kitchen.*` 命名空间（覆盖既有算子的不碰）。关键细节：

- nvfp4/mxfp8/fp8 家族：上层包装传**整数 dtype code**（0=f32 1=f16 2=bf16 5=e4m3fn 6=e5m2），eager 实现要 `torch.dtype` → 插入转换适配器
- `quantize_fp8/dequantize_fp8` 在 `__all__` 里没有同名函数 → 映射到 `quantize_per_tensor_fp8/dequantize_per_tensor_fp8`
- 闭包晚绑定陷阱：适配器一律用默认参数固化 `_orig`（踩过一次 `dequantize_nvfp4` 调到 fp8 实现的坑）
- 顺手修 nvfp4-linear 慢路径的 fp32×bf16 失配（dispatch 表劫持，反量化后以权重 dtype 为锚统一）

### 3.3 ops.py 三处手术（`patches/ops.py.H3-all.patch`）

1. **量化权重透传**：`to_dequant()` 遇到 `QuantizedTensor` 直接返回，交给 layout 算子按算子级反量化（nvfp4 embedding 只 gather 需要的行）——治全量反量化 OOM
2. **分块 SDPA**：`_h3_chunked_sdpa()` 按 query 轴分块（每块分数矩阵预算 300MB），**数学上与全注意力严格等价**（softmax 只沿 key 轴，行间独立）；matmul 与 softmax 全程 fp32 保证精度
3. legacy CPU 路径同样透传 QTensor

### 3.4 model_management 一行（`patches/model_management.py.H3-aimdo.patch`）

`DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE`: 16G → 2G。24G 统一内存的机器预订 16G 的 cast 缓冲必然失败。

### 3.5 启动参数（`patches/start-comfyui.sh`）

```bash
python main.py --listen 0.0.0.0 --port 8188 \
    --enable-dynamic-vram --vram-headroom 1 --cache-lru 1 \
    --disable-pinned-memory        # 关键：staged 权重页保持可回收
```

## 4. 模型与磁盘布局（bind mount 方案）

125G 物理盘切了 8 块虚拟盘（`tegra_virt_storage*`），单盘装不下 ComfyUI 全家（≈59G），按"程序/模型/超大件"三盘分工 + **bind mount**（硬链接跨文件系统不可行，软链又怕断，bind 是内核级目录视图，应用看到的就是真文件）：

```
/opt/m/ComfyUI/                 ← 程序盘 vblkdev50 (30G)
   models/vae               ── bind ── /opt/m0/comfyui-data/vae        (5.5G)
   models/loras             ── bind ── /opt/m0/comfyui-data/loras      (2.6G)
   models/diffusion_models  ── bind ── /opt/update/diffusion_models   (20G)
   models/text_encoders/    ── qwen3vl H3 TE 软链到 /opt/update（8G qwen_3_4b 留本盘）
   output                   ── bind ── /opt/m0/comfyui-data/output
```

`/etc/fstab` 持久化：

```
/opt/m0/comfyui-data/vae                /opt/m/ComfyUI/models/vae               none bind 0 0
/opt/m0/comfyui-data/loras              /opt/m/ComfyUI/models/loras             none bind 0 0
/opt/m0/comfyui-data/output             /opt/m/ComfyUI/output                   none bind 0 0
/opt/update/diffusion_models            /opt/m/ComfyUI/models/diffusion_models  none bind 0 0
```

| 模型 | 大小 | 来源 |
|---|---|---|
| minimax_h3_fl2va_pruned_int8_convrot.safetensors | 20.97G | Comfy-Org/MiniMax-H3 |
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | 15.69G | HF（nvfp4+awq 混合量化 32B） |
| minimax_h3_video_vae_fp16 / audio_vae_fp32 | 4.9G / 578M | HF |
| larryvrh v4_step600_ema（Turbo LoRA）| 744M | larryvrh/MiniMax-H3-Turbo-Lora |
| lightx2v FL2VA 4-step v1.2 768p | 1.96G | lightx2v/Minimax-h3-Turbo |

## 5. 工作流（`workflows/`，API 格式，POST 到 `/prompt` 直接用）

```
UNETLoader(H3 int8) → MiniMaxH3TurboLoRA ──┐
CLIPLoader(type="minimax") ─┐              ├─ BasicGuider ─┐
VAELoader(video) ───────────┤ ImageToVideo  │               ├ SamplerCustomAdvanced
VAELoader(audio) ───────────┘ (prompt,W,H,L)│ BasicScheduler ┘
RandomNoise ────────────────────────────────┘     ↓
                              VAEDecodeTiled + VAEDecodeAudio → CreateVideo(24fps) → SaveVideo
```

- **Turbo LoRA 两种模式**：`low_vram=false`（bypass，运行时注入，锐利）✅ 推荐；`true`（merge 进 int8 权重，delta 被舍入 → 明显变糊）❌
- lightx2v 768p 系 LoRA 必须配 `MiniMaxH3SigmaShift(shift_video=6, shift_audio=3)`（官方 544p 基模默认 12/3）
- 帧数对齐 17k+5 网格；训练区间 124–362 帧（5–15s），39 帧以下出片质量崩

## 6. 实测性能（Orin 24G，本仓库补丁栈）

| 配置 | 分辨率/时长 | 步数 | 耗时 |
|---|---|---|---|
| H3 + larryvrh v4 + bypass | 832×480 / 2.3s | 6 | **18 min** |
| H3 + larryvrh v4 + bypass | 832×480 / 5.2s | 6 | **63 min** |
| H3 + lightx2v v1.2 + shift6/3 | 1008×576 / 5.2s | 4 | **94 min** |
| （对照）Z-Image-Turbo Q4_K_M 文生图 | 1024² / 8步 | — | 89.8 s |

瓶颈：无 flash attention → 分块 fp32 注意力（精度换速度的取舍）；统一内存带宽 ~100GB/s 级。

## 7. 复现步骤

```bash
# 1) 垫片与桥接
cp -r torchaudio-shim/torchaudio torchaudio-shim/torchaudio-2.2.0.dist-info \
      $VENV/lib/python3.10/site-packages/       # 先 pip uninstall torchaudio
cp patches/torch22_compat_h3ops.py  $VENV/lib/python3.10/site-packages/
cp patches/torch22_compat_h3ops.pth $VENV/lib/python3.10/site-packages/

# 2) ComfyUI 核心补丁（先备份）
cd /path/to/ComfyUI && cp comfy/ops.py comfy/ops.py.bak
patch -p0 < patches/ops.py.H3-all.patch          # 或手工套用三处修改
cp comfy/model_management.py comfy/model_management.py.bak
patch -p0 < patches/model_management.py.H3-aimdo.patch

# 3) 启动（含关键参数）
bash patches/start-comfyui.sh

# 4) 验证：50 个 kitchen 算子就位 + 出图/出视频
python -c "import torch; print(len([n for n in dir(torch.ops.comfy_kitchen) if not n.startswith('_')]))"
curl -s localhost:8188/object_info/UNETLoader | head -c 200
# 5) 提交工作流
curl -X POST localhost:8188/prompt -H 'Content-Type: application/json' \
     -d '{"prompt": <workflows/h3_t2va_480p_56f_api.json>, "client_id": "test"}'
```

## 8. 已知限制

- 采样速度受 fp32 分块注意力支配（611 s/step @ 124f）；换支持 flash attention 的 torch 可提速一个量级，但 JetPack 5 / CUDA 11.4 上没有现成轮子
- 1344×768 全原生分辨率 + 5s ≈ 数小时/条（全注意力 FLOPs 随 token 平方增长），1008×576 是实用甜点
- `low_vram=true`（merge 进 int8 权重）会因舍入明显降质，24G 机器 bypass 模式内存也够，别用
- comfy_kitchen eager 纯 Python 实现性能弱于 CUDA 扩展（本设备 CUDA 扩展只能编 2 个算子，CUDA 11.4 编不了 nvfp4/fp8 系）
- 音频质量与快速运动画面是 H3 Turbo 蒸馏的已知短板（上游问题）

## 9. 延伸：Ollama JetPack5 构建运行 27B LLM（GPU 全层 offload）

同一台设备上用 Ollama v0.34.0 官方 `arm64-jetpack5` 构建（CUDA 11.4 / sm_87），
**66/66 层全量 offload** 运行 Qwen3.8-27B Q4_K_M，实测 **7.58 tok/s**（132 ms/tok）。
关键钥匙：`JETSON_JETPACK=5` 环境变量 + 手工 manifest 绕过 quantize 校验。
详见 **[OLLAMA-DEPLOY.md](OLLAMA-DEPLOY.md)**。

## 10. 致谢与参考

- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) — 官方量化模型
- [larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo) — Turbo LoRA 节点与权重
- [ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) · [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) — lightx2v 加速 LoRA
- [comfy-org/comfy-kitchen](https://github.com/comfy-org/comfy-kitchen) — 量化算子库（本仓库桥接其 eager 后端）
- [MATLOWAI/minimax-h3-fused-turbo-int8-convrot](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot) — 融合底模思路参考

## License

代码按 Apache-2.0；文档 CC-BY-4.0。模型权重遵循 MiniMax-H3 社区许可。
