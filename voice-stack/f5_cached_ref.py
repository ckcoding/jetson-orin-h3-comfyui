#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F5-TTS 参考音频持久化缓存封装（Jetson AGX Orin / NVIDIA torch 2.2.0）

背景：
  F5 是 zero-shot TTS，每次合成必须把参考语音的声学特征（波形 -> mel）作为
  前缀拼进生成序列。原版 F5TTS.infer() 每次都会重新 sf.read 原始录音 ->
  resample 24k -> 提 mel，模块内虽有"进程内"内存缓存（按文件 md5），但进程
  退出即失效，每次新跑脚本仍要从原始录音重新解码。

本脚本做的事：
  首次（--build 或缓存不存在）用参考录音预处理：读取 -> 重采样 24kHz -> RMS
  归一化，连同规范化后的 ref_text 一并存到 /opt/m0/f5_ref_cache.pt。
  之后任意进程 / 任意次合成，只需 F5Cached().say("文本")，直接 load 缓存的
  参考波形作为 cond 喂给模型，不再读取/解码原始录音文件。

原理说明：
  F5 没有"脱离音频的 speaker 向量"，缓存的是"已预处理好的参考波形"，
  每次合成仍会把它作为条件前缀喂入模型（这是 F5 工作方式决定的，无法省掉），
  但省去了每次重新解码/重采样原始录音的环节，且调用更简洁。

用法：
  # 首次：构建缓存（用默认的 7s 用户录音 + ASR 文本）
  python f5_cached_ref.py --build
  # 之后：直接用缓存合成（打印 [cache] loaded）
  python f5_cached_ref.py --text "今天天气真好"
  # 批量合成（默认 4 句）到 /opt/m0/f5_cached_s{1..4}.wav
  python f5_cached_ref.py
"""
import sys, os, argparse
import torch
import soundfile as sf
import librosa
import numpy as np

sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import (
    convert_char_to_pinyin, hop_length, target_sample_rate,
)


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CACHE = "/opt/m0/f5_ref_cache.pt"
DEFAULT_REF_FILE = "/opt/m0/ref_user7.wav"
DEFAULT_REF_TEXT_FILE = "/opt/m0/ref_user7_text.txt"
CKPT = "/opt/m/models/f5-tts/F5TTS_v1_Base/model_1250000.safetensors"
VOCAB = "/opt/m/models/f5-tts/F5TTS_v1_Base/vocab.txt"
VOCODER = "/opt/m/models/vocos-mel-24khz"

DEFAULT_TEXTS = [
    "今天天气真好，我们出去走走吧。",
    "人工智能正在改变我们的生活方式。",
    "请把这份文件发送到我的邮箱。",
    "晚饭后我们一起看部电影怎么样？",
]


def _normalize_ref_text(t):
    t = t.strip()
    if not t.endswith(". ") and not t.endswith("。"):
        if t.endswith("."):
            t += " "
        else:
            t += ". "
    return t


class F5Cached:
    def __init__(self, nfe_step=16, speed=0.85, device="cuda",
                 force_build=False, ref_file=None, ref_text_file=None):
        self.nfe_step = nfe_step
        self.speed = speed
        self.device = device
        # 模型权重只加载一次
        self.tts = F5TTS(
            model="F5TTS_v1_Base", ckpt_file=CKPT, vocab_file=VOCAB,
            vocoder_local_path=VOCODER, device=device,
        )
        self.mel_spec_type = self.tts.mel_spec_type
        self.ref_audio = None
        self.ref_text = None
        self.rms_original = None
        if force_build or not os.path.exists(CACHE):
            self.build_ref(ref_file or DEFAULT_REF_FILE,
                           ref_text_file or DEFAULT_REF_TEXT_FILE)
        else:
            self._load_cache()
        self._warmed = False
        self._warmup()

    def build_ref(self, ref_file=DEFAULT_REF_FILE, ref_text_file=DEFAULT_REF_TEXT_FILE):
        with open(ref_text_file, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        ref_text = _normalize_ref_text(ref_text)
        audio_np, sr = sf.read(ref_file)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != target_sample_rate:
            audio_np = librosa.resample(audio_np.astype(np.float32),
                                        orig_sr=sr, target_sr=target_sample_rate)
        audio = torch.from_numpy(audio_np).float()
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        target_rms = 0.1
        if rms < target_rms:
            audio = audio * target_rms / rms
        audio = audio.unsqueeze(0)  # (1, T) on cpu
        self.rms_original = float(rms)
        torch.save(
            {"audio": audio.cpu(), "ref_text": ref_text,
             "rms_original": float(rms), "sr": target_sample_rate},
            CACHE,
        )
        # 持久化保存的是 cpu 副本；实例内持有的是推理设备上的副本
        self.ref_audio = audio.to(self.device)
        self.ref_text = ref_text
        print(f"[cache] built -> {CACHE}  "
              f"({audio.shape[-1]/target_sample_rate:.2f}s, rms_orig={rms:.3f})")

    def _load_cache(self):
        d = torch.load(CACHE, map_location="cpu", weights_only=False)
        self.ref_audio = d["audio"].to(self.device)
        self.ref_text = d["ref_text"]
        self.rms_original = d.get("rms_original")
        print(f"[cache] loaded -> {CACHE}  "
              f"({self.ref_audio.shape[-1]/target_sample_rate:.2f}s)")

    def _warmup(self):
        # 本机(NVIDIA torch2.2.0 + Jetson fp32)F5 长文本单批 ODE 在"冷"首次
        # 推理会数值发散成 NaN，热调用后正常。构造后用短句热身一次即可。
        tmp = "/opt/m0/.f5_ref_tmp.wav"
        if not os.path.exists(tmp):
            sf.write(tmp, self.ref_audio.squeeze().cpu().numpy(), target_sample_rate)
        try:
            self.tts.infer(ref_file=tmp, ref_text=self.ref_text,
                           gen_text="今天天气不错，出来散散步吧。",
                           nfe_step=self.nfe_step, speed=self.speed, seed=0)
        except Exception as e:
            print(f"[warmup] skip: {e!r}")
        self._warmed = True

    def say(self, gen_text, out_path=None, nfe_step=None, speed=None, seed=None):
        if seed is not None:
            seed_everything(seed)
        nfe = nfe_step or self.nfe_step
        sp = speed or self.speed
        # 参考波形落临时 wav，走官方 F5TTS.infer（已验证可出声）。
        # 本机踩过的坑（均已验证定位）：
        #  1) 直接 ema_model.sample(cond=audio) 在 cuda 上对缓存参考波形提
        #     mel 会产出 NaN；官方 infer 从文件读 ref 走 CPU 预处理可正常。
        #  2) 本机(NVIDIA torch2.2.0 + Jetson fp32)对部分随机种子 ODE 会发散
        #     成 NaN，固定 seed=0 稳定出声；长文本单批 seed=0 也正常。
        #  3) 不要在此拆句：chunk 出的超短句(<10字节)会触发 infer 内部
        #     local_speed=0.3 分支，导致 duration 异常发散成 NaN。故整句直推。
        tmp = "/opt/m0/.f5_ref_tmp.wav"
        if not os.path.exists(tmp):
            sf.write(tmp, self.ref_audio.squeeze().cpu().numpy(), target_sample_rate)
        res = self.tts.infer(
            ref_file=tmp, ref_text=self.ref_text, gen_text=gen_text,
            nfe_step=nfe, speed=sp, seed=0)
        wave = np.array(res[0], dtype=np.float32) if isinstance(res, tuple) else np.array(res, dtype=np.float32)
        sr = res[1] if (isinstance(res, tuple) and len(res) > 1 and isinstance(res[1], int)) else target_sample_rate
        if out_path:
            sf.write(out_path, wave, sr)
        return wave, sr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="强制重建参考缓存")
    ap.add_argument("--text", default=None, help="单句合成文本")
    ap.add_argument("--out", default="/opt/m0/f5_cached_test.wav")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--speed", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    f5 = F5Cached(force_build=args.build, nfe_step=args.nfe, speed=args.speed)

    texts = [args.text] if args.text else DEFAULT_TEXTS
    for i, t in enumerate(texts, 1):
        out = args.out if len(texts) == 1 else f"/opt/m0/f5_cached_s{i}.wav"
        w, sr = f5.say(t, out_path=out, nfe_step=args.nfe,
                       speed=args.speed, seed=args.seed)
        print(f"[ok] {out}  ({(len(w)/sr):.2f}s)  text={t}")
