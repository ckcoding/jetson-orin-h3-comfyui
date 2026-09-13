import sys, os, time
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["HF_HUB_CACHE"] = "/opt/m0/.cache/huggingface"
os.environ["HF_HOME"] = "/opt/m0/.cache/huggingface"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim as _shim
sys.modules["torchaudio"] = _shim
from f5_tts.api import F5TTS
import soundfile as sf, torch, numpy as np

MODEL = "/opt/m0/models/f5-tts/F5TTS_v1_Base/model_1250000.safetensors"
VOCAB = "/opt/m0/models/f5-tts/F5TTS_v1_Base/vocab.txt"
VOCOS = "/opt/m0/models/vocos-mel-24khz"
REF = "/tmp/bronya_s1.wav"
REF_TEXT = "你好，这是语音合成测试，请听一听中文音色是否自然。"
SENT = [
    "F5 语音合成模型，正在英伟达边缘计算设备上流畅运行。",
    "今天天气真不错，我们一起出去走走吧。",
    "人工智能正在悄悄改变我们的生活方式。",
    "床前明月光，疑是地上霜，举头望明月，低头思故乡。",
]
SPEED = 0.85
NFES = [32, 16, 8]
T0 = time.time()

print("loading F5TTS (cuda)...", flush=True)
_t = time.time()
f5tts = F5TTS(
    model="F5TTS_v1_Base", ckpt_file=MODEL, vocab_file=VOCAB,
    vocoder_local_path=VOCOS, device="cuda",
    hf_cache_dir="/opt/m0/.cache/huggingface",
)
print("loaded in %.1fs" % (time.time() - _t), flush=True)

# 4 次预热 (DVFS/allocator 稳态)
print("warmup x4 ...", flush=True)
for w in range(4):
    _t = time.time()
    f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text="第%d段预热句子，让显卡进入稳态。" % w,
                nfe_step=16, speed=SPEED, seed=7)
    print("  warm%d %.2fs (t+%.0fs)" % (w, time.time() - _t, time.time() - T0), flush=True)

summary = {}
for nfe in NFES:
    print("=" * 58, flush=True)
    print("NFE_STEP = %d  (speed=%.2f)" % (nfe, SPEED), flush=True)
    tot_dt = tot_dur = 0.0
    for i, s in enumerate(SENT, 1):
        _t = time.time()
        wav, sr, _ = f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=s,
                                  nfe_step=nfe, speed=SPEED, seed=42)
        dt = time.time() - _t
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav).squeeze()
        sr = sr or 24000
        dur = len(wav) / sr
        rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
        tot_dt += dt
        tot_dur += dur
        tag = "nfe%02d_sp085" % nfe
        sf.write("/opt/m0/f5_%s_s%d.wav" % (tag, i), wav, sr)
        print("  s%d audio=%.2fs synth=%.2fs RTF=%.3f rms=%.4f %s"
              % (i, dur, dt, dt / dur, rms, "OK" if rms > 0.001 else "SILENT!"), flush=True)
    avg = tot_dt / tot_dur
    summary[nfe] = (dt, tot_dt, tot_dur, avg)
    print("  SUBTOTAL synth=%.2fs audio=%.2fs avgRTF=%.3f" % (tot_dt, tot_dur, avg), flush=True)

print("=" * 58, flush=True)
print("NFE SWEEP SUMMARY (speed=%.2f, Orin CUDA)" % SPEED, flush=True)
for nfe in NFES:
    _, dt, dur, avg = summary[nfe]
    print("  nfe=%-3d | 合成总耗时=%.2fs | 音频总长=%.2fs | 平均RTF=%.3f %s"
          % (nfe, dt, dur, avg, "<-- 实时!" if avg < 1.0 else ""), flush=True)
print("DONE", flush=True)
