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
# duration = ... / speed  -> speed<1 = 语速变慢(音频更长); speed>1 = 语速变快
SPEEDS = [1.0, 0.85, 0.75]

print("loading F5TTS (cuda)...", flush=True)
t0 = time.time()
f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=MODEL,
    vocab_file=VOCAB,
    vocoder_local_path=VOCOS,
    device="cuda",
    hf_cache_dir="/opt/m0/.cache/huggingface",
)
print("loaded in %.1fs" % (time.time() - t0), flush=True)

# warmup (discarded): first CUDA inference collapses to silence without it
print("warmup...", flush=True)
_ = f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text="这是一段用来预热的测试句子。",
                nfe_step=32, speed=1.0, seed=7)
print("warmup done", flush=True)

summary = []
for sp in SPEEDS:
    tag = "sp%03d" % int(sp * 100)
    print("=" * 58, flush=True)
    print("SPEED = %.2f" % sp, flush=True)
    tot_dt = tot_dur = 0.0
    for i, s in enumerate(SENT, 1):
        t0 = time.time()
        wav, sr, _ = f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=s,
                                 nfe_step=32, speed=sp, seed=42)
        dt = time.time() - t0
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav).squeeze()
        sr = sr or 24000
        dur = len(wav) / sr
        rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
        tot_dt += dt
        tot_dur += dur
        print("  s%d audio=%.2fs synth=%.2fs RTF=%.3f rms=%.4f %s"
              % (i, dur, dt, dt / dur, rms, "OK" if rms > 0.001 else "SILENT!"), flush=True)
        sf.write("/opt/m0/f5_%s_s%d.wav" % (tag, i), wav, sr)
    avg = tot_dt / tot_dur
    summary.append((sp, tot_dt, tot_dur, avg))
    print("  SUBTOTAL synth=%.2fs audio=%.2fs avgRTF=%.3f" % (tot_dt, tot_dur, avg), flush=True)

print("=" * 58, flush=True)
print("SPEED SWEEP SUMMARY (nfe_step=32, Orin CUDA)", flush=True)
base = summary[0]
base_dt, base_dur = base[1], base[2]
for sp, dt, dur, avg in summary:
    tag = " (baseline)" if sp == 1.0 else ""
    # 音频更长 == 语速更慢，用总音频时长反向表示说话快慢
    slower = (dur / base_dur - 1.0) * 100.0   # 音频变长百分比 = 吐字变慢百分比
    cost = (dt / base_dt - 1.0) * 100.0       # 合成耗时变化
    print("  speed=%.2f | 总音频=%.2fs (语速慢%+.0f%%) | 总合成=%.2fs (%+.0f%%) | 平均RTF=%.3f%s"
          % (sp, dur, slower, dt, cost, avg, tag), flush=True)
print("DONE", flush=True)
