import sys, os, time, json
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
SPEEDS = [1.0, 0.85, 0.75]
ROUNDS = 3
T0 = time.time()

print("loading F5TTS (cuda)...", flush=True)
_t = time.time()
f5tts = F5TTS(
    model="F5TTS_v1_Base", ckpt_file=MODEL, vocab_file=VOCAB,
    vocoder_local_path=VOCOS, device="cuda",
    hf_cache_dir="/opt/m0/.cache/huggingface",
)
print("loaded in %.1fs" % (time.time() - _t), flush=True)

# ---- heavy warmup: DVFS/allocator settle makes single warmup unreliable ----
print("warmup x4 ...", flush=True)
for w in range(4):
    _t = time.time()
    f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text="第%d段预热句子，让显卡充分进入稳态。" % w,
                nfe_step=32, speed=1.0, seed=7)
    print("  warm%d %.2fs (t+%.0fs)" % (w, time.time() - _t, time.time() - T0), flush=True)

# {(speed, sent_idx): [dt, dt, ...]}
res = {}
lat = {}
for rnd in range(ROUNDS):
    print("---- ROUND %d ----" % rnd, flush=True)
    # 每轮速度顺序反转一次, 抵消单向漂移(如持续降频)对某一档的系统性偏袒
    order = SPEEDS if rnd % 2 == 0 else list(reversed(SPEEDS))
    for sp in order:
        for i, s in enumerate(SENT, 1):
            _t = time.time()
            wav, sr, _ = f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=s,
                                     nfe_step=32, speed=sp, seed=42)
            dt = time.time() - _t
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav).squeeze()
            sr = sr or 24000
            res.setdefault((sp, i), []).append((dt, len(wav) / sr, wav))
            lat.setdefault(sp, []).append((dt, len(wav) / sr))
            print("  t+%.0fs sp=%.2f s%d dt=%.2fs audio=%.2fs RTF=%.3f"
                  % (time.time() - T0, sp, i, dt, len(wav) / sr, dt / (len(wav) / sr)), flush=True)

print("=" * 62, flush=True)
print("BEST-OF-%d 汇总 (Orin CUDA, nfe_step=32)" % ROUNDS, flush=True)
table = {}
for sp in SPEEDS:
    print("-- speed=%.2f --" % sp, flush=True)
    b_dt = b_du = m_dt = m_du = 0.0
    for i in range(1, len(SENT) + 1):
        trials = res[(sp, i)]
        # 非稳态只可能让结果变慢, 取最快的一次作为接近稳态的估计
        best = min(trials, key=lambda x: x[0])
        med = sorted(t[0] for t in trials)[len(trials) // 2]
        print("   s%d: best dt=%.2fs audio=%.2fs RTF=%.3f  | median dt=%.2fs RTF=%.3f"
              % (i, best[0], best[1], best[0] / best[1], med, med / best[1]), flush=True)
        b_dt += best[0]; b_du += best[1]
        m_dt += med;     m_du += best[1]
        if sp == 0.85 or sp == 0.75:
            sf.write("/opt/m0/f5_clean_sp%03d_s%d.wav" % (int(sp * 100), i), best[2], sr)
    table[sp] = (b_dt, b_du, b_dt / b_du, m_dt / m_du)
    print("   SUB: best-> synth=%.2fs audio=%.2fs RTF=%.3f | median RTF=%.3f"
          % (b_dt, b_du, b_dt / b_du, m_dt / m_du), flush=True)

print("=" * 62, flush=True)
print("结论表 (best-of-%d):" % ROUNDS, flush=True)
base = table[1.0]
for sp in SPEEDS:
    dt, du, rtf, mrtf = table[sp]
    print("  speed=%.2f | 音频总长=%.2fs (吐字 %+.0f%%) | 合成总耗时=%.2fs (%+.0f%%) | RTF=%.3f (中位 %.3f)"
          % (sp, du, (du / base[1] - 1) * 100, dt, (dt / base[0] - 1) * 100, rtf, mrtf), flush=True)
print("DONE", flush=True)
