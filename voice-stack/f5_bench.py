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
import soundfile as sf, torch

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
# warmup call (discarded) -- first CUDA inference on this model tends to
# collapse to silence; warming up avoids contaminating the measured samples.
print("warmup...", flush=True)
_ = f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text="这是一段用来预热的测试句子。", nfe_step=32, speed=1.0, seed=7)
print("warmup done", flush=True)
total_dt = 0.0
total_dur = 0.0
for i, s in enumerate(SENT, 1):
    t0 = time.time()
    wav, sr, _ = f5tts.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=s, nfe_step=32, speed=1.0, seed=42)
    dt = time.time() - t0
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    dur = len(wav) / sr if sr else len(wav) / 24000
    total_dt += dt
    total_dur += dur
    print(f"s{i} sr={sr} audio={dur:.2f}s synth={dt:.2f}s RTF={dt/dur:.3f}", flush=True)
    sf.write(f"/opt/m0/f5_out_s{i}.wav", wav, sr)
print("OVERALL synth=%.2fs audio=%.2fs avgRTF=%.3f" % (total_dt, total_dur, total_dt / total_dur), flush=True)
print("DONE", flush=True)
