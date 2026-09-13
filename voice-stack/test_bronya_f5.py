import sys, time
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim  # 让 f5_tts 可 import（fake torchaudio）
import soundfile as sf
import sherpa_onnx

TEXT = "你好，这是一段用于对比不同语音合成音色的测试语音，希望听起来自然清晰。"

# ---------- Bronya (vits-zh-hf) via sherpa_onnx ----------
D = "/opt/m/models/vits-zh-hf-bronya"
bronya = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{D}/bronya.onnx",
            lexicon=f"{D}/lexicon.txt",
            tokens=f"{D}/tokens.txt",
        ),
        num_threads=4,
    ),
    rule_fsts=f"{D}/date.fst,{D}/number.fst,{D}/phone.fst",
    max_num_sentences=2,
))
t0 = time.perf_counter()
r = bronya.generate(TEXT, sid=0)
dt = time.perf_counter() - t0
print(f"[bronya] synth={dt:.2f}s sr={r.sample_rate} n={len(r.samples)} "
      f"RTF={dt/(len(r.samples)/r.sample_rate):.3f}")
sf.write("/opt/m0/bronya_test.wav", r.samples, r.sample_rate)

# ---------- F5 克隆用户声音 (复用已建缓存) ----------
from f5_cached_ref import F5Cached
f5 = F5Cached()
t0 = time.perf_counter()
w, sr = f5.say(TEXT)
dt = time.perf_counter() - t0
print(f"[f5-clone] synth={dt:.2f}s sr={sr} n={len(w)} "
      f"RTF={dt/(len(w)/sr):.3f}")
sf.write("/opt/m0/f5_clone_test.wav", w, sr)

print("DONE")
