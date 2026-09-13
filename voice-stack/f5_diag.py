import sys, numpy as np
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim
from f5_cached_ref import F5Cached
import sherpa_onnx


def f5rms(tag):
    f5 = F5Cached()
    w, sr = f5.say("测试一下声音是否正常")
    arr = np.array(w) if not isinstance(w, np.ndarray) else w
    rms = float(np.sqrt((arr.astype(float) ** 2).mean()))
    print(f"{tag} type={type(w).__name__} dtype={arr.dtype} shape={arr.shape} "
          f"rms={rms:.4f} max={float(arr.max()):.2f}", flush=True)


print("=== A: F5 only ===", flush=True)
f5rms("A")

print("=== B: sherpa bronya first, then F5 ===", flush=True)
D = "/opt/m/models/vits-zh-hf-bronya"
b = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{D}/bronya.onnx",
            lexicon=f"{D}/lexicon.txt",
            tokens=f"{D}/tokens.txt"),
    ),
    rule_fsts=f"{D}/date.fst,{D}/number.fst,{D}/phone.fst",
))
_ = b.generate("你好", sid=0)
print("bronya ok", flush=True)
f5rms("B")
