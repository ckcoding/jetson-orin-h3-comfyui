import sys, numpy as np
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim
from f5_cached_ref import F5Cached

TEXT = "你好，这是一段用于对比不同语音合成音色的测试语音，希望听起来自然清晰。"
f5 = F5Cached()
w, sr = f5.say(TEXT, out_path="/opt/m0/f5_clone_test.wav")
arr = np.array(w)
print("F5 say wave", arr.shape, "nan", bool(np.isnan(arr).any()),
      "rms", float(np.sqrt((arr.astype(float) ** 2).mean())), flush=True)
