# 容器版 TTS 输出验证: 生成并 ASR 反向识别
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/m/CosyVoice")
sys.path.insert(0, "/opt/m/CosyVoice/third_party/Matcha-TTS")
import urllib.request, io, wave
import numpy as np
import sherpa_onnx
import scipy.signal as sps
from math import gcd

boundary = "----X"
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n好的，马上帮您拨打电话。\r\n"
        f"--{boundary}--\r\n").encode()
req = urllib.request.Request("http://localhost:9880/tts/vits", data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
data = urllib.request.urlopen(req, timeout=60).read()
with wave.open(io.BytesIO(data)) as w:
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
print(f"TTS 输出: sr={sr}, {len(x)/sr:.1f}s")
g = gcd(sr, 16000)
y = sps.resample_poly(x, 16000//g, sr//g).astype(np.float32)
r = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model="/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
    tokens="/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
    use_itn=True, num_threads=4)
stream = r.create_stream()
stream.accept_waveform(16000, y)
r.decode_stream(stream)
print(f"ASR 验证: {stream.result.text}")
open("/tmp/docker_tts.wav", "wb").write(data)
