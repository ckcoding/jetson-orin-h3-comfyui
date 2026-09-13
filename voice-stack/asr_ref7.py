import soundfile as s, numpy as np, os, re, sherpa_onnx

# 1) 裁取前 7 秒
d, sr = s.read("/opt/m0/ref_user.wav")
n = int(7 * sr)
seg = d[:n]
s.write("/opt/m0/ref_user7.wav", seg, sr)
print("cut %.2fs" % (len(seg) / sr), flush=True)

# 2) 对 7 秒片段重新 ASR
MODEL_DIR = "/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=os.path.join(MODEL_DIR, "model.int8.onnx"),
    tokens=os.path.join(MODEL_DIR, "tokens.txt"),
    language="zh", num_threads=4,
)
audio, _ = s.read("/opt/m0/ref_user7.wav")
if audio.ndim > 1:
    audio = audio[:, 0]
st = rec.create_stream()
st.accept_waveform(16000, audio.astype(np.float32))
rec.decode_stream(st)
raw = st.result.text
clean = re.sub(r"<\|[A-Za-z]+?\|>", "", raw).strip()
print("REF7_TEXT:", clean, flush=True)
open("/opt/m0/ref_user7_text.txt", "w").write(clean)
