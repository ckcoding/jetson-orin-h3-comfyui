import sherpa_onnx, soundfile as s, numpy as np, os

MODEL_DIR = "/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
model = os.path.join(MODEL_DIR, "model.int8.onnx")
tokens = os.path.join(MODEL_DIR, "tokens.txt")
wav = "/opt/m0/ref_user.wav"

recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=model, tokens=tokens, language="zh", num_threads=4,
)
audio, sample_rate = s.read(wav)
if audio.ndim > 1:
    audio = audio[:, 0]
if sample_rate != 16000:
    import librosa
    audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=16000)
    sample_rate = 16000

stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio.astype(np.float32))
recognizer.decode_stream(stream)
raw = stream.result.text
print("RAW:", repr(raw), flush=True)
# SenseVoice 可能在文本里带 <|zh|> / 情感 / 事件标记，只取中文主体
import re
clean = re.sub(r"<\|[A-Za-z]+?\|>", "", raw)
clean = clean.strip()
print("REF_TEXT:", clean, flush=True)
with open("/opt/m0/ref_user_text.txt", "w") as f:
    f.write(clean)
