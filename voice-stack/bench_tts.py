import sys, os, time
sys.path.insert(0, "/opt/m/CosyVoice")
sys.path.insert(0, "/opt/m/CosyVoice/third_party/Matcha-TTS")

import torch
from cosyvoice.cli.cosyvoice import CosyVoice

MODEL_DIR = "/opt/m/models/cosyvoice/models/damo--cosyvoice-300m/snapshots/master"

print("=== 加载 CosyVoice 模型 ===")
t0 = time.time()
model = CosyVoice(MODEL_DIR)
t1 = time.time()
print("加载耗时: %.2fs" % (t1 - t0))

# 检查可用音色
print("\n=== 可用音色 ===")
spks = model.list_available_spks() if hasattr(model, "list_available_spks") else model.list_avaliable_spks()
print(spks)

# 生成 TTS（zero-shot 克隆模式，用仓库自带参考音频）
text = "你好，我是一个语音助手，很高兴为你服务。"
prompt_text = "希望你以后能够做的比我还好呦。"
prompt_wav = "/opt/m/CosyVoice/asset/zero_shot_prompt.wav"
print("\n=== 生成 TTS (zero-shot): %s ===" % text)
from cosyvoice.utils.file_utils import load_wav
prompt_speech = load_wav(prompt_wav, 16000)
t2 = time.time()
results = list(model.inference_zero_shot(text, prompt_text, prompt_speech))
t3 = time.time()
print("TTS 生成耗时: %.2fs" % (t3 - t2))

if results:
    audio = results[0]["tts_audio"]
    import soundfile as sf
    import numpy as np
    audio_np = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
    out_path = "/tmp/tts_test.wav"
    sf.write(out_path, audio_np, 24000)
    audio_len = len(audio_np) / 24000
    print("音频已保存: %s" % out_path)
    print("音频长度: %.2fs" % audio_len)
    print("RTF (实时率): %.2f (越小越好, <1.0 表示比实时快)" % ((t3 - t2) / audio_len))
