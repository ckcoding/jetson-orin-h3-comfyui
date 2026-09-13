#!/usr/bin/env python3
"""
Orin 语音对话: ASR(SenseVoice) + LLM(Qwen) + TTS(CosyVoice)
用法: python voice_chat.py <input.wav>
输出: /tmp/output.wav
"""
import sys
import os
import time
import wave
import requests

# ============ 配置 ============
SENSEVOICE_MODEL = "/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx"
SENSEVOICE_TOKENS = "/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"
COSYVOICE_MODEL_DIR = "/opt/m/models/cosyvoice/models/damo--cosyvoice-300m/snapshots/master"
LLM_API = "http://127.0.0.1:8080/v1/chat/completions"
OUTPUT_WAV = "/tmp/output.wav"

def log(msg):
    print(f"[voice_chat] {msg}", flush=True)

# ============ 1. ASR: SenseVoice 语音转文字 ============
def asr_sensevoice(wav_file):
    log("=== 1. ASR (SenseVoice) ===")
    import sherpa_onnx
    import numpy as np

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=SENSEVOICE_MODEL,
        tokens=SENSEVOICE_TOKENS,
        use_itn=True,
        num_threads=4,
    )

    # 读取 wav
    with wave.open(wav_file, "rb") as f:
        sample_rate = f.getframerate()
        channels = f.getnchannels()
        samples = f.readframes(f.getnframes())
        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0

    log(f"音频: {len(audio)/sample_rate:.1f}s, {sample_rate}Hz, {channels}ch")

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    text = stream.result.text.strip()
    log(f"识别结果: {text}")
    return text

# ============ 2. LLM: Qwen 对话 ============
def llm_qwen(text):
    log("=== 2. LLM (Qwen) ===")
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "你是车载语音助手，请用简短自然的中文回答，不超过50字。"},
            {"role": "user", "content": text}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    t0 = time.time()
    resp = requests.post(LLM_API, json=payload, timeout=60)
    t1 = time.time()
    result = resp.json()["choices"][0]["message"]["content"]
    log(f"LLM 回复 ({t1-t0:.1f}s): {result}")
    return result

# ============ 3. TTS: CosyVoice 文字转语音 ============
def tts_cosyvoice(text, output_wav):
    log("=== 3. TTS (CosyVoice) ===")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice
        import torchaudio

        cosyvoice = CosyVoice(COSYVOICE_MODEL_DIR)
        log("CosyVoice 模型加载完成")

        t0 = time.time()
        for i, chunk in enumerate(cosyvoice.inference_sft(text, "中文女", stream=False)):
            torchaudio.save(output_wav, chunk["tts_speech"], 22050)
            log(f"音频片段 {i} 已保存")
        t1 = time.time()
        log(f"TTS 完成 ({t1-t0:.1f}s), 输出: {output_wav}")
        return True
    except Exception as e:
        log(f"CosyVoice 失败: {e}")
        log("尝试用 edge-tts 替代...")
        return tts_edge(text, output_wav)

def tts_edge(text, output_wav):
    """备用: edge-tts (在线)"""
    try:
        import asyncio
        import edge_tts
        async def gen():
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(output_wav)
        asyncio.run(gen())
        log(f"edge-tts 完成, 输出: {output_wav}")
        return True
    except Exception as e:
        log(f"edge-tts 也失败: {e}")
        return False

# ============ 主流程 ============
def main():
    if len(sys.argv) < 2:
        print("用法: python voice_chat.py <input.wav>")
        sys.exit(1)

    input_wav = sys.argv[1]
    if not os.path.exists(input_wav):
        print(f"文件不存在: {input_wav}")
        sys.exit(1)

    t_start = time.time()

    # 1. ASR
    text = asr_sensevoice(input_wav)
    if not text:
        log("未识别到语音")
        sys.exit(1)

    # 2. LLM
    reply = llm_qwen(text)

    # 3. TTS
    tts_cosyvoice(reply, OUTPUT_WAV)

    t_end = time.time()
    log(f"=== 总耗时: {t_end-t_start:.1f}s ===")
    print(f"\n你说: {text}")
    print(f"回复: {reply}")
    print(f"音频: {OUTPUT_WAV}")

if __name__ == "__main__":
    main()
