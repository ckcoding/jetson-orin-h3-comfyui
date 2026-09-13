#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分阶段测量 speech-to-speech 各环节耗时，定位延迟来源。

用法（在 192.168.8.111 上，容器外，用 ss-venv）：
    CUDA_VISIBLE_DEVICES= PYTHONPATH=/opt/m/ss-shim NLTK_DATA=/opt/other/nltk_data \
        /opt/m/ss-venv/bin/python bench_stages.py [输入wav]
"""
import json
import os
import sys
import time
import urllib.request

WAV = sys.argv[1] if len(sys.argv) > 1 else "/opt/m/CosyVoice/asset/zero_shot_prompt.wav"
TEXT = sys.argv[2] if len(sys.argv) > 2 else "好的，我明白了，你说得对。"
LLM = os.environ.get("SS_LLM_BASE", "http://127.0.0.1:8080/v1")

RESULT = {}


def banner(s):
    print("\n" + "=" * 62)
    print(s)
    print("=" * 62, flush=True)


def head(s):
    print("\n--- " + s + " ---", flush=True)


# ---------------------------------------------------------------- 0. 输入
banner("0) 输入音频")
import soundfile as sf

_info = sf.info(WAV)
rate, nch, nframes = _info.samplerate, _info.channels, _info.frames
dur = nframes / float(rate)
print(f"  文件   : {WAV}")
print(f"  格式   : {_info.format} / {_info.subtype}")
print(f"  采样率 : {rate} Hz | 声道 {nch} | 时长 {dur:.2f}s", flush=True)
RESULT["input_dur_s"] = round(dur, 2)

# ---------------------------------------------------------------- 1. STT
banner("1) STT  faster-whisper small / int8 / CPU")
from faster_whisper import WhisperModel

t0 = time.perf_counter()
model = WhisperModel("small", device="cpu", compute_type="int8")
print(f"  模型加载      : {time.perf_counter()-t0:6.2f}s  (启动时一次，不计入单次延迟)", flush=True)

for _label, lang in (("zh", "zh"), ("auto", None)):
    t0 = time.perf_counter()
    segs, info = model.transcribe(WAV, language=lang, beam_size=1)
    txt = "".join(s.text for s in segs)
    dt = time.perf_counter() - t0
    print(f"  转写(lang={_label:4s}) : {dt:6.2f}s   RTF={dt/dur:5.2f}   检测={info.language}   {txt.strip()!r}", flush=True)
    RESULT[f"stt_{_label}_s"] = round(dt, 2)
    RESULT[f"stt_{_label}_rtf"] = round(dt / dur, 2)

# ---------------------------------------------------------------- 2. LLM
banner("2) LLM  上游 llama-server (OpenAI 兼容 / stream)")
print(f"  端点   : {LLM}", flush=True)


def llm_stream(msgs, max_tokens=60):
    body = json.dumps(
        {"messages": msgs, "max_tokens": max_tokens, "stream": True, "temperature": 0.7}
    ).encode()
    req = urllib.request.Request(
        LLM.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-local-noauth"},
    )
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    out = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            piece = (j.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
                out.append(piece)
    total = time.perf_counter() - t0
    return ttft, total, ntok, "".join(out)


ttft, total, ntok, text = llm_stream(
    [
        {"role": "system", "content": "你是一个语音助手，回答要简短口语化，一两句话即可。"},
        {"role": "user", "content": "希望你以后能够做得比我还要好哟"},
    ]
)
print(f"  首 token(TTFT) : {ttft:6.2f}s", flush=True)
print(f"  整段生成       : {total:6.2f}s   ({ntok} chunk)", flush=True)
print(f"  回复           : {text.strip()!r}", flush=True)
RESULT["llm_ttft_s"] = round(ttft, 2)
RESULT["llm_total_s"] = round(total, 2)

# ---------------------------------------------------------------- 3. TTS
banner("3) TTS  Kokoro-82M / CPU")
from kokoro import KPipeline

t0 = time.perf_counter()
pipe = KPipeline(lang_code="z", device="cpu")
print(f"  模型加载      : {time.perf_counter()-t0:6.2f}s  (启动时一次，不计入单次延迟)", flush=True)

for voice in ("zf_xiaobei",):
    t0 = time.perf_counter()
    first = None
    audio_len = 0
    for _gs, _ps, audio in pipe(TEXT, voice=voice, speed=1.0):
        if first is None:
            first = time.perf_counter() - t0
        audio_len += len(audio)
    total = time.perf_counter() - t0
    audio_s = audio_len / 24000.0
    print(
        f"  voice={voice:12s} 首块={first:6.2f}s  整段={total:6.2f}s  "
        f"音频={audio_s:5.2f}s  RTF={total/audio_s:5.2f}",
        flush=True,
    )
    RESULT["tts_first_chunk_s"] = round(first, 2)
    RESULT["tts_total_s"] = round(total, 2)
    RESULT["tts_audio_s"] = round(audio_s, 2)
    RESULT["tts_rtf"] = round(total / audio_s, 2)

# ---------------------------------------------------------------- 汇总
banner("汇总")
print(f"  输入语音时长            : {RESULT['input_dur_s']:.2f}s")
print(f"  STT (lang=zh)           : {RESULT['stt_zh_s']:.2f}s   RTF={RESULT['stt_zh_rtf']}")
print(f"  STT (lang=auto)         : {RESULT['stt_auto_s']:.2f}s   RTF={RESULT['stt_auto_rtf']}")
print(f"  LLM 首 token            : {RESULT['llm_ttft_s']:.2f}s")
print(f"  LLM 整段                : {RESULT['llm_total_s']:.2f}s")
print(f"  TTS 首块                : {RESULT['tts_first_chunk_s']:.2f}s")
print(f"  TTS 整段                : {RESULT['tts_total_s']:.2f}s   RTF={RESULT['tts_rtf']}")
print()
print(f"  理论首包 = STT + LLM_TTFT + TTS首块 = "
      f"{RESULT['stt_zh_s'] + RESULT['llm_ttft_s'] + RESULT['tts_first_chunk_s']:.2f}s")
print(f"  理论说完 = STT + LLM整段 + TTS整段  = "
      f"{RESULT['stt_zh_s'] + RESULT['llm_total_s'] + RESULT['tts_total_s']:.2f}s")
print()
print("JSON " + json.dumps(RESULT, ensure_ascii=False), flush=True)
