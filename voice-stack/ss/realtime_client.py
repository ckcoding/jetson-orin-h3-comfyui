#!/usr/bin/env python3
"""speech-to-speech Realtime 链路验收客户端 (OpenAI Realtime 协议)

用法（在宿主机上，用服务同一套 venv 执行）:
    /opt/m/ss-venv/bin/python /opt/m/ss/realtime_client.py \
        --wav /opt/m/CosyVoice/asset/zero_shot_prompt.wav \
        --out /opt/update/speech-to-speech/out.wav --wait 60

说明：本项目用的是新版 OpenAI Realtime 事件名
      (response.output_audio.delta / response.output_audio_transcript.delta)，
      这里同时兼容旧版 response.audio.delta 命名。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave

import numpy as np
import soundfile as sf
import soxr
import websockets

AUDIO_DELTA = ("response.output_audio.delta", "response.audio.delta")
TEXT_DELTA = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")


async def run(url: str, wav_path: str, out_path: str, wait_s: float):
    # 1) 读音频 -> 16k 单声道 PCM16
    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = soxr.resample(data, sr, 16000).astype(np.float32)
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()
    print(f"[in ] {wav_path}  {sr}Hz -> 16kHz, {len(pcm)/32000:.2f}s, {len(pcm)} bytes")

    t0 = time.time()
    audio_chunks: list[bytes] = []
    out_rate = 24000
    transcript_parts: list[str] = []
    resp_text: list[str] = []
    first_audio_at = None
    errors: list[str] = []

    async with websockets.connect(url, max_size=None) as ws:
        # 2) 建立会话
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "你是中文语音助手，回答简短（1-2 句）。",
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}},
                    "output": {"voice": "zf_xiaobei"},
                },
            },
        }))
        print(f"[ws ] connected, session.update sent (t={time.time()-t0:.2f}s)")

        # 3) 喂音频（20ms 一帧，模拟实时）
        chunk = 640  # 20ms @16k mono = 320 samples = 640 bytes
        for i in range(0, len(pcm), chunk):
            frame = pcm[i:i + chunk]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
            await asyncio.sleep(0.02)
        print(f"[in ] 音频喂完 t={time.time()-t0:.2f}s")
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

        # 4) 收事件
        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.5, deadline - time.time()))
            except asyncio.TimeoutError:
                break
            ev = json.loads(raw)
            et = ev.get("type", "")

            if et in ("session.created", "session.updated"):
                sess = ev.get("session", {})
                out_rate = sess.get("audio", {}).get("output", {}).get("sample_rate", out_rate) or out_rate
                print(f"[ev ] {et}  out_rate={out_rate}")
            elif et == "conversation.item.input_audio_transcription.completed":
                txt = ev.get("transcript", "")
                transcript_parts.append(txt)
                print(f"[asr] {txt}")
            elif et in AUDIO_DELTA:
                if first_audio_at is None:
                    first_audio_at = time.time() - t0
                    print(f"[tts] 首个音频包 t={first_audio_at:.2f}s")
                audio_chunks.append(base64.b64decode(ev["delta"]))
            elif et in TEXT_DELTA:
                resp_text.append(ev.get("delta", ""))
                print(f"[llm] {ev.get('delta','')}", end="", flush=True)
            elif et == "response.done":
                print("[ev ] response.done")
                if audio_chunks:
                    deadline = min(deadline, time.time() + 3)
            elif et == "error":
                msg = json.dumps(ev.get("error", ev), ensure_ascii=False)[:300]
                errors.append(msg)
                print(f"[err] {msg}")
            else:
                print(f"[ev ] {et}")

    if audio_chunks:
        buf = b"".join(audio_chunks)
        with wave.open(out_path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(int(out_rate))
            f.writeframes(buf)
        print(f"[out] 音频 {len(buf)} bytes -> {out_path} ({len(buf)/2/int(out_rate):.2f}s)")

    print("-" * 50)
    print(f"识别文本: {''.join(transcript_parts).strip()[:200]}")
    print(f"回复文本: {''.join(resp_text).strip()[:200]}")
    print(f"首包延迟: {first_audio_at if first_audio_at else 'N/A'}")
    if errors:
        print(f"错误: {errors[0]}")
    print(f"结论: {'PASS 全链路出声' if audio_chunks else 'FAIL 未收到音频'}")
    return bool(audio_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    ap.add_argument("--wav", default="/opt/m/CosyVoice/asset/zero_shot_prompt.wav")
    ap.add_argument("--out", default="/opt/update/speech-to-speech/out.wav")
    ap.add_argument("--wait", type=float, default=60)
    a = ap.parse_args()
    ok = asyncio.run(run(a.url, a.wav, a.out, a.wait))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
