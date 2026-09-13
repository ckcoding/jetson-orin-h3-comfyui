#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 voice_adapter 的 OpenAI 兼容接口（模拟 speech-to-speech 的调用方式）。"""
import io
import json
import struct
import time
import urllib.request
import uuid

ADAPTER = "http://127.0.0.1:9881"
WAV = "/opt/m/CosyVoice/asset/zero_shot_prompt.wav"


def post_multipart(url, fields, files):
    b = "----b" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                   % (b, k, v)).encode())
    for k, (fn, data, ct) in files.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                   "Content-Type: %s\r\n\r\n" % (b, k, fn, ct)).encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(("--%s--\r\n" % b).encode())
    req = urllib.request.Request(
        url, data=out.getvalue(),
        headers={"Content-Type": "multipart/form-data; boundary=" + b})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=180)
    return time.perf_counter() - t0, r.read()


def post_json(url, obj):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=180)
    return time.perf_counter() - t0, r.read()


wav = open(WAV, "rb").read()

print("=== 1) OpenAI 兼容 STT: POST /v1/audio/transcriptions ===")
for i in range(3):
    dt, body = post_multipart(
        ADAPTER + "/v1/audio/transcriptions",
        {"response_format": "json", "model": "sensevoice"},
        {"file": ("audio.wav", wav, "audio/wav")})
    print("  第%d次 %6.3fs  %s" % (i + 1, dt, body.decode("utf-8", "ignore")[:120]))

print()
print("=== 2) OpenAI 兼容 TTS: POST /v1/audio/speech ===")
for txt in ["好的，我明白了，你说得对。", "好的，我会不断进步的。"]:
    dt, body = post_json(ADAPTER + "/v1/audio/speech",
                         {"model": "melo", "input": txt, "voice": "default",
                          "response_format": "wav"})
    is_riff = body[:4] == b"RIFF"
    rate = struct.unpack("<I", body[24:28])[0] if is_riff and len(body) > 28 else 0
    dur = (len(body) - 44) / 2.0 / rate if rate else 0
    print("  %6.2fs  RIFF=%s  rate=%d  音频=%.2fs  %r" % (dt, is_riff, rate, dur, txt))
