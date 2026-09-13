#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测 9880 voice-service 的 ASR / 快速TTS 延迟，作为 speech-to-speech 的候选后端。"""
import io
import time
import urllib.request
import uuid

B = "http://127.0.0.1:9880"


def post_multipart(path, fields, files, timeout=180):
    b = "----b" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                   % (b, k, v)).encode())
    for k, (fn, data) in files.items():
        out.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                   "Content-Type: application/octet-stream\r\n\r\n" % (b, k, fn)).encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(("--%s--\r\n" % b).encode())
    body = out.getvalue()
    req = urllib.request.Request(
        B + path, data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + b})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=timeout)
    d = r.read()
    return time.perf_counter() - t0, r.status, d


wav = open("/opt/m/CosyVoice/asset/zero_shot_prompt.wav", "rb").read()
print("=== 9880 ASR (/asr/recognize)  输入 3.48s 音频 ===")
for i in range(3):
    try:
        dt, code, d = post_multipart("/asr/recognize", {}, {"audio": ("a.wav", wav)})
        txt = d[:300].decode("utf-8", "ignore")
        print("  第%d次: %6.2fs  HTTP%s  %s" % (i + 1, dt, code, txt))
    except Exception as e:
        print("  第%d次: 失败 %s" % (i + 1, e))

print()
print("=== 9880 TTS (/tts/vits) ===")
for txt in ["好的，我明白了，你说得对。", "好的，我会不断进步的。"]:
    for eng in ("fanchen", "melo"):
        try:
            dt, code, d = post_multipart("/tts/vits", {"text": txt, "engine": eng}, {})
            hdr = d[:4]
            print("  engine=%-8s %6.2fs  HTTP%s  音频=%dB  %r" % (eng, dt, code, len(d), txt))
        except Exception as e:
            print("  engine=%-8s 失败 %s" % (eng, str(e)[:120]))
