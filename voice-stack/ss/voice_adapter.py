#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""voice_adapter —— 把 9880 voice-service 包装成 OpenAI 兼容接口。

speech-to-speech 流水线只认 OpenAI 协议：
    POST /v1/audio/transcriptions   (multipart, 字段 file)   -> {"text": ...}
    POST /v1/audio/speech           (json)                    -> 音频字节

本机 9880 的 voice-service 协议不同：
    POST /asr/recognize             (multipart, 字段 audio)   -> {"text": ...}
    POST /tts/vits                  (multipart, 字段 text/engine) -> WAV

这一层做协议转换，顺带复用 9880 上已加载好的 SenseVoice(ASR) 与 melo(TTS)。
好处：ASR 从 whisper-small 的 ~5s 降到 ~0.16s，且中文识别更准。

关于采样率（很重要）：
    9880 的 melo TTS 输出 44.1kHz，而流水线内部是 16kHz。若原样透传，
    流水线的流式重采样器会为 44100->16000 这个比例（441/160）生成一个
    8821 抽头的 FIR，并对每个音频块做一次超长卷积——实测把 1 秒音频拖成
    48 秒才吐完。
    所以这里直接一次性重采样到 16kHz，并以裸 PCM 返回；此时流水线侧的
    重采样退化成 up==down，零开销。

环境变量：
    ADAPTER_PORT   监听端口（默认 9881）
    ASR_URL        上游 ASR 地址
    TTS_URL        上游 TTS 地址
    TTS_ENGINE     melo（默认）或 fanchen
    TTS_OUT_RATE   输出采样率（默认 16000，与流水线一致）
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ADAPTER_PORT", "9881"))
ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:9880/asr/recognize")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9880/tts/vits")
TTS_ENGINE = os.environ.get("TTS_ENGINE", "melo")
TTS_OUT_RATE = int(os.environ.get("TTS_OUT_RATE", "16000"))
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "120"))

# 重采样器：优先 soxr（快且质量高），退回 scipy，再退回 numpy 线性插值
try:
    import numpy as np
    import soxr

    def _resample(x, src: int, dst: int):
        return soxr.resample(x, src, dst)

    _BACKEND = "soxr"
except Exception:                                              # noqa: BLE001
    try:
        import numpy as np
        from scipy.signal import resample_poly

        def _resample(x, src: int, dst: int):
            g = int(np.gcd(src, dst))
            return resample_poly(x, dst // g, src // g)

        _BACKEND = "scipy"
    except Exception:                                          # noqa: BLE001
        import numpy as np

        def _resample(x, src: int, dst: int):
            n = int(round(len(x) * dst / src))
            return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)

        _BACKEND = "numpy"


# ---------------------------------------------------------------- wav
def wav_to_pcm16_16k(raw: bytes, out_rate: int) -> bytes:
    """把上游返回的 WAV 解成目标采样率的 16bit 单声道裸 PCM。"""
    with wave.open(io.BytesIO(raw), "rb") as w:
        src_rate = w.getframerate()
        nch = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if width == 2:
        x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(frames, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError("unsupported WAV width=%d" % width)

    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)

    if src_rate != out_rate:
        x = _resample(x, src_rate, out_rate)

    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()



# ---------------------------------------------------------------- multipart
def parse_boundary(content_type: str) -> bytes:
    for token in content_type.split(";"):
        token = token.strip()
        if token.lower().startswith("boundary="):
            return token[9:].strip('"').encode()
    raise ValueError("缺少 multipart boundary")


def parse_multipart(body: bytes, boundary: bytes) -> dict:
    """极简 multipart 解析：只取字段名 -> 原始字节。

    刻意不用 email 模块——它对 8bit 二进制会做文本化处理，音频数据会损坏。
    边界串是随机 32 位十六进制，与音频字节碰撞的概率可忽略。
    """
    out: dict = {}
    delim = b"--" + boundary
    for piece in body.split(delim)[1:]:
        if piece[:2] == b"--":          # 结束标记
            break
        piece = piece.lstrip(b"\r\n")
        head, sep, data = piece.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        name = None
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                for token in line.decode("latin-1").split(";"):
                    token = token.strip()
                    if token.lower().startswith("name="):
                        name = token[5:].strip('"')
        if name:
            out[name] = data
    return out


def build_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = ("----adapter" + uuid.uuid4().hex).encode()
    buf = bytearray()
    for k, v in fields.items():
        buf += b"--" + boundary + b"\r\n"
        buf += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode()
        buf += str(v).encode() + b"\r\n"
    for k, (fn, data, ct) in files.items():
        buf += b"--" + boundary + b"\r\n"
        buf += ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                % (k, fn)).encode()
        buf += ("Content-Type: %s\r\n\r\n" % ct).encode()
        buf += data + b"\r\n"
    buf += b"--" + boundary + b"--\r\n"
    return bytes(buf), "multipart/form-data; boundary=" + boundary.decode()


def post_multipart(url: str, fields: dict, files: dict) -> tuple[int, bytes, str]:
    body, ctype = build_multipart(fields, files)
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


# ---------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "voice-adapter/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("ADAPTER_VERBOSE") == "1":
            sys.stderr.write("[adapter] " + fmt % args + "\n")

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S") + ".%03d" % (int(time.time() * 1000) % 1000)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # ------------------------------------------------------------ GET
    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._json(200, {"status": "ok", "asr": ASR_URL, "tts": TTS_URL,
                             "engine": TTS_ENGINE, "out_rate": TTS_OUT_RATE,
                             "resampler": _BACKEND})
        else:
            self._json(404, {"error": "not found"})

    # ------------------------------------------------------------ POST
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        t0 = time.perf_counter()
        try:
            if path == "/v1/audio/transcriptions":
                self._transcribe(t0)
            elif path == "/v1/audio/speech":
                self._speech(t0)
            else:
                self._json(404, {"error": "not found: " + path})
        except Exception as exc:                              # noqa: BLE001
            sys.stderr.write("[adapter] %s 出错: %r\n" % (path, exc))
            try:
                self._json(500, {"error": str(exc)})
            except Exception:
                pass

    def _transcribe(self, t0: float):
        body = self._read_body()
        boundary = parse_boundary(self.headers.get("Content-Type", ""))
        parts = parse_multipart(body, boundary)
        audio = parts.get("file") or parts.get("audio") or parts.get("data")
        if not audio:
            self._json(400, {"error": "缺少 file 字段"})
            return

        code, raw, _ = post_multipart(ASR_URL, {}, {"audio": ("a.wav", audio, "audio/wav")})
        if code != 200:
            self._json(code, {"error": "上游 ASR 失败: " + raw[:200].decode("utf-8", "ignore")})
            return
        try:
            text = json.loads(raw).get("text", "")
        except Exception:
            text = raw.decode("utf-8", "ignore").strip()
        self._json(200, {"text": text, "language": "zh"})
        if os.environ.get("ADAPTER_VERBOSE") == "1":
            sys.stderr.write("[adapter] %s ASR %.3fs %r -> %r\n"
                             % (self._ts(), time.perf_counter() - t0, len(audio), text[:40]))

    def _speech(self, t0: float):
        raw = self._read_body()
        try:
            req = json.loads(raw or b"{}")
        except Exception as e:
            sys.stderr.write("[adapter] /v1/audio/speech JSON 解析失败: %r\n  body=%r\n"
                             % (e, (raw or b"")[:300]))
            sys.stderr.flush()
            self._json(400, {"error": "JSON 解析失败: " + str(e)})
            return
        text = req.get("input") or req.get("text") or ""
        if not text.strip():
            self._json(400, {"error": "缺少 input"})
            return
        engine = os.environ.get("TTS_ENGINE") or req.get("engine") or TTS_ENGINE
        code, audio, ctype = post_multipart(TTS_URL, {"text": text, "engine": engine}, {})
        if code != 200:
            self._json(code, {"error": "上游 TTS 失败: " + audio[:200].decode("utf-8", "ignore")})
            return
        if audio[:4] != b"RIFF":
            self._json(502, {"error": "上游 TTS 未返回 WAV"})
            return

        # 一次性重采样到流水线采样率，避免流水线侧的流式重采样拖垮吞吐
        want = req.get("response_format") or "pcm"
        if want == "wav":
            self._send(200, audio, "audio/wav")
        else:
            pcm = wav_to_pcm16_16k(audio, TTS_OUT_RATE)
            self._send(200, pcm, "audio/pcm")

        if os.environ.get("ADAPTER_VERBOSE") == "1":
            sys.stderr.write("[adapter] %s TTS %.3fs %r -> %dB(WAV) => %.2fs@%dHz\n"
                             % (self._ts(), time.perf_counter() - t0, text[:30], len(audio),
                                (len(audio) - 44) / 2 / 44100, TTS_OUT_RATE))
            sys.stderr.flush()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    sys.stderr.write("[adapter] listening on 127.0.0.1:%d\n" % PORT)
    sys.stderr.write("[adapter]   ASR -> %s\n" % ASR_URL)
    sys.stderr.write("[adapter]   TTS -> %s (engine=%s, out=%dHz, resampler=%s)\n"
                     % (TTS_URL, TTS_ENGINE, TTS_OUT_RATE, _BACKEND))
    sys.stderr.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
