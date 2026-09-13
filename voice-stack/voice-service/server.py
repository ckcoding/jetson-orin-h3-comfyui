#!/usr/bin/env python3
"""Voice Service: SenseVoice ASR + CosyVoice / F5-TTS (Jetson Orin)

HTTP API:
  POST /asr/recognize   - 文件识别
  POST /tts/zero_shot   - CosyVoice 合成（流式）
  POST /tts/vits        - 快速 TTS（f5 克隆 24k，单个 WAV；其他 engine 已下线）
  GET  /health

WebSocket 实时电话流水线:  ws://0.0.0.0:9880/ws
  上行 binary: PCM16 mono 音频 (16k 或 8k, start 时声明)
  上行 text(json): {"cmd":"start","sample_rate":16000,"llm":true,
                     "tts_engine":"f5"}   # f5(默认) | cosyvoice
                   {"cmd":"barge_in"} {"cmd":"stop"}
  下行 text(json): {"type":"asr_partial","text":..}   (保留字段)
                   {"type":"asr_final","text":..}
                   {"type":"llm_start"} / {"type":"tts_start"} / {"type":"tts_end"}
                   {"type":"llm_text","text":..}
  下行 binary: PCM16 mono 16kHz TTS 音频 (tts_start..tts_end 之间)

TTS 后端:
  * f5        - 调用常驻 F5-TTS WebSocket 服务 (ws://127.0.0.1:9882, 音色克隆)
  * cosyvoice - 进程内 CosyVoice zero-shot (GPU fp16)
"""
import os
import sys
import io
import json
import wave
import time
import uuid
import asyncio
import warnings
import tempfile
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

sys.path.insert(0, "/opt/m/CosyVoice")
sys.path.insert(0, "/opt/m/CosyVoice/third_party/Matcha-TTS")

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
import uvicorn
import websockets

COSYVOICE_DIR = os.environ.get("COSYVOICE_MODEL_DIR",
    "/opt/m/models/cosyvoice/models/damo--cosyvoice-300m/snapshots/master")
SENSEVOICE_DIR = os.environ.get("SENSEVOICE_MODEL_DIR",
    "/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
PROMPT_WAV = "/opt/m/CosyVoice/asset/zero_shot_prompt.wav"
PROMPT_TEXT = "希望你以后能够做的比我还好呦"
LLM_BASE = "http://localhost:8080/v1"
LLM_MODEL = "qwen"

# F5-TTS 常驻 WebSocket 服务 (f5ws.service, 端口 9882, 24kHz 克隆音色)
F5_WS_URL = os.environ.get("F5_WS_URL", "ws://127.0.0.1:9882")
F5_NFE_STEP = int(os.environ.get("F5_NFE_STEP", "16"))
F5_SPEED = float(os.environ.get("F5_SPEED", "0.85"))

# CosyVoice GPU 开关：默认启用 GPU(fp16)；jit/trt 经环境变量开启
CV_GPU  = os.environ.get("CV_GPU", "1") != "0"
CV_FP16 = os.environ.get("CV_FP16", "1") == "1"
CV_JIT  = os.environ.get("CV_JIT", "0") == "1"
CV_TRT  = os.environ.get("CV_TRT", "0") == "1"

# 快速 TTS 路由（仅 f5 克隆）：sherpa-onnx VITS 引擎（bronya/kokoro）已下线于 2026-09-13
# ss / adapter(9881) 以 OpenAI 协议调用，返回单个完整 WAV

app = FastAPI(title="Voice Service Realtime")

_state = {"asr": None, "tts_cv": None,
          "prompt_speech": None, "vad_config": None}


# ==================== 模型加载 ====================

def get_asr():
    if _state["asr"] is None:
        import sherpa_onnx
        _state["asr"] = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{SENSEVOICE_DIR}/model.int8.onnx",
            tokens=f"{SENSEVOICE_DIR}/tokens.txt",
            use_itn=True, num_threads=4)
    return _state["asr"]


def get_vad():
    if _state["vad_config"] is None:
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = "/opt/m/models/silero_vad.onnx"
        cfg.sample_rate = 16000
        _state["vad_config"] = cfg
    return _state["vad_config"]


def get_tts_cv():
    """CosyVoice (高质量) - HTTP 用；自动走 GPU(fp16)，cuda 不可用时回退 CPU"""
    if _state["tts_cv"] is None:
        from cosyvoice.cli.cosyvoice import CosyVoice
        from cosyvoice.utils.file_utils import load_wav
        import torch
        if CV_GPU and torch.cuda.is_available():
            cv = CosyVoice(COSYVOICE_DIR, load_jit=CV_JIT, load_trt=CV_TRT, fp16=CV_FP16)
            print("[tts] CosyVoice 加载于 GPU  device=%s  fp16=%s jit=%s trt=%s"
                  % (torch.cuda.get_device_name(0), CV_FP16, CV_JIT, CV_TRT), flush=True)
        else:
            cv = CosyVoice(COSYVOICE_DIR)
            print("[tts] CosyVoice 加载于 CPU (cuda_available=%s)"
                  % torch.cuda.is_available(), flush=True)
        _state["tts_cv"] = cv
        _state["prompt_speech"] = load_wav(PROMPT_WAV, 16000)
        _install_prompt_cache(cv)
        cv._do_warm()
    return _state["tts_cv"]


def _install_prompt_cache(cv):
    f = cv.frontend
    global_prompt_speech = _state["prompt_speech"]
    cached = {}
    orig_fezs = f.frontend_zero_shot

    def frontend_zero_shot_cached(tts_text, prompt_text, prompt_speech_16k,
                                  resample_rate, zero_shot_spk_id=""):
        if prompt_speech_16k is global_prompt_speech and cached.get("model_input") is not None:
            import torch
            mi = dict(cached["model_input"])
            text_token, text_token_len = f._extract_text_token(tts_text)
            # GPU 下缓存张量在 cuda，新建文本 token 需对齐设备，否则前向 device 不匹配
            ref = next((v for v in mi.values()
                        if hasattr(v, "device")), None)
            if ref is not None:
                dev = ref.device
                text_token = text_token.to(dev)
                text_token_len = text_token_len.to(dev)
            mi["text"] = text_token
            mi["text_len"] = text_token_len
            return mi
        return orig_fezs(tts_text, prompt_text, prompt_speech_16k,
                         resample_rate, zero_shot_spk_id)

    f.frontend_zero_shot = frontend_zero_shot_cached

    def warm():
        mi = orig_fezs("预热", PROMPT_TEXT, global_prompt_speech, cv.sample_rate, "")
        cached["model_input"] = mi
        print("[tts] prompt 特征缓存完成", flush=True)

    cv._do_warm = warm


# ==================== 工具 ====================

def _read_audio_to_16k(data: bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    try:
        samples, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        samples = samples[:, 0]
        if sr != 16000:
            import librosa
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        return samples.astype(np.float32)
    finally:
        os.unlink(tmp.name)


def normalize_peak(s: np.ndarray, target: float = 0.9) -> np.ndarray:
    """峰值归一化 (TTS 输出音量偏低, 不归一化听不清)；空数组原样返回"""
    if s.size == 0:
        return s
    peak = float(np.abs(s).max())
    if peak > 1e-6:
        s = s * (target / peak)
    return np.clip(s, -1.0, 1.0)


def _decode_wav_bytes(data: bytes):
    """解析完整 WAV 字节 → (sample_rate, float32 mono samples)"""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        s = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        s = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        s = np.frombuffer(frames, dtype=np.float32)
    if ch > 1:
        s = s.reshape(-1, ch).mean(axis=1)
    return sr, s.astype(np.float32)


def resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样 (polyphase 滤波, 保波形)"""
    if orig_sr == target_sr:
        return x.astype(np.float32)
    import scipy.signal as sps
    from math import gcd
    g = gcd(int(orig_sr), int(target_sr))
    up, down = target_sr // g, orig_sr // g
    out = sps.resample_poly(x.astype(np.float64), up, down)
    return out.astype(np.float32)


async def _tts_f5_ws(text: str, nfe_step: int = None, speed: float = None):
    """调用常驻 F5 WS 服务做克隆 TTS，返回 (sample_rate, float32 mono samples)。

    复用 /ws 流水线的同一套 F5 服务 (ws://127.0.0.1:9882)，使用已缓存的用户音色。
    """
    async with websockets.connect(F5_WS_URL, max_size=32 * 1024 * 1024) as f5:
        await f5.send(json.dumps({"text": text,
                                  "nfe_step": nfe_step or F5_NFE_STEP,
                                  "speed": speed or F5_SPEED}))
        meta_raw = await f5.recv()
        wav_bytes = await f5.recv()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8", "ignore")
    try:
        sr = int(json.loads(meta_raw).get("sr", 24000))
    except Exception:
        sr = 24000
    _, samples = _decode_wav_bytes(wav_bytes)
    return sr, normalize_peak(samples)


def _wav_response(samples: np.ndarray, sr: int) -> "Response":
    """把 float32 mono 采样打包成单个完整 WAV 响应（samples 须已归一化）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav")


def _tts_stream(model_output, sample_rate):
    def gen():
        for chunk in model_output:
            audio = chunk["tts_speech"].cpu().numpy().flatten()
            audio = normalize_peak(audio)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes((audio * 32768).astype(np.int16).tobytes())
            yield buf.getvalue()
    return StreamingResponse(gen(), media_type="audio/wav")


# ==================== HTTP ====================

@app.post("/asr/recognize")
async def asr_recognize(audio: UploadFile = File(...)):
    recognizer = get_asr()
    data = await audio.read()
    samples = _read_audio_to_16k(data)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    return {"text": stream.result.text}


@app.post("/tts/zero_shot")
async def tts_zero_shot(text: str = Form(...),
                        prompt_text: str = Form(PROMPT_TEXT),
                        prompt_wav: UploadFile = File(None)):
    cv = get_tts_cv()
    if prompt_wav is not None and prompt_wav.filename:
        data = await prompt_wav.read()
        samples = _read_audio_to_16k(data)
        import torch
        prompt_speech = torch.from_numpy(samples).unsqueeze(0)
    else:
        prompt_speech = _state["prompt_speech"]
    out = cv.inference_zero_shot(text, prompt_text, prompt_speech, stream=True)
    return _tts_stream(out, cv.sample_rate)


@app.post("/tts/vits")
async def tts_vits(text: str = Form(...),
                   engine: str = Form("f5"),
                   sid: int = Form(0),
                   speed: float = Form(0.0)):
    """快速 TTS —— 当前仅 f5 克隆（24k）；其他 engine 已下线。

    供 voice_adapter(9881) 以 OpenAI 协议调用；返回单个完整 WAV。
    """
    if engine != "f5":
        sys.stderr.write(f"[tts/vits] engine={engine!r} 已下线，自动改为 f5\n")
        engine = "f5"
    try:
        sr, samples = await _tts_f5_ws(text)
    except Exception as e:
        sys.stderr.write(f"[tts/vits] F5 调用失败, 返回静音: {e!r}\n")
        sr, samples = 24000, np.zeros(int(24000 * 0.2), dtype=np.float32)
    if samples.size == 0:
        samples = np.zeros(max(1, int(sr * 0.2)), dtype=np.float32)
    return _wav_response(samples, sr)


@app.get("/health")
async def health():
    return {"status": "ok",
            "asr": _state["asr"] is not None,
            "tts_cosyvoice": _state["tts_cv"] is not None,
            "tts_f5": F5_WS_URL}


@app.get("/")
async def root():
    return {"service": "Voice Pipeline Realtime",
            "ws": "ws://host:9880/ws",
            "http": {"/asr/recognize": "POST audio", "/tts/zero_shot": "POST text",
                     "/tts/vits": "POST text  (engine 固定 f5)"},
            "ws_tts_engine": "f5 | cosyvoice"}


# ==================== WebSocket 电话流水线 ====================

def _llm_stream_sync(user_text: str):
    """调 llama-server 流式 chat, 逐 chunk yield 文本 (阻塞, 在线程池跑)"""
    import urllib.request
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是车载语音助手，回答简短口语化，不要用列表和markdown，每句话控制在30字以内。"},
            {"role": "user", "content": user_text},
        ],
        "stream": True,
        "max_tokens": 120,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{LLM_BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    buf = ""
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
        except Exception:
            continue
        if delta:
            yield delta


def _split_sentences(text: str, min_len: int = 2):
    """按中英文标点切句"""
    import re
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


class PhoneSession:
    """一个电话会话: VAD → ASR → LLM → TTS"""

    def __init__(self, ws, use_llm=True, tts_engine="f5"):
        self.ws = ws
        self.use_llm = use_llm
        self.tts_engine = tts_engine
        self.in_rate = 16000
        self.history = []
        self.vad = None
        self.gen_task = None          # 当前 LLM+TTS 生成任务
        self.session_id = str(uuid.uuid1())[:8]
        self.tts_queue = asyncio.Queue(maxsize=100)   # 待合成句子
        self.tts_active = False

    def reset_vad(self):
        import sherpa_onnx
        self.vad = sherpa_onnx.VoiceActivityDetector(get_vad(), buffer_size_in_seconds=120)

    # ---------- 音频输入 → VAD ----------
    async def feed_audio(self, pcm16: bytes):
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if self.in_rate != 16000:
            samples = resample_linear(samples, self.in_rate, 16000)
        if self.vad is None:
            self.reset_vad()
        loop = asyncio.get_event_loop()

        def vad_feed():
            self.vad.accept_waveform(samples)
            segments = []
            while not self.vad.empty():
                seg = self.vad.front
                self.vad.pop()
                segments.append(np.asarray(seg.samples, dtype=np.float32))
            return segments

        segments = await loop.run_in_executor(None, vad_feed)
        if segments:
            print(f"[ws] VAD 检测到 {len(segments)} 段语音", flush=True)
        for seg in segments:
            await self.on_speech_end(seg)

    # ---------- ASR ----------
    async def on_speech_end(self, seg: np.ndarray):
        if len(seg) < 1600:  # <0.1s 丢弃
            return
        loop = asyncio.get_event_loop()

        def do_asr():
            recognizer = get_asr()
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, seg)
            recognizer.decode_stream(stream)
            return stream.result.text

        text = await loop.run_in_executor(None, do_asr)
        text = text.strip()
        if not text:
            return
        print(f"[ws] ASR: {text}", flush=True)
        await self.ws.send_json({"type": "asr_final", "text": text})
        if self.use_llm:
            # 若已有任务在跑, 先打断
            await self.stop_gen()
            self.gen_task = asyncio.create_task(self.llm_tts_pipeline(text))

    # ---------- LLM + TTS 流水线 ----------
    async def stop_gen(self):
        if self.gen_task and not self.gen_task.done():
            self.gen_task.cancel()
            try:
                await self.gen_task
            except (asyncio.CancelledError, Exception):
                pass
        self.gen_task = None

    def _pick_sentence(self, buf: str):
        """从缓冲取完整句子"""
        import re
        m = re.search(r"^(.*?[。！？!?；;\n])", buf, re.S)
        if m:
            return m.group(1).strip(), buf[m.end():]
        return None, buf

    async def _tts_generate(self, sentence: str):
        """按 tts_engine 路由: f5 → 常驻 F5 WS 服务; cosyvoice → 进程内 CosyVoice"""
        if self.tts_engine == "cosyvoice":
            loop = asyncio.get_event_loop()

            def _cv():
                cv = get_tts_cv()
                chunks = []
                for out in cv.inference_zero_shot(
                        sentence, PROMPT_TEXT, _state["prompt_speech"], stream=True):
                    chunks.append(out["tts_speech"].cpu().numpy().flatten())
                audio = (np.concatenate(chunks) if chunks
                         else np.zeros(1, dtype=np.float32))
                return normalize_peak(audio), cv.sample_rate

            return await loop.run_in_executor(None, _cv)

        # 默认: F5 常驻 WS 服务 (24kHz 克隆音色)
        async with websockets.connect(F5_WS_URL, max_size=32 * 1024 * 1024) as f5:
            await f5.send(json.dumps({"text": sentence,
                                      "nfe_step": F5_NFE_STEP,
                                      "speed": F5_SPEED}))
            meta_raw = await f5.recv()
            wav_bytes = await f5.recv()
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8", "ignore")
        try:
            sr = int(json.loads(meta_raw).get("sr", 24000))
        except Exception:
            sr = 24000
        _, samples = _decode_wav_bytes(wav_bytes)
        return normalize_peak(samples), sr

    async def llm_tts_pipeline(self, user_text: str):
        loop = asyncio.get_event_loop()
        await self.ws.send_json({"type": "llm_start"})
        self.history.append({"role": "user", "content": user_text})
        # 只保留最近 6 轮
        messages = ([{"role": "system",
                      "content": "你是车载语音助手，回答简短口语化，不用列表和markdown，每句不超过30字。"}]
                    + self.history[-6:])
        reply_buf = ""
        reply_all = ""
        sentence_q = asyncio.Queue()

        # --- LLM 流式, 按句切出 ---
        def llm_worker():
            import urllib.request
            body = json.dumps({"model": LLM_MODEL, "messages": messages,
                               "stream": True, "max_tokens": 150,
                               "temperature": 0.7}).encode()
            req = urllib.request.Request(f"{LLM_BASE}/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                buf = ""
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        d = json.loads(payload)["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if not d:
                        continue
                    buf += d
                    while True:
                        sent, buf = self._pick_sentence(buf)
                        if sent is None:
                            break
                        sentence_q.put_nowait(sent)
                if buf.strip():
                    sentence_q.put_nowait(buf.strip())
            except Exception as e:
                sentence_q.put_nowait(None)
                raise
            finally:
                sentence_q.put_nowait(None)  # 结束标记

        llm_task = loop.run_in_executor(None, llm_worker)
        tts_tasks = []
        batch_id = str(uuid.uuid1())[:8]

        await self.ws.send_json({"type": "tts_start", "batch": batch_id})

        try:
            while True:
                sent = await sentence_q.get()
                if sent is None:
                    break
                reply_all += sent
                await self.ws.send_json({"type": "llm_text", "text": sent})
                # 合成 (f5 走 WS 服务 / cosyvoice 走线程池)
                samples, sr = await self._tts_generate(sent)
                if self.gen_task and self.gen_task.cancelled():
                    return
                # 统一重采样到 16k PCM16 下发 (f5 原生 24k, cosyvoice 22.05k)
                out_rate = 16000
                out = resample_linear(samples, sr, out_rate)
                pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()
                await self.ws.send_bytes(pcm)
            await self.ws.send_json({"type": "tts_end", "batch": batch_id})
            self.history.append({"role": "assistant", "content": reply_all})
        except asyncio.CancelledError:
            await self.ws.send_json({"type": "tts_end", "batch": batch_id, "cancelled": True})
            raise


@app.websocket("/ws")
async def ws_phone(ws: WebSocket):
    await ws.accept()
    session = None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                if session:
                    await session.feed_audio(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    cmd = json.loads(msg["text"])
                except Exception:
                    continue
                c = cmd.get("cmd")
                if c == "start":
                    session = PhoneSession(
                        ws, use_llm=cmd.get("llm", True),
                        tts_engine=cmd.get("tts_engine", "f5"))
                    session.in_rate = int(cmd.get("sample_rate", 16000))
                    # 预热模型
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, get_asr)
                    await loop.run_in_executor(None, get_vad)
                    if session.tts_engine == "cosyvoice":
                        # CosyVoice 首次加载较慢(GPU fp16), 提前预热
                        await loop.run_in_executor(None, get_tts_cv)
                    # f5 由常驻服务 :9882 托管, 无需在此预热
                    await ws.send_json({"type": "ready",
                                        "asr": "sensevoice",
                                        "tts": session.tts_engine,
                                        "llm": session.use_llm,
                                        "in_rate": session.in_rate,
                                        "out_rate": 16000})
                elif c == "barge_in" and session:
                    await session.stop_gen()
                    await ws.send_json({"type": "tts_end", "cancelled": True})
                elif c == "stop" and session:
                    await session.stop_gen()
                elif c == "ping":
                    await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
    finally:
        if session:
            await session.stop_gen()


if __name__ == "__main__":
    print("[voice] 预加载 ASR (SenseVoice)...", flush=True)
    get_asr()
    print("[voice] 预加载 VAD...", flush=True)
    get_vad()
    print("[voice] 未预加载 CosyVoice (按需加载); F5 由独立服务 :9882 提供", flush=True)
    print("[voice] 模型就绪, 启动 API+WS :9880", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=9880, ws_ping_interval=20)
