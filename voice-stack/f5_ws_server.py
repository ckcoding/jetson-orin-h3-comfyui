"""F5-TTS WebSocket 服务（Orin / voice-venv）。
复用 f5_cached_ref.F5Cached（已修复 NaN 四坑、构造期 warmup、固定 seed=0）。

协议：
- 客户端 -> 服务端：文本消息。可发纯文本，或 JSON {"text","nfe_step","speed"}。
- 服务端 -> 客户端：先发一帧 JSON 元信息（text），紧接一帧 binary（完整 wav 字节）。
"""
import asyncio, sys, os, json, io, time

sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim          # 必须在 import f5_tts 前注入 shim
from f5_cached_ref import F5Cached
import soundfile as sf
import websockets

HOST = "0.0.0.0"
PORT = 9882
DEFAULT_NFE = 16
DEFAULT_SPEED = 0.85

print("[ws] loading F5Cached (model + cache + warmup) ...", flush=True)
F5 = F5Cached()
print(f"[ws] ready on ws://{HOST}:{PORT}", flush=True)


async def handler(websocket):
    async for message in websocket:
        text, nfe, speed = None, DEFAULT_NFE, DEFAULT_SPEED
        try:
            if isinstance(message, (bytes, bytearray)):
                message = bytes(message).decode("utf-8", "ignore")
            try:
                obj = json.loads(message)
                if isinstance(obj, dict):
                    text = obj.get("text")
                    nfe = int(obj.get("nfe_step", obj.get("nfe", DEFAULT_NFE)))
                    speed = float(obj.get("speed", DEFAULT_SPEED))
            except Exception:
                text = message.strip()
            if not text:
                await websocket.send(json.dumps({"error": "empty text"}))
                continue

            t0 = time.perf_counter()
            wave, sr = F5.say(text, nfe_step=nfe, speed=speed)
            dt = time.perf_counter() - t0
            ad = len(wave) / sr
            buf = io.BytesIO()
            sf.write(buf, wave, sr, format="WAV")
            data = buf.getvalue()
            await websocket.send(json.dumps({
                "ok": True, "sr": sr, "samples": len(wave),
                "audio_sec": round(ad, 2), "synth_sec": round(dt, 2),
                "rtf": round(dt / ad, 3), "bytes": len(data),
            }))
            await websocket.send(data)
            print(f"[ws] <- {text[:24]!r} synth={dt:.2f}s rtf={dt/ad:.3f} sent={len(data)}B", flush=True)
        except Exception as e:
            try:
                await websocket.send(json.dumps({"error": repr(e)}))
            except Exception:
                pass
            print("[ws] error:", repr(e), flush=True)


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=32 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
