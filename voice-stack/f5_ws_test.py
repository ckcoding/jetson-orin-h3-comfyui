"""同机 WebSocket 客户端测试（Orin voice-venv，连 127.0.0.1:9882）。"""
import asyncio, sys, json

import websockets

URL = "ws://127.0.0.1:9882"


async def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，这是一段用于验证 WebSocket 语音合成的测试语音。"
    out = sys.argv[2] if len(sys.argv) > 2 else "/opt/m0/f5_ws_test.wav"
    for attempt in range(30):
        try:
            async with websockets.connect(URL) as ws:
                await ws.send(json.dumps({"text": text, "nfe_step": 16, "speed": 0.85}))
                meta = await ws.recv()
                audio = await ws.recv()
                print("META:", meta)
                with open(out, "wb") as f:
                    f.write(audio)
                print(f"WROTE {out} bytes={len(audio)}")
                return
        except Exception as e:
            if attempt == 0:
                print("connecting...", repr(e))
            await asyncio.sleep(2)
    print("FAILED to connect after retries")


asyncio.run(main())
