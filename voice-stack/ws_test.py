import asyncio, json, time
import numpy as np
import websockets
import wave
import scipy.signal as sps

async def test():
    uri = "ws://localhost:9880/ws"
    t_start = time.time()
    async with websockets.connect(uri, max_size=10*1024*1024) as ws:
        await ws.send(json.dumps({"cmd":"start", "sample_rate":8000, "llm":True, "tts_engine":"fanchen"}))
        ready = json.loads(await ws.recv())
        print("[%.1fs] ready: %s" % (time.time()-t_start, ready))

        with wave.open("/opt/m/models/sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav") as w:
            sr = w.getframerate()
            speech = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        speech8k = sps.resample_poly(speech, 1, 2).astype(np.float32)
        pcm = (np.clip(speech8k,-1,1)*32767).astype(np.int16).tobytes()

        chunk = 1600  # 100ms @8k
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i+chunk])
        silence = np.zeros(9600, dtype=np.int16).tobytes()
        for i in range(0, len(silence), chunk):
            await ws.send(silence[i:i+chunk])
        print("[%.1fs] audio sent (%.1fs speech)" % (time.time()-t_start, len(pcm)/1600*0.1))

        got_asr = got_llm = None
        first_audio = None
        audio_bytes = 0
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                if isinstance(msg, bytes):
                    if first_audio is None:
                        first_audio = time.time() - t_start
                    audio_bytes += len(msg)
                else:
                    ev = json.loads(msg)
                    t = ev.get("type")
                    if t == "asr_final":
                        got_asr = time.time() - t_start
                        print("[%0.1fs] ASR: %s" % (got_asr, ev["text"]))
                    elif t == "llm_text":
                        if got_llm is None: got_llm = time.time()-t_start
                        print("[%0.1fs] LLM: %s" % (time.time()-t_start, ev["text"]))
                    elif t == "tts_end":
                        print("[%0.1fs] TTS done: %d bytes audio, first_audio@%.1fs" % (time.time()-t_start, audio_bytes, first_audio or -1))
                        break
                    elif t not in ("ready",):
                        print("[%0.1fs] event: %s %s" % (time.time()-t_start, t, ev if t in ("error",) else ""))
        except asyncio.TimeoutError:
            print("timeout")

asyncio.run(test())
