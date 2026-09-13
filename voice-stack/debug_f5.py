import sys, numpy as np, torch
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim
from f5_cached_ref import F5Cached
from f5_tts.infer.utils_infer import convert_char_to_pinyin, hop_length, target_sample_rate

f5 = F5Cached()
audio = f5.ref_audio
print("ref_audio", tuple(audio.shape), "max", float(audio.abs().max()),
      "nan", bool(audio.isnan().any()), flush=True)

gen_text = "测试一下声音是否正常"
text_list = [f5.ref_text + gen_text]
ftl = convert_char_to_pinyin(text_list)
ref_audio_len = audio.shape[-1] // hop_length
ref_text_len = len(f5.ref_text.encode("utf-8"))
gen_text_len = len(gen_text.encode("utf-8"))
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / 0.85)
print("duration", duration, flush=True)

with torch.inference_mode():
    # 检查 mel 提取
    if audio.ndim == 2:
        mel = f5.tts.ema_model.mel_spec(audio)
        print("mel", tuple(mel.shape), "max", float(mel.abs().max()),
              "nan", bool(mel.isnan().any()), flush=True)
    # 直接 sample 路径
    generated, _ = f5.tts.ema_model.sample(
        cond=audio, text=ftl, duration=duration,
        steps=16, cfg_strength=2.0, sway_sampling_coef=-1)
    print("generated", tuple(generated.shape), "nan", bool(generated.isnan().any()),
          "max", float(generated.abs().max()), flush=True)
    generated = generated.to(torch.float32)
    generated = generated[:, ref_audio_len:, :].permute(0, 2, 1)
    wave = f5.tts.vocoder.decode(generated) if f5.mel_spec_type == "vocos" else f5.tts.vocoder(generated)
    wave = wave.squeeze().cpu().numpy()
    print("WAVE(sample) nan", bool(np.isnan(wave).any()),
          "rms", float(np.sqrt((wave.astype(float) ** 2).mean())), flush=True)

# 官方 infer 路径对照：把缓存波形写临时 wav，走 F5TTS.infer
import soundfile as sf
sf.write("/tmp/ref_tmp.wav", audio.squeeze().cpu().numpy(), target_sample_rate)
r = f5.tts.infer(ref_file="/tmp/ref_tmp.wav", ref_text=f5.ref_text,
                 gen_text=gen_text, nfe_step=16, speed=0.85, seed=0)
print("infer return type", type(r), flush=True)
w2 = r[0] if isinstance(r, (tuple, list)) else r
arr2 = np.array(w2)
print("WAVE(official) nan", bool(np.isnan(arr2).any()),
      "rms", float(np.sqrt((arr2.astype(float) ** 2).mean())), flush=True)
sf.write("/opt/m0/f5_debug_official.wav", arr2, target_sample_rate)
print("OFFICIAL saved", flush=True)
