import sys

path = "/opt/m/voice-venv/lib/python3.10/site-packages/f5_tts/infer/utils_infer.py"
s = open(path).read()

start_marker = "    else:  # first pass, do preprocess"
end_marker = "        # Cache the processed reference audio"

if start_marker not in s or end_marker not in s:
    print("MARKERS NOT FOUND")
    sys.exit(1)

start = s.index(start_marker)
end = s.index(end_marker)
cache_assign = "_ref_audio_cache[audio_hash] = ref_audio"
cache_idx = s.index(cache_assign, end) + len(cache_assign)

new_block = '''    else:  # first pass, do preprocess (patched: soundfile instead of pydub/ffmpeg)
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        import numpy as np
        _ad, _sr = sf.read(ref_audio_orig)
        if _ad.ndim > 1:
            _ad = _ad.mean(axis=1)
        if _sr != 24000:
            _ad = librosa.resample(_ad.astype(np.float32), orig_sr=_sr, target_sr=24000)
            _sr = 24000
        sf.write(temp_path, _ad.astype(np.float32), _sr)
        ref_audio = temp_path
        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio'''

s2 = s[:start] + new_block + s[cache_idx:]
open(path, "w").write(s2)
print("PATCHED ok; removed %d bytes" % (cache_idx - start - len(new_block)))
