path = "/opt/m/voice-venv/lib/python3.10/site-packages/f5_tts/infer/utils_infer.py"
s = open(path).read()
old = "        vocoder.load_state_dict(state_dict)"
new = "        # strict=False: the saved checkpoint carries feature_extractor.mel_spec.*\n        # buffers produced by the real torchaudio MelSpectrogram; our torchaudio\n        # shim does not reproduce that exact submodule tree, and the feature\n        # extractor is unused during vocoding (mel->audio only uses the head),\n        # so dropping those keys is safe for inference.\n        vocoder.load_state_dict(state_dict, strict=False)"
assert old in s, "anchor not found"
s = s.replace(old, new)
open(path, "w").write(s)
print("patched load_vocoder -> strict=False")
