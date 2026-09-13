path = "/opt/m/voice-venv/lib/python3.10/site-packages/f5_tts/model/__init__.py"
s = open(path).read()
old = "from f5_tts.model.cfm import CFM\nfrom f5_tts.model.trainer import Trainer\n"
new = "from f5_tts.model.cfm import CFM\ntry:\n    from f5_tts.model.trainer import Trainer\nexcept ImportError:\n    Trainer = None\n"
assert old in s, "anchor not found"
s = s.replace(old, new)
open(path, "w").write(s)
print("patched model/__init__.py")
