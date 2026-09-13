import sys, inspect
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim
from f5_tts.model import cfm
from f5_tts.infer import utils_infer

with open("/opt/m/cfm_src.txt", "w", encoding="utf-8") as out:
    out.write("===== CFM.sample =====\n")
    out.write(inspect.getsource(cfm.CFM.sample))
    out.write("\n\n===== infer_process =====\n")
    out.write(inspect.getsource(utils_infer.infer_process))
    out.write("\n\n===== F5TTS.infer =====\n")
    out.write(inspect.getsource(cfm.F5TTS.infer) if hasattr(cfm.F5TTS, "infer") else "no F5TTS.infer in cfm")
