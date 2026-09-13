import sys, inspect
sys.path.insert(0, "/opt/m")
import f5_torchaudio_shim
from f5_tts.api import F5TTS
from f5_tts.infer import utils_infer

with open("/opt/m/infer_src.txt", "w", encoding="utf-8") as o:
    o.write("===== F5TTS.infer =====\n")
    o.write(inspect.getsource(F5TTS.infer))
    o.write("\n\n===== infer_batch_process =====\n")
    o.write(inspect.getsource(utils_infer.infer_batch_process))
    o.write("\n\n===== chunk_text =====\n")
    o.write(inspect.getsource(utils_infer.chunk_text))
