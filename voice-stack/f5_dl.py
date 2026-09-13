import os, time
os.environ["HF_HUB_CACHE"]="/opt/m0/.cache/huggingface"
os.environ["HF_HOME"]="/opt/m0/.cache/huggingface"
os.environ["HF_HUB_DISABLE_XET"]="1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
t=time.time()
m=snapshot_download("SWivid/F5-TTS", local_dir="/opt/m0/models/f5-tts", allow_patterns=["F5TTS_v1_Base/*"], local_dir_use_symlinks=False)
print("MODEL_DIR", m, "%.1fs"%(time.time()-t))
v=snapshot_download("charactr/vocos-mel-24khz", local_dir="/opt/m0/models/vocos-mel-24khz", local_dir_use_symlinks=False)
print("VOCOS_DIR", v)
