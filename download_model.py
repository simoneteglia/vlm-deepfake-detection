from huggingface_hub import snapshot_download
from huggingface_hub import login

login(token="YOUR_HF_TOKEN")

MODEL_NAME = ""
MODEL_PATH = "./" + MODEL_NAME.split("/")[-1]

local_model_path = snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=MODEL_PATH,
    local_dir_use_symlinks=False
)

print("Model downloaded to:", local_model_path)