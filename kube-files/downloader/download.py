import os

import huggingface_hub as hf

MODEL_ID = "CompVis/stable-diffusion-v1-4"
HUGGING_FACE_HUB_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN', default=False)

print(f"Model ID: {MODEL_ID}")
print(HUGGING_FACE_HUB_TOKEN)

hf.snapshot_download(repo_id=MODEL_ID, use_auth_token=HUGGING_FACE_HUB_TOKEN)
