"""
Hugging Face Deployment Script for Pakistani Traffic Object Detection & Tracking
Deploys Model Hub Repository: Murtazakhanpro/Pakistani-Traffic-YOLO-Model
"""

import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from huggingface_hub import HfApi, get_token

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise ValueError("No Hugging Face token found. Please set HF_TOKEN or run huggingface-cli login.")

PROJECT_DIR = r"D:\Assignment_SMIT_AIDS_2026-main\Assignment_12_ObjectDetection_CV"

api = HfApi(token=token)
user_info = api.whoami()
username = user_info["name"]
print(f"Authenticated as Hugging Face user: {username}")

model_repo_id = f"{username}/object-detections-model"
print(f"Model Hub Repository: https://huggingface.co/{model_repo_id}")
files = api.list_repo_files(model_repo_id)
print(f"Active Files in Repository: {files}")
