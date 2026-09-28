"""
Hugging Face helper for the Pakistani Traffic Object Detection model.

    python deploy_hf.py            # show the model repo and its files
    python deploy_hf.py --upload   # upload the local .pt weights to the model repo

Requires a token: set HF_TOKEN or run `huggingface-cli login`.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, get_token

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = ["Pakistani_Trafic_V2.pt", "Pakistan_Traffic_Model.pt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-name", default="object-detections-model", help="model repo name under your account")
    parser.add_argument("--upload", action="store_true", help="upload the local weight files")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("No Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login`.")

    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.repo_name}"
    print(f"Authenticated as: {username}")
    print(f"Model repository: https://huggingface.co/{repo_id}")

    if args.upload:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        for name in WEIGHTS:
            path = os.path.join(PROJECT_DIR, name)
            if os.path.exists(path):
                print(f"Uploading {name} ...")
                api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo_id)
            else:
                print(f"Skipping missing file: {name}")

    print(f"Files in repository: {api.list_repo_files(repo_id)}")


if __name__ == "__main__":
    main()
