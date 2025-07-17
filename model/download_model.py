import json
import os
import shutil
from typing import Optional

from huggingface_hub import snapshot_download
from loguru import logger

MODEL_INFO_FILE_NAME = "model_info.json"


def download_hf_model(
        model_repo_id: str,
        model_revision: str,
        model_local_dir: str,
        repo_type: str = "model",
        hf_token: Optional[str] = None,
) -> Optional[str]:
    """
    Download model from Hugging Face hub and manage local storage
    
    Args:
        hf_token: token for Hugging Face authentication
        model_repo_id: The Hugging Face model repository ID
        model_revision: Model revision/version to download
        model_local_dir: Local directory to store the model
        repo_type: Repository type, defaults to "model"
        
    Returns:
        Path to downloaded model or None if failed
    """
    model_info_path = os.path.join(model_local_dir, MODEL_INFO_FILE_NAME)

    try:
        if os.path.exists(model_info_path):
            with open(model_info_path, "r", encoding="utf-8") as file:
                model_info = json.load(file)
                if (model_info["repo_id"] != model_repo_id or
                        model_info["revision"] != model_revision):
                    shutil.rmtree(model_local_dir)
                    logger.info(f"Deleted previous model files due to different repo_id or revision")

        if not os.path.exists(model_local_dir):
            os.makedirs(model_local_dir)
            logger.info(f"Downloading model from Hugging Face hub to: {model_local_dir}")

        try:
            snapshot_download(
                repo_id=model_repo_id,
                repo_type=repo_type,
                revision=model_revision,
                local_dir=model_local_dir,
                token=hf_token,
            )
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None

        if not os.path.exists(model_info_path):
            with open(model_info_path, "w", encoding="utf-8") as file:
                json.dump({
                    "repo_id": model_repo_id,
                    "revision": model_revision,
                }, file, indent=4)
            logger.info(f"Saved model info to: {model_info_path}")

        return model_local_dir

    except Exception as e:
        logger.error(f"Error in download_hf_model: {e}")
        return None
