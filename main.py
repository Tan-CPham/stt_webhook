import os
import sys
from dotenv import load_dotenv
from service.stt import STTService
from model.stt import STTModel
from model.ac import AstACModel
from model.download_model import download_hf_model

from logger_config import logger

# Load environment variables
load_dotenv()

# Environment config helper
def get_env_variable(var_name: str, default_value: str = None) -> str:
    """Get environment variable with optional default"""
    value = os.getenv(var_name, default_value)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required")
    return value

# Model configurations from environment
WEIGHT_DIR = get_env_variable("WEIGHT_DIR", "weight")
DEVICE = "cuda" if "cuda" in os.getenv("DEVICE", "cpu").lower() else "cpu"

# Model configurations - must be set in .env
AST_AC_MODEL_REPO_ID = get_env_variable("AST_AC_MODEL_REPO_ID")
AST_AC_MODEL_REVISION = get_env_variable("AST_AC_MODEL_REVISION")
STT_MODEL_PATH = get_env_variable("STT_MODEL_DICO_PATH")
FEATURE_EXTRACTOR_PATH = get_env_variable("FEATURE_EXTRACTOR_PATH")
TOKENIZER_PATH = get_env_variable("TOKENIZER_PATH")
HF_TOKEN = get_env_variable("HF_TOKEN", None)

def ensure_model_downloaded(model_repo_id: str, model_revision: str = "main") -> str:
    """Download model if not exists and return local path"""
    model_local_dir = os.path.join(WEIGHT_DIR, f"models--{model_repo_id.replace('/', '--')}")
    
    if os.path.exists(model_local_dir):
        return model_local_dir
    
    downloaded_path = download_hf_model(
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        model_local_dir=model_local_dir,
        hf_token=HF_TOKEN
    )
    
    if downloaded_path is None:
        raise ValueError(f"Failed to download model: {model_repo_id}")
    
    return downloaded_path

def ensure_models_ready():
    """Ensure all required models are downloaded and ready"""
    try:
        ensure_model_downloaded(STT_MODEL_PATH)
        ensure_model_downloaded(FEATURE_EXTRACTOR_PATH)
        ensure_model_downloaded(TOKENIZER_PATH)
        return True
        
    except Exception as e:
        logger.error(f"Error preparing models: {e}")
        return False

def initialize_stt_service():
    """Initialize and return STT service"""
    try:
        # Get model paths
        stt_model_local_path = os.path.join(WEIGHT_DIR, f"models--{STT_MODEL_PATH.replace('/', '--')}")
        feature_extractor_local_path = os.path.join(WEIGHT_DIR, f"models--{FEATURE_EXTRACTOR_PATH.replace('/', '--')}")
        tokenizer_local_path = os.path.join(WEIGHT_DIR, f"models--{TOKENIZER_PATH.replace('/', '--')}")
        
        stt_model = STTModel(
            model_path=stt_model_local_path,
            feature_extractor_path=feature_extractor_local_path,
            tokenizer_path=tokenizer_local_path,
            hf_token=HF_TOKEN,
            device=DEVICE,
            cache_dir=WEIGHT_DIR
        )
        
        return STTService(
            ast_ac_model=None,
            stt_model=stt_model,
            chunk_duration=15.0,
            overlap_duration=3.0,
            enable_denoise=True,
            enable_normalize=True,
            enable_silence_split=True,
            enable_multiple_attempts=True
        )
    except Exception as e:
        logger.error(f"Error loading STT model: {e}")
        return None

def main():
    """Main entry point - Setup models and start webhook server"""
    
    print("Starting Speech-to-Text Service...")
    
    # Ensure models are ready
    if not ensure_models_ready():
        print("Failed to setup models. Check .env configuration.")
        return
    
    # Initialize STT service
    stt_service = initialize_stt_service()
    if not stt_service:
        print("Failed to initialize STT service.")
        return
    
    # Start webhook server
    try:
        from api.http.webhook import initialize_webhook, start_webhook
        initialize_webhook(stt_service)
        
        port = int(os.getenv("PORT", 8000))
        print(f"Webhook server starting at http://0.0.0.0:{port}")
        print(f"For local development: ngrok http {port}")
        
        start_webhook("0.0.0.0", port)
        
    except Exception as e:
        logger.error(f"Error starting webhook: {e}")

if __name__ == "__main__":
    main() 