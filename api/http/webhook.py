import os
import sys
import time
from datetime import datetime
from typing import Optional
import io

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import uvicorn

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from util import audio as audio_util
from logger_config import logger

# Response Models  
class AudioResponse(BaseModel):
    status: str
    message: str
    audio_filename: str
    result: Optional[dict] = None
    error_detail: Optional[str] = None
    metadata: Optional[dict] = {}

# Global variables
stt_service = None
app = FastAPI(title="STT Webhook API")

def initialize_webhook(service):
    """Initialize webhook with STT service from main.py"""
    global stt_service
    stt_service = service
    logger.info("Webhook initialized with STT service")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy" if stt_service else "unhealthy",
        "timestamp": datetime.now()
    }

@app.post("/webhook/process-audio", response_model=AudioResponse)
async def process_audio(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    timestamp: Optional[str] = Form(None, description="Request timestamp"),
    metadata: Optional[str] = Form("{}", description="Additional metadata as JSON string")
):
    """Main webhook endpoint for ERP - receives audio via form-data"""
    
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT service not ready")
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not audio_file.content_type or not any(
            audio_file.content_type.startswith(mime) 
            for mime in ['audio/', 'application/octet-stream']
        ):
            return AudioResponse(
                status="error",
                message="Invalid file type",
                audio_filename=audio_file.filename,
                error_detail=f"Expected audio file, got: {audio_file.content_type}",
                metadata={}
            )
        
        # Read file content into memory
        file_content = await audio_file.read()
        if len(file_content) == 0:
            return AudioResponse(
                status="error",
                message="Empty file",
                audio_filename=audio_file.filename,
                error_detail="Received empty audio file",
                metadata={}
            )
        
        # Process audio: load from memory + extract left channel + STT
        audio_data, sr = audio_util.load_and_extract_left_channel(
            io.BytesIO(file_content), target_sr=16000, normalize=True
        )
        
        text, confidence, chunks_count = stt_service.advanced_transcribe(audio_data, sr)
        
        if not text.strip():
            return AudioResponse(
                status="error",
                message="No speech detected",
                audio_filename=audio_file.filename, 
                error_detail="No speech detected in audio",
                metadata={}
            )
        
        # Success response
        processing_time = time.time() - start_time
        
        return AudioResponse(
            status="success",
            message="Audio processed successfully",
            audio_filename=audio_file.filename,
            result={
                "text": text.strip(),
                "confidence": confidence,
                "audio_duration": len(audio_data) / sr,
                "processing_time": processing_time,
                "chunks_processed": chunks_count,
                "file_size_bytes": len(file_content)
            },
            metadata={}
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return AudioResponse(
            status="error",
            message="Audio processing failed",
            audio_filename=audio_file.filename or "unknown",
            error_detail=str(e),
            metadata={}
        )

def start_webhook(host="0.0.0.0", port=8000):
    """Start webhook server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_webhook() 