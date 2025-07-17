#!/usr/bin/env python3
"""
Main STT Service - Sử dụng Advanced STT Service
"""

import os
import sys
import torch
from pathlib import Path

# Import service và models
from service.stt import STTService
from model.stt import STTModel
from model.ac import AstACModel

def get_env_variable(name: str, default: str = None) -> str:
    """Get environment variable with optional default"""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value

def main():
    print("🎤 Advanced STT Service Tool")
    print("=" * 50)
    
    # Constants (same as main.py)
    WEIGHT_DIR = get_env_variable("WEIGHT_DIR", "weight")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AST_AC_MODEL_REPO_ID = get_env_variable("AST_AC_MODEL_REPO_ID", "MIT/ast-finetuned-audioset-10-10-0.4593")
    AST_AC_MODEL_REVISION = get_env_variable("AST_AC_MODEL_REVISION", "f826b80d28226b62986cc218e5cec390b1096902")
    HF_TOKEN = get_env_variable("HF_TOKEN", "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")

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python main_stt_service.py <audio_file> [options]")
        print("\nOptions:")
        print("  --output, -o file.txt     Save result to file")
        print("  --chunk-size 20          Chunk size in seconds")
        print("  --no-denoise             Disable denoising")
        print("  --no-normalize           Disable normalization") 
        print("  --no-silence-split       Disable silence-based splitting")
        print("  --no-multiple-attempts   Disable multiple processing attempts")
        print("  --use-speech-detection   Use AST AC for speech detection first")
        print("  --speech-validation      Use AST AC only to validate speech, then fixed chunking")
        print("  --enhance-confidence     Enable advanced confidence boosting")
        print("  --smart-mode             Auto-enable best features for audio length")
        return
    
    audio_file = sys.argv[1]
    output_file = None
    chunk_size = 15
    use_denoise = True
    use_normalize = True
    use_silence_split = True
    use_multiple_attempts = True
    use_speech_detection = False
    speech_validation_only = False
    enhance_confidence = False
    smart_mode = False
    
    # Parse options
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] in ['--output', '-o'] and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--chunk-size' and i + 1 < len(sys.argv):
            chunk_size = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--no-denoise':
            use_denoise = False
            i += 1
        elif sys.argv[i] == '--no-normalize':
            use_normalize = False
            i += 1
        elif sys.argv[i] == '--no-silence-split':
            use_silence_split = False
            i += 1
        elif sys.argv[i] == '--no-multiple-attempts':
            use_multiple_attempts = False
            i += 1
        elif sys.argv[i] == '--use-speech-detection':
            use_speech_detection = True
            i += 1
        elif sys.argv[i] == '--speech-validation':
            use_speech_detection = True # Force speech detection for validation
            speech_validation_only = True
            i += 1
        elif sys.argv[i] == '--enhance-confidence':
            enhance_confidence = True
            i += 1
        elif sys.argv[i] == '--smart-mode':
            smart_mode = True
            i += 1
        else:
            i += 1
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Smart mode detection
    if smart_mode:
        # Get audio duration for smart decisions
        try:
            import librosa
            temp_audio, temp_sr = librosa.load(audio_file, sr=None)
            audio_duration = len(temp_audio) / temp_sr
            
            print(f"🧠 Smart mode activated for {audio_duration:.1f}s audio")
            
            # For longer audio (>30s), enable speech detection and enhanced confidence
            if audio_duration > 30:
                use_speech_detection = True
                enhance_confidence = True
                print("   🎯 Long audio detected: enabling speech detection + enhanced confidence")
            
            # For very long audio (>90s), use smaller chunks and silence splitting
            if audio_duration > 90:
                chunk_size = min(chunk_size, 12)  # Use smaller chunks
                use_silence_split = True
                print(f"   📦 Very long audio: using {chunk_size}s chunks with silence splitting")
                
        except Exception as e:
            print(f"⚠️  Smart mode failed to analyze audio: {e}")
    
    print(f"📁 File: {audio_file}")
    print(f"🔧 Chunk size: {chunk_size}s")
    print(f"🔧 Features - denoise: {use_denoise}, normalize: {use_normalize}")
    print(f"🔧 Advanced - silence_split: {use_silence_split}, multiple_attempts: {use_multiple_attempts}")
    print(f"🔧 Speech detection: {use_speech_detection}")
    print(f"🔧 Enhanced confidence: {enhance_confidence}")
    print(f"🔧 Smart mode: {smart_mode}")
    
    # Initialize models
    print(f"🖥️  Device: {DEVICE}")
    
    print("🎙️  Loading STT model...")
    try:
        stt_model = STTModel(
            model_path="tel4vn-team/stt_dico",
            feature_extractor_path="nguyenvulebinh/wav2vec2-bartpho",
            tokenizer_path="nguyenvulebinh/wav2vec2-bartpho",
            hf_token=HF_TOKEN,
            device=DEVICE,
            cache_dir=WEIGHT_DIR
        )
        print("✅ STT model loaded!")
    except Exception as e:
        print(f"❌ Error loading STT model: {e}")
        return
    
    # Initialize AST AC model if needed
    ast_ac_model = None
    if use_speech_detection:
        print("🎙️  Loading AST AC model...")
        try:
            ast_ac_model = AstACModel(
                model_repo_id=AST_AC_MODEL_REPO_ID,
                model_revision=AST_AC_MODEL_REVISION,
                model_local_dir=os.path.join(WEIGHT_DIR, "ast_ac"),
                device=DEVICE
            )
            print("✅ AST AC model loaded!")
        except Exception as e:
            print(f"⚠️  AST AC model failed to load: {e}")
            print("📢 Continuing without speech detection...")
    
    # Initialize STT Service
    print("🔧 Initializing Advanced STT Service...")
    stt_service = STTService(
        ast_ac_model=ast_ac_model,
        stt_model=stt_model,
        chunk_duration=chunk_size,
        overlap_duration=3.0,
        enable_denoise=use_denoise,
        enable_normalize=use_normalize,
        enable_silence_split=use_silence_split,
        enable_multiple_attempts=use_multiple_attempts if not enhance_confidence else True  # Force enable for enhanced mode
    )
    
    # Override ensemble mode if enhanced confidence is requested
    if enhance_confidence:
        print("🚀 Enhanced confidence mode activated!")
        stt_service.enable_multiple_attempts = True
    
    # Process audio
    print("🚀 Processing audio...")
    try:
        if use_speech_detection and ast_ac_model:
            if speech_validation_only:
                print("🔍 Using speech validation approach...")
                # Load audio manually for validation
                import librosa
                audio, sr = librosa.load(audio_file, sr=None)
                text, confidence, metadata = stt_service.process_with_speech_validation(audio, sr)
            else:
                print("📡 Using speech detection approach...")
                # Load audio manually for speech detection
                import librosa
                audio, sr = librosa.load(audio_file, sr=None)
                text, confidence, segment_results = stt_service.process_with_speech_detection(audio, sr)
                
                # Create metadata
                metadata = {
                    'audio_duration': len(audio) / sr,
                    'segments_found': len(segment_results),
                    'sample_rate': sr
                }
        else:
            print("🎯 Using direct STT approach...")
            text, confidence, metadata = stt_service.process_audio_file(audio_file)
        
        # Results
        print("\n" + "🎉 " + "="*50)
        print("ADVANCED STT SERVICE RESULTS:")
        print("="*54)
        print(f"📝 Text: {text}")
        print(f"🎯 Confidence: {confidence:.1f}%")
        print(f"📁 File: {audio_file}")
        print(f"⏱️  Audio duration: {metadata.get('audio_duration', 0):.2f}s")
        
        if 'processing_time' in metadata:
            print(f"⚡ Processing time: {metadata['processing_time']:.2f}s")
            print(f"🚀 Real-time factor: {metadata['realtime_factor']:.2f}x")
        
        if 'chunks_processed' in metadata:
            print(f"🧩 Chunks processed: {metadata['chunks_processed']}")
        elif 'segments_found' in metadata:
            print(f"🎯 Speech segments: {metadata['segments_found']}")
            
        print("="*54)
        
        # Save to file
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Advanced STT Service Results\n")
                    f.write(f"File: {audio_file}\n")
                    f.write(f"Audio duration: {metadata.get('audio_duration', 0):.2f}s\n")
                    
                    if 'processing_time' in metadata:
                        f.write(f"Processing time: {metadata['processing_time']:.2f}s\n")
                        f.write(f"Real-time factor: {metadata['realtime_factor']:.2f}x\n")
                    
                    f.write(f"Confidence: {confidence:.1f}%\n")
                    
                    if 'chunks_processed' in metadata:
                        f.write(f"Chunks processed: {metadata['chunks_processed']}\n")
                    elif 'segments_found' in metadata:
                        f.write(f"Speech segments: {metadata['segments_found']}\n")
                    
                    f.write(f"Text: {text}\n")
                print(f"\n💾 Results saved to: {output_file}")
            except Exception as e:
                print(f"\n❌ Error saving file: {e}")
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return

if __name__ == "__main__":
    main() 