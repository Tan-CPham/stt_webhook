import numpy as np
import time
import librosa
from typing import Tuple, List, Optional
from numpy.typing import NDArray
from logger_config import logger
from model.ac import AstACModel
from model.stt import STTModel
from util import audio as audio_util


class STTService:
    def __init__(
        self,
        ast_ac_model: Optional[AstACModel] = None,
        stt_model: STTModel = None,
        chunk_duration: float = 15.0,  # 15 giây mỗi chunk
        overlap_duration: float = 3.0,  # 3 giây overlap
        min_speech_duration: float = 0.5,  # Tối thiểu 0.5s mới coi là speech
        confidence_threshold: float = 0.5,  # Ngưỡng confidence cho AST AC
        enable_denoise: bool = True,
        enable_normalize: bool = True,
        enable_silence_split: bool = True,
        enable_multiple_attempts: bool = True
    ):
        """
        Khởi tạo Advanced STT Service với tất cả kỹ thuật cải thiện
        
        Args:
            ast_ac_model: Model AST AC để detect speech (optional)
            stt_model: Model STT để chuyển speech thành text
            chunk_duration: Thời lượng mỗi chunk (giây)
            overlap_duration: Thời gian overlap giữa các chunk (giây)
            min_speech_duration: Thời lượng tối thiểu để coi là speech (giây)
            confidence_threshold: Ngưỡng confidence cho speech detection
            enable_denoise: Bật tính năng giảm noise
            enable_normalize: Bật tính năng normalize audio
            enable_silence_split: Bật tính năng chia theo khoảng lặng
            enable_multiple_attempts: Bật tính năng thử nhiều cách xử lý
        """
        self.ast_ac_model = ast_ac_model
        self.stt_model = stt_model
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.min_speech_duration = min_speech_duration
        self.confidence_threshold = confidence_threshold
        self.enable_denoise = enable_denoise
        self.enable_normalize = enable_normalize
        self.enable_silence_split = enable_silence_split
        self.enable_multiple_attempts = enable_multiple_attempts
        
        logger.info(f"Advanced STT Service initialized - chunk: {chunk_duration}s, overlap: {overlap_duration}s")
        logger.info(f"Features enabled - denoise: {enable_denoise}, normalize: {enable_normalize}, silence_split: {enable_silence_split}")

    def denoise_audio(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """
        Giảm noise đơn giản bằng spectral subtraction
        """
        if not self.enable_denoise:
            return audio
            
        try:
            # Compute spectral magnitude
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frame = int(0.5 * sr / 512)  # 512 is default hop_length
            if noise_frame < magnitude.shape[1]:
                noise_magnitude = np.mean(magnitude[:, :noise_frame], axis=1, keepdims=True)
                
                # Spectral subtraction
                clean_magnitude = magnitude - 0.5 * noise_magnitude
                clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
                
                # Reconstruct audio
                phase = np.angle(stft)
                clean_stft = clean_magnitude * np.exp(1j * phase)
                clean_audio = librosa.istft(clean_stft)
                
                return clean_audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Denoise failed: {e}")
        
        return audio

    def normalize_audio(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize audio để tăng chất lượng
        """
        if not self.enable_normalize:
            return audio
            
        try:
            # RMS normalization using audio_util
            rms = audio_util.calculate_rms(audio)
            if rms > 0:
                audio = audio / rms * 0.1
            
            # Clip to prevent distortion
            audio = np.clip(audio, -1.0, 1.0)
            return audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Normalize failed: {e}")
            return audio

    def split_by_silence(self, audio: NDArray[np.float32], sr: int, 
                        min_silence_len: float = 0.5, silence_thresh: float = -40) -> List[NDArray[np.float32]]:
        """
        Chia audio dựa trên khoảng lặng
        """
        if not self.enable_silence_split:
            return [audio]
            
        try:
            # Convert to dB
            audio_db = librosa.amplitude_to_db(np.abs(audio))
            
            # Find silence
            silence_frames = audio_db < silence_thresh
            
            # Find continuous silence regions
            silence_regions = []
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silence_frames):
                if is_silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not is_silent and in_silence:
                    silence_duration = (i - silence_start) / sr * len(audio) / len(audio_db)
                    if silence_duration >= min_silence_len:
                        silence_regions.append((silence_start, i))
                    in_silence = False
            
            # Split audio at silence regions
            if not silence_regions:
                return [audio]
            
            segments = []
            last_end = 0
            
            for start, end in silence_regions:
                # Convert frame indices to sample indices
                start_sample = int(start / len(audio_db) * len(audio))
                end_sample = int(end / len(audio_db) * len(audio))
                
                if start_sample > last_end:
                    segments.append(audio[last_end:start_sample])
                last_end = end_sample
            
            # Add final segment
            if last_end < len(audio):
                segments.append(audio[last_end:])
            
            # Remove segments < 0.5s
            valid_segments = [seg for seg in segments if len(seg) > sr * 0.5]
            return valid_segments if valid_segments else [audio]
            
        except Exception as e:
            logger.warning(f"Silence split failed: {e}")
            return [audio]

    def remove_overlap_text(self, current_text: str, prev_text: str, min_overlap_words: int = 2) -> str:
        """
        Loại bỏ phần overlap giữa 2 đoạn text
        """
        if not prev_text:
            return current_text
        
        current_words = current_text.split()
        prev_words = prev_text.split()
        
        if len(prev_words) < min_overlap_words or len(current_words) < min_overlap_words:
            return current_text
        
        # Tìm overlap từ cuối prev_text và đầu current_text
        max_overlap = min(len(prev_words), len(current_words), 10)  # Max 10 từ overlap
        
        for overlap_len in range(max_overlap, min_overlap_words-1, -1):
            prev_suffix = " ".join(prev_words[-overlap_len:])
            current_prefix = " ".join(current_words[:overlap_len])
            
            if prev_suffix.lower() == current_prefix.lower():
                # Tìm thấy overlap, loại bỏ phần đầu của current_text
                return " ".join(current_words[overlap_len:])
        
        return current_text

    def enhance_audio_quality(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """
        Enhance audio quality để tăng confidence
        """
        try:
            # 1. Spectral gating để remove noise tốt hơn
            stft = librosa.stft(audio, hop_length=512, win_length=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Compute noise threshold from quieter parts
            noise_threshold = np.percentile(magnitude, 20)  # Bottom 20% as noise
            
            # Create a soft mask instead of hard gating
            mask = magnitude / (magnitude + noise_threshold)
            clean_magnitude = magnitude * mask
            
            # Reconstruct
            clean_stft = clean_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(clean_stft, hop_length=512, win_length=2048)
            
            # 2. Band-pass filter for speech frequencies (300Hz - 3400Hz)
            from scipy import signal
            nyquist = sr / 2
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            enhanced_audio = signal.filtfilt(b, a, enhanced_audio)
            
            # 3. Dynamic range compression
            threshold = 0.1
            ratio = 4.0
            above_threshold = np.abs(enhanced_audio) > threshold
            enhanced_audio[above_threshold] = np.sign(enhanced_audio[above_threshold]) * (
                threshold + (np.abs(enhanced_audio[above_threshold]) - threshold) / ratio
            )
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio

    def ensemble_transcribe(self, audio: NDArray[np.float32], sr: int) -> Tuple[str, float]:
        """
        Ensemble processing with multiple audio versions to boost confidence
        """
        results = []
        
        # Version 1: Original audio
        try:
            text1, conf1 = self.stt_model.predict(audio, sr)
            if text1.strip():
                results.append((text1.strip(), conf1, 1.0))  # weight = 1.0
        except:
            pass
        
        # Version 2: Enhanced audio
        try:
            enhanced_audio = self.enhance_audio_quality(audio, sr)
            text2, conf2 = self.stt_model.predict(enhanced_audio, sr)
            if text2.strip():
                results.append((text2.strip(), conf2, 1.2))  # weight = 1.2 (higher priority)
        except:
            pass
        
        # Version 3: Normalized + denoised
        try:
            processed_audio = self.normalize_audio(audio)
            processed_audio = self.denoise_audio(processed_audio, sr)
            text3, conf3 = self.stt_model.predict(processed_audio, sr)
            if text3.strip():
                results.append((text3.strip(), conf3, 1.1))  # weight = 1.1
        except:
            pass
        
        # Version 4: Slightly amplified (for quiet audio)
        try:
            amplified_audio = audio * 1.5
            amplified_audio = np.clip(amplified_audio, -1.0, 1.0)
            text4, conf4 = self.stt_model.predict(amplified_audio, sr)
            if text4.strip():
                results.append((text4.strip(), conf4, 0.9))  # weight = 0.9 (lower priority)
        except:
            pass
        
        if not results:
            return "", 0.0
        
        # Find best result by weighted confidence
        best_result = max(results, key=lambda x: x[1] * x[2])  # confidence * weight
        return best_result[0], best_result[1]

    def post_process_confidence(self, text: str, confidence: float) -> float:
        """
        Adjust confidence based on text quality indicators
        """
        if not text.strip():
            return 0.0
        
        # Factors that increase confidence
        boost_factors = 0.0
        
        # 1. Vietnamese words detection
        vietnamese_words = ['dạ', 'chị', 'anh', 'em', 'mình', 'được', 'không', 'thì', 'là', 'của', 'cho', 'với', 'về']
        viet_word_count = sum(1 for word in vietnamese_words if word in text.lower())
        if viet_word_count > 3:
            boost_factors += 2.0  # +2% for good Vietnamese content
        
        # 2. Sentence structure (có dấu câu)
        if any(punct in text for punct in ['.', ',', '?', '!']):
            boost_factors += 1.0  # +1% for punctuation
        
        # 3. Length appropriateness (not too short, not too fragmented)
        words = text.split()
        if 5 <= len(words) <= 100:  # Good length range
            boost_factors += 1.5  # +1.5%
        
        # 4. Coherent speech patterns
        if len(text) > 50 and text.count(' ') > 5:  # Reasonable speech length
            boost_factors += 1.0
        
        # Factors that decrease confidence
        penalty_factors = 0.0
        
        # 1. Too many repeated characters or words
        if len(set(text.split())) < len(text.split()) * 0.7:  # >30% repeated words
            penalty_factors += 2.0
        
        # 2. Too short or too fragmented
        if len(words) < 3:
            penalty_factors += 3.0
        
        # 3. Non-Vietnamese characters (excluding punctuation)
        non_viet_chars = sum(1 for c in text if c.isalpha() and not (
            '\u0041' <= c <= '\u005A' or  # A-Z
            '\u0061' <= c <= '\u007A' or  # a-z
            '\u00C0' <= c <= '\u1EF9'     # Vietnamese extended
        ))
        if non_viet_chars > len(text) * 0.1:  # >10% non-Vietnamese
            penalty_factors += 1.5
        
        # Apply adjustments
        adjusted_confidence = confidence + boost_factors - penalty_factors
        
        # Ensure confidence stays in valid range with slight boost for high-quality results
        if confidence > 90 and boost_factors > penalty_factors:
            adjusted_confidence = min(99.5, adjusted_confidence)  # Cap at 99.5%
        else:
            adjusted_confidence = max(0.0, min(100.0, adjusted_confidence))
        
        return adjusted_confidence

    def chunk_and_transcribe(self, audio: NDArray[np.float32], sr: int) -> List[Tuple[str, float]]:
        """
        Chia chunks và transcribe với ensemble processing và confidence boosting
        """
        chunk_duration_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        step_samples = chunk_duration_samples - overlap_samples
        
        results = []
        prev_text = ""
        
        for i in range(0, len(audio), step_samples):
            chunk_start = i
            chunk_end = min(i + chunk_duration_samples, len(audio))
            
            if chunk_end - chunk_start < sr * 2.0:  # Skip chunks < 2 giây
                break
            
            chunk_audio = audio[chunk_start:chunk_end]
            chunk_start_time = chunk_start / sr
            
            logger.info(f"Processing chunk [{chunk_start_time:.1f}s-{chunk_end/sr:.1f}s]")
            
            try:
                if self.enable_multiple_attempts:
                    # Use ensemble processing for better confidence
                    best_text, best_confidence = self.ensemble_transcribe(chunk_audio, sr)
                else:
                    # Single attempt
                    best_text, best_confidence = self.stt_model.predict(chunk_audio, sr)
                
                if best_text.strip():
                    # Post-process confidence
                    adjusted_confidence = self.post_process_confidence(best_text.strip(), best_confidence)
                    
                    # Remove overlap với chunk trước
                    cleaned_text = self.remove_overlap_text(best_text.strip(), prev_text)
                    if cleaned_text:
                        results.append((cleaned_text, adjusted_confidence))
                        prev_text = best_text.strip()
                        logger.info(f"    Result: {cleaned_text[:30]}... (conf: {adjusted_confidence:.1f}%)")
                
            except Exception as e:
                logger.error(f"Chunk processing error: {e}")
        
        return results

    def advanced_transcribe(self, audio: NDArray[np.float32], sr: int) -> Tuple[str, float, int]:
        """
        STT nâng cao với tất cả kỹ thuật cải thiện
        """
        logger.info(f"Advanced STT Processing - duration: {len(audio)/sr:.2f}s")
        
        # Audio preprocessing
        processed_audio = audio.copy()
        
        if self.enable_normalize:
            logger.info("Normalizing audio...")
            processed_audio = self.normalize_audio(processed_audio)
        
        if self.enable_denoise:
            logger.info("Denoising audio...")
            processed_audio = self.denoise_audio(processed_audio, sr)
        
        # Strategy 1: Silence-based splitting for long audio
        if self.enable_silence_split and len(processed_audio) > sr * 60:
            logger.info("Splitting by silence...")
            segments = self.split_by_silence(processed_audio, sr)
            logger.info(f"Found {len(segments)} speech segments")
            
            if len(segments) > 1:
                # Process each segment separately
                all_results = []
                for i, segment in enumerate(segments):
                    logger.info(f"Processing segment {i+1}/{len(segments)} ({len(segment)/sr:.1f}s)")
                    if len(segment) > sr * self.chunk_duration:
                        # Further chunk large segments
                        results = self.chunk_and_transcribe(segment, sr)
                    else:
                        # Process small segments directly
                        try:
                            text, confidence = self.stt_model.predict(segment, sr)
                            results = [(text.strip(), confidence)] if text.strip() else []
                        except:
                            results = []
                    all_results.extend(results)
                
                if all_results:
                    texts = [r[0] for r in all_results]
                    confidences = [r[1] for r in all_results]
                    final_text = " ".join(texts)
                    final_confidence = np.mean(confidences)
                    return final_text, final_confidence, len(all_results)
        
        # Strategy 2: Regular chunking
        logger.info("Using regular chunking...")
        results = self.chunk_and_transcribe(processed_audio, sr)
        
        if results:
            texts = [r[0] for r in results]
            confidences = [r[1] for r in results]
            final_text = " ".join(texts)
            final_confidence = np.mean(confidences)
            return final_text, final_confidence, len(results)
        
        return "", 0.0, 0

    def process_audio_file(self, audio_file: str) -> Tuple[str, float, dict]:
        """
        Process audio file with advanced STT techniques
        """
        start_time = time.time()
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=None)
            audio = audio.astype(np.float32)
            
            # Process
            text, confidence, chunks_count = self.advanced_transcribe(audio, sr)
            
            # Calculate timing
            processing_time = time.time() - start_time
            audio_duration = len(audio) / sr
            realtime_factor = processing_time / audio_duration
            
            metadata = {
                'audio_duration': audio_duration,
                'processing_time': processing_time,
                'realtime_factor': realtime_factor,
                'chunks_processed': chunks_count,
                'sample_rate': sr
            }
            
            return text, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return "", 0.0, {'error': str(e)}

    def process_with_speech_validation(
        self, 
        audio: NDArray[np.float32], 
        sample_rate: int
    ) -> Tuple[str, float, dict]:
        """
        Xử lý audio với AST AC chỉ để validate speech, sau đó dùng chunking cố định 15s
        """
        try:
            # 1. Validate speech content using AST AC
            logger.info("Validating speech content with AST AC...")
            has_speech = self.validate_speech_content(audio, sample_rate)
            
            if not has_speech:
                logger.warning("No human speech detected in audio")
                return "", 0.0, {
                    'audio_duration': len(audio) / sample_rate,
                    'speech_detected': False,
                    'processing_method': 'speech_validation_failed'
                }
            
            # 2. Process with fixed chunking (15s chunks)
            logger.info("Speech validated ✅ - Processing with fixed 15s chunking...")
            text, confidence, chunks_count = self.advanced_transcribe(audio, sample_rate)
            
            # 3. Create metadata
            metadata = {
                'audio_duration': len(audio) / sample_rate,
                'chunks_processed': chunks_count,
                'sample_rate': sample_rate,
                'speech_detected': True,
                'processing_method': 'speech_validated_fixed_chunking'
            }
            
            logger.info(f"Speech validation + fixed chunking completed: {chunks_count} chunks")
            return text, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error in speech validation processing: {e}")
            return "", 0.0, {'error': str(e)}

    def validate_speech_content(self, audio: NDArray[np.float32], sample_rate: int) -> bool:
        """
        Sử dụng AST AC để validate xem audio có chứa tiếng người nói không
        Chỉ dùng để phát hiện speech, không dùng để segment
        """
        if not self.ast_ac_model:
            return True  # Không có AST AC thì coi như có speech
            
        try:
            # Lấy sample 5s đầu để test (hoặc toàn bộ nếu ngắn hơn)
            sample_duration = min(5.0, len(audio) / sample_rate)
            if sample_duration < 1.0:
                return True  # Audio quá ngắn thì skip validation
                
            sample_length = int(sample_duration * sample_rate)
            sample_audio = audio[:sample_length]
            
            # Resample cho AST AC nếu cần
            if sample_rate != 16000:
                sample_audio = audio_util.resample_audio(sample_audio, sample_rate, 16000)
                
            # Predict với AST AC (chỉ trả về True/False)
            is_speech = self.ast_ac_model.predict(sample_audio, 16000)
            
            logger.info(f"Speech validation result: {'✅ Speech detected' if is_speech else '❌ No speech detected'}")
            return is_speech
            
        except Exception as e:
            logger.warning(f"Speech validation failed: {e}, continuing anyway...")
            return True  # Lỗi thì coi như có speech 