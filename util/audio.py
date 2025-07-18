import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Tuple, Union
import io


def resample_audio(audio: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    """
    Resamples the given audio signal from the original sample rate to the target sample rate.

    **Parameters:**
    - `audio` (NDArray): Input audio signal as a NumPy array.
    - `orig_sr` (int): Original sample rate of the audio.
    - `target_sr` (int): Target sample rate to resample the audio.

    **Returns:**
    - `NDArray`: The resampled audio signal.
    """
    if orig_sr == target_sr:
        return audio

    gcd = np.gcd(orig_sr, target_sr)

    up = target_sr // gcd
    down = orig_sr // gcd

    resample_audio = signal.resample_poly(audio, up, down)

    return resample_audio

def calculate_rms(audio: NDArray[np.float32], eps=1e-10) -> float:
    """
    Calculate the Root Mean Square (RMS) of an audio signal.

    **Parameters:**
    - `audio` (NDArray[np.float32]): Input audio signal as a NumPy array.
    - `eps` (float): A small value to avoid division by zero (default: 1e-10).

    **Returns:**
    - `float`: The RMS value of the audio signal.
    """
    rms = np.sqrt(np.mean(np.square(audio)))
    return max(rms, eps)

def calculate_dbfs(audio: NDArray[np.float32], eps=1e-10) -> float:
    """
    Calculates the decibels relative to full scale (dBFS) of an audio signal.

    **Parameters:**
    - `audio` (NDArray[np.float32]): Input audio signal as a NumPy array.
    - `eps` (float): A small value to avoid log of zero errors (default: 1e-10).

    **Returns:**
    - `float`: The dBFS value of the audio signal.
    """
    rms = calculate_rms(audio, eps)
    dbfs = 20 * np.log10(rms)
    return dbfs

def load_and_extract_left_channel(
    file_input: Union[str, io.BytesIO], 
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[NDArray[np.float32], int]:
    """
    Load audio file (any format), extract left channel (mono), and return as numpy array.
    
    This function combines:
    - Format conversion (CGI, MP3, WAV, etc. -> internal processing)
    - Stereo to mono conversion (extract left channel only)
    - Optional resampling to target sample rate
    - Memory-only processing (no temporary files)
    
    **Parameters:**
    - `file_input` (Union[str, io.BytesIO]): Path to input audio file OR BytesIO object with audio data
    - `target_sr` (int): Target sample rate (default: 16000 for STT models)
    - `normalize` (bool): Whether to normalize audio to [-1, 1] range
    
    **Returns:**
    - `Tuple[NDArray[np.float32], int]`: (audio_data, sample_rate)
        - audio_data: Left channel audio as float32 numpy array
        - sample_rate: Actual sample rate of the returned audio
    
    **Examples:**
    ```python
    # Load CGI file and extract left channel for STT
    audio, sr = load_and_extract_left_channel("call_recording.cgi")
    
    # Load from memory (for webhook)
    import io
    audio_bytes = io.BytesIO(file_content)
    audio, sr = load_and_extract_left_channel(audio_bytes)
    ```
    """
    try:
        from pydub import AudioSegment
        
        # Load audio - from file path or BytesIO
        if isinstance(file_input, str):
            audio_segment = AudioSegment.from_file(file_input)
        else:
            # BytesIO object - seek to beginning
            file_input.seek(0)
            audio_segment = AudioSegment.from_file(file_input)
        original_sr = audio_segment.frame_rate
        
        # Extract left channel (mono)
        if audio_segment.channels == 1:
            mono_segment = audio_segment
        elif audio_segment.channels == 2:
            mono_segment = audio_segment.split_to_mono()[0]  # Left channel
        else:
            mono_segment = audio_segment.split_to_mono()[0]  # First channel
        
        # Convert to numpy array
        # pydub returns int16 by default, convert to float32
        audio_data = np.array(mono_segment.get_array_of_samples(), dtype=np.float32)
        
        # Normalize int16 range to [-1, 1]
        if mono_segment.sample_width == 2:  # 16-bit
            audio_data = audio_data / 32768.0
        elif mono_segment.sample_width == 3:  # 24-bit  
            audio_data = audio_data / 8388608.0
        elif mono_segment.sample_width == 4:  # 32-bit
            audio_data = audio_data / 2147483648.0
        
        # Optional normalization
        if normalize:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95  # Leave some headroom
        
        # Resample if needed
        current_sr = original_sr
        if target_sr is not None and target_sr != original_sr:
            audio_data = resample_audio(audio_data, original_sr, target_sr)
            current_sr = target_sr
        
        return audio_data.astype(np.float32), current_sr
        
    except ImportError:
        raise ImportError("pydub is required. Install with: pip install pydub")
    except Exception as e:
        file_desc = file_input if isinstance(file_input, str) else "memory buffer"
        raise RuntimeError(f"Failed to load audio from {file_desc}: {e}")

def load_and_extract_both_channels(
    file_path: str,
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[NDArray[np.float32], NDArray[np.float32], int]:
    """
    Load audio file and extract both channels separately (agent/customer).
    
    **Parameters:**
    - `file_path` (str): Path to input stereo audio file
    - `target_sr` (int): Target sample rate (default: 16000)
    - `normalize` (bool): Whether to normalize audio
    
    **Returns:**
    - `Tuple[NDArray[np.float32], NDArray[np.float32], int]`: 
        (left_channel, right_channel, sample_rate)
    """
    try:
        from pydub import AudioSegment
        
        audio_segment = AudioSegment.from_file(file_path)
        
        if audio_segment.channels < 2:
            raise ValueError("Audio file must be stereo (2 channels) for dual channel extraction")
        
        original_sr = audio_segment.frame_rate
        
        # Split stereo to mono channels
        left_segment, right_segment = audio_segment.split_to_mono()
        
        # Convert to numpy arrays
        left_data = np.array(left_segment.get_array_of_samples(), dtype=np.float32)
        right_data = np.array(right_segment.get_array_of_samples(), dtype=np.float32)
        
        # Normalize from int16 range
        if audio_segment.sample_width == 2:
            left_data = left_data / 32768.0
            right_data = right_data / 32768.0
        
        # Optional normalization
        if normalize:
            for channel_data in [left_data, right_data]:
                max_val = np.max(np.abs(channel_data))
                if max_val > 0:
                    channel_data *= (0.95 / max_val)
        
        # Resample if needed
        current_sr = original_sr
        if target_sr is not None and target_sr != original_sr:
            left_data = resample_audio(left_data, original_sr, target_sr)
            right_data = resample_audio(right_data, original_sr, target_sr)
            current_sr = target_sr
        
        return left_data.astype(np.float32), right_data.astype(np.float32), current_sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load stereo audio {file_path}: {e}")