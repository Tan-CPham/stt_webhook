# Speech-to-Text Webhook

A backend system for Speech-to-Text (STT) processing, featuring a webhook API for seamless integration with ERP systems. Built with FastAPI and AI models.

## System Requirements

- Python 3.10+ and < 3.12
- CUDA-compatible GPU (Optional, for accelerated processing)
- RAM: 4GB Minimum
- Disk Space: 5GB+ for model storage

## Installation

### 1. Clone repository
```bash
git clone https://github.com/Tan-CPham/stt_webhook
cd speech_to_text
```

### 2. Install FFmpeg

A backend system for Speech-to-Text (STT) processing, which uses a webhook API to automatically push results to your applications. Built with FastAPI and AI models.

#### Method 1: Using a Package Manager (Recommended)

This is the easiest and most recommended way to install FFmpeg.

*   **On Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

*   **On macOS (using [Homebrew](https://brew.sh/)):**
    ```bash
    brew install ffmpeg
    ```

*   **On Windows (using [Winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) or [Chocolatey](https://chocolatey.org/)):**
    ```powershell
    # Using Winget (recommended, built-in on Windows 11)
    winget install "FFmpeg (Gyan.dev)"

    # Or using Chocolatey
    choco install ffmpeg
    ```

#### Method 2: Manual Installation

Use this method if you cannot use a package manager or need a specific version.

1.  **Download a pre-built binary** for your OS from a trusted source like [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases).
2.  **Extract the archive** to a permanent location on your computer (e.g., `C:\ffmpeg` on Windows, or `/opt/ffmpeg` on Linux).
3.  **Add the `bin` directory** inside the extracted folder to your system's `PATH` environment variable so you can run `ffmpeg` from any terminal. (See platform-specific instructions below).

---

**To verify your installation, open a *new* terminal window and run:**
```bash
ffmpeg -version

### 3. Create a Virtual Environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Create the .env File
Create a file named .env in the root directory with the following environment variables:

```env
# Directory to store downloaded models
WEIGHT_DIR=weight

# Hugging Face Token (optional, for private models)
HF_TOKEN=your_huggingface_token_here

# Model paths
STT_MODEL_WAV2VEC2_PATH=nguyenvulebinh/wav2vec2-bartpho
FEATURE_EXTRACTOR_PATH=nguyenvulebinh/wav2vec2-bartpho
TOKENIZER_PATH=nguyenvulebinh/wav2vec2-bartpho

# AST AC Model
AST_AC_MODEL_REPO_ID=MIT/ast-finetuned-audioset-10-10-0.4593
AST_AC_MODEL_REVISION=f826b80d28226b62986cc218e5cec390b1096902

# Device configuration
DEVICE=cuda  # or cpu
```

### 2. File .env.example
Refer to the .env.example file for a sample configuration.

## Usage

### Running the Server
```bash
python main.py
```

The server will:
1. Initialize the STT service.
2. Start the webhook server at http://localhost:8000.

## API Endpoints

### POST /webhook/process-audio

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  ```
  audio_file: File (required) - The audio file (.cgi, .wav, .mp3, .m4a).
  timestamp: string (optional) - The request timestamp.
  metadata: string (optional) - Additional metadata as a JSON string.
  ```

**Response:**
```json
{
  "status": "success",
  "message": "Audio processed successfully",
  "result": {
    "text": "....",
    "confidence": 0.95,
    "duration": 5.2,
    "processing_time": 1.8
  },
  "error_detail": null
}
```

### GET /health

Checks the health status of the server.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Development

### Local Development with ngrok
To test the webhook with an external ERP system:

1. Install ngrok:
```bash
# Download https://ngrok.com/
# Or use a package manager
```

2. Expose local server:
```bash
ngrok http 8000
```

3. Use the public ngrok URL as the webhook endpoint in your ERP system.
```bash
https://..........a.ngrok-free.app -> http://localhost:8000
```

### Testing
```bash
# Test with curl
curl -X POST "http://localhost:8000/webhook/process-audio" \
  -F "audio_file=@path/to/audio.wav" \
  -F "timestamp=2024-01-01T12:00:00Z"
```

## Directory Structure

```
speech_to_text/
├── main.py
├── logger_config.py
├── .env
├──api
│   └── http
│        └── webhook.py
├── model/
│   ├── download_model.py
│   ├── stt.py
│   └── ac.py
├── service/
│   └── stt.py
├── util/
│   └── audio.py  
├── weights/          
└── audio/ 
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce the processing batch size or switch to CPU
   - Setting DEVICE=cpu in your .env file.

2. **Model download failed**
   - Check your internet connection
   - Verify Hugging Face token

3. **Audio format not supported**
   - Check format file (.cgi, .wav, .mp3, .m4a)
   - Verify file is not corrupted

4. **Environment variables missing**
   - Make sure the .env file exists
   - Verify that all required variables are set

## Support

For direct support, contact me via email at: tanpham090921@gmail.com
