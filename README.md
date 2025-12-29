# Video Processing Pipeline

A 3-stage intelligence pipeline to automatically Transcribe, Analyze, and Trim videos using Whisper and Gemini.

## Features
- **Stage 1 (Transcribe):** Uses OpenAI Whisper to transcribe video to JSON.
- **Stage 2 (Analyze):** Uses Google Gemini (via `google-genai` SDK) to find meaningful segments, remove fillers/silence, and generate a title/keywords.
- **Stage 3 (Cut):** Uses MoviePy to trim and cross-dissolve the selected meaningful segments into a final polished video.

## Setup

### 1. Install FFmpeg (Required)
The pipeline requires FFmpeg to be installed and accessible in your system PATH.

#### Windows
1.  Download the **essential build** from [gyan.dev](https://www.gyan.dev/tt/ffmpeg/git-essentials.7z).
2.  Extract the `.7z` file.
3.  Copy `ffmpeg.exe`, `ffplay.exe`, and `ffprobe.exe` from the `bin` folder.
4.  Paste them into the root of this project folder (where `utils.py` is).
    *   *Alternatively, add the `bin` folder to your System PATH to make it accessible globally.*

#### Mac (via Homebrew)
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update && sudo apt install ffmpeg
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `config.json` file in the root directory:
```json
{
    "gemini_api_key": "YOUR_GEMINI_API_KEY",
    "model_size": "small",
    "enable_audio_cleaning": true
}
```

## Usage

### 1. Transcribe (Stage 1)
Scan `input/` folder for `.mp4` files and generate transcriptions.
```bash
python transcribe.py
```

### 2. Analyze & Refine (Stage 2)
Send transcriptions to Gemini to identify best segments.
```bash
python genai.py
```

### 3. Cut Video (Stage 3)
Trim the original video based on Gemini's selected segments.
```bash
python video_cut.py
```
