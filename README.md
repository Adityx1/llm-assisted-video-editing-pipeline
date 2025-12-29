# Video Processing Pipeline

A 3-stage intelligence pipeline to automatically Transcribe, Analyze, and Trim videos using Whisper and Gemini.

## Features
- **Stage 1 (Transcribe):** Uses OpenAI Whisper to transcribe video to JSON.
- **Stage 2 (Analyze):** Uses Google Gemini (via `google-genai` SDK) to find meaningful segments, remove fillers/silence, and generate a title/keywords.
- **Stage 3 (Cut):** Uses MoviePy to trim and cross-dissolve the selected meaningful segments into a final polished video.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You also need FFmpeg installed and in your PATH.*

2. **Configuration:**
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
