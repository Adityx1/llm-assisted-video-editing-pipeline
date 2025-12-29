import os
import json
import logging
from pathlib import Path
import re
import time
import ffmpeg
import whisper
from google import genai
from moviepy import VideoFileClip, concatenate_videoclips, vfx
from colorama import Fore, Style

logger = logging.getLogger(__name__)

def setup_logging(is_main_process=False):
    """Configure logging for the application."""
    handlers = [logging.StreamHandler()]
    if is_main_process:
        handlers.append(logging.FileHandler("pipeline.log", encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

def load_config(path):
    """Load configuration from JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def filter_hallucinations(text):
    """Filter out common Whisper hallucinations."""
    if not text:
        return ""
    garbage_pattern = r'[ḍὁ\s]+'
    if re.fullmatch(garbage_pattern, text):
        return ""
    return text

def extract_audio(video_path, audio_path, enable_cleaning=False):
    """Extract audio from video using FFmpeg."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, ac=1, ar=16000) # Mono, 16kHz for Whisper
            .overwrite_output()
            .run(quiet=True)
        )
        
        if enable_cleaning:
            clean_audio = str(Path(audio_path).with_suffix('.clean.wav'))
            try:
                (
                    ffmpeg
                    .input(audio_path)
                    .filter('highpass', f='200')
                    .filter('lowpass', f='3000')
                    .filter('dynaudnorm')
                    .output(clean_audio)
                    .overwrite_output()
                    .run(quiet=True)
                )
                # Retry loop for Windows file locking
                for _ in range(3):
                    try:
                        time.sleep(0.5)
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                        os.rename(clean_audio, audio_path)
                        break
                    except PermissionError:
                        continue
            except ffmpeg.Error:
                pass
        
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg extract error: {e.stderr.decode() if e.stderr else str(e)}")
        return False

def transcribe_audio(audio_path, model_size="small", device="cpu"):
    """Transcribe audio using Whisper."""
    try:
        json_cache = Path(audio_path).with_suffix('.json')
        if json_cache.exists():
            try:
                with open(json_cache, 'r') as f:
                    data = json.load(f)
                    return data, data["language"]
            except:
                pass

        logger.info(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size, device=device)
        
        logger.info(f"Transcribing {audio_path}...")
        # verbose=True gives live terminal output per segment
        result = model.transcribe(audio_path, word_timestamps=True, verbose=True)
        
        for segment in result["segments"]:
            segment["text"] = filter_hallucinations(segment["text"])
            
        with open(json_cache, 'w') as f:
            json.dump(result, f)
            
        return result, result["language"]
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise

def process_with_gemini(transcript_json_path, config, prompt_text):
    """
    Send Whisper transcript to Gemini using the google-genai SDK.
    Accepts the fully constructed prompt text.
    """
    api_key = config.get("gemini_api_key")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Please set 'gemini_api_key' in config.json")

    # Initialize New SDK Client
    client = genai.Client(api_key=api_key)
    
    model_name = 'gemini-3-flash-preview' # Or config choice
    
    logger.info(f"Sending transcript {transcript_json_path} to Gemini ({model_name})...")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config={'response_mime_type': 'application/json'}
        )
        
        refined_data = json.loads(response.text)
        
        # Handle case where Gemini returns a list
        if isinstance(refined_data, list):
            if len(refined_data) > 0:
                refined_data = refined_data[0]
            else:
                refined_data = {}
        
        # Save refined JSON
        refined_path = Path(transcript_json_path).with_name(f"{Path(transcript_json_path).stem}_refined.json")
        with open(refined_path, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, indent=2)
            
        return refined_data, refined_path
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        if hasattr(e, 'response'):
             logger.error(f"Response: {e.response}")
        raise

def trim_video_from_segments(video_path, segments, output_path):
    """
    Use MoviePy to cut and concatenate the meaningful segments with cross-dissolve transitions.
    """
    if not segments:
        logger.warning("No segments found to keep!")
        return False

    logger.info(f"Trimming video based on {len(segments)} segments using MoviePy...")
    
    clips = []
    video = None
    try:
        video = VideoFileClip(video_path)
        
        for i, seg in enumerate(segments):
            start = seg.get("start")
            end = seg.get("end")
            
            if start is None or end is None or (end - start) < 0.5:
                continue
            
            # Create subclip
            clip = video.subclipped(start, end)
            
            # Apply fade in to all clips except the very first one
            if i > 0:
                clip = clip.with_effects([vfx.CrossFadeIn(0.5)])
            
            clips.append(clip)

        if not clips:
            logger.warning("No valid clips created.")
            return False

        # Concatenate with 'compose' method to handle the crossfades
        final_clip = concatenate_videoclips(clips, method="compose", padding=-0.5)
        
        # Write output
        final_clip.write_videofile(
            output_path, 
            codec='libx264', 
            audio_codec='aac', 
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True,
            logger=None # Suppress MoviePy's standard progress bar
        )
        
        # Close clips to release resources
        for clip in clips:
            clip.close()
        video.close()
        
        return True

    except Exception as e:
        logger.error(f"MoviePy trim error: {e}")
        if video: video.close()
        return False