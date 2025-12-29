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

def process_with_gemini(transcript_json_path, config):
    """
    Send Whisper transcript to Gemini using the google-genai SDK.
    """
    api_key = config.get("gemini_api_key")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Please set 'gemini_api_key' in config.json")

    # Initialize New SDK Client
    client = genai.Client(api_key=api_key)
    
    # User requested gemini-2.5-flash (assuming they mean a future model or 2.0)
    # As of late 2025, gemini-2.0-flash is the standard for fast/efficient.
    model_name = 'gemini-3-flash-preview' # Or config choice
    
    with open(transcript_json_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    simplified_transcript = []
    for seg in transcript_data.get("segments", []):
        simplified_transcript.append({
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "")
        })
    
    prompt = f"""
    You are an expert video editor and content curator. I will provide you with a JSON transcript of a video.
    Your task:
    1. Identify a compelling Title for the video based on the context. 
    2. Detect the Language (one of: Hindi, English, Hinglish).
    3. Extract as many as possible relevant Keywords.
    4. Select segments that comprise the "meaningful content" related to the core topic.
    5. CRITICAL: Remove all fillers, silence mentions, long unrelated personal stories (if they don't add value), and any gibberish.
    6. Ensure the final selected segments flow logically.
    7. Targeted length: The final video should ideally be 15-20 minutes in total duration.
    
    TRANSCRIPT JSON:
    {json.dumps(simplified_transcript, indent=2)}
    
    OUTPUT FORMAT (STRICT JSON):
    {{
      "title": "...",
      "detected_language": "...",
      "detected_keywords": ["...", "..."],
      "meaningful_segments": [
        {{"start": 0.0, "end": 10.5, "text": "..."}},
        ...
      ]
    }}
    """
    
    logger.info(f"Sending transcript {transcript_json_path} to Gemini ({model_name})...")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        
        refined_data = json.loads(response.text)
        
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

# --- Wrappers ---

class Stage1Wrapper:
    """Pipeline Stage 1: Video -> Transcript JSON"""
    def __init__(self, config_path="config.json"):
        os.environ["PATH"] += os.pathsep + os.getcwd()
        self.config = load_config(config_path)

    def process(self, file_path):
        file_path = Path(file_path)
        print(f"{Fore.CYAN}Stage 1 Processing: {file_path}{Style.RESET_ALL}")
        
        try:
            temp_audio = file_path.with_suffix('.wav')
            
            # 1. Extract
            if not extract_audio(str(file_path), str(temp_audio), self.config.get("enable_audio_cleaning", False)):
                return f"Failed extract: {file_path.name}"
            
            # 2. Transcribe
            try:
                result, lang = transcribe_audio(str(temp_audio), self.config.get("model_size", "small"))
            except Exception as e:
                return f"Failed transcribe: {file_path.name} - {e}"

            # 3. Save purely the transcript for next stage
            # (Already saved to .json by transcribe_audio)
            
            # Cleanup audio if needed
            if temp_audio.exists():
                try: os.remove(temp_audio)
                except: pass
                
            json_path = file_path.with_suffix('.json')
            return f"Success: Transcript saved to {json_path.name}"

        except Exception as e:
            return f"Error: {e}"

class Stage2Wrapper:
    """Pipeline Stage 2: Transcript JSON -> Gemini Refinement JSON"""
    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)

    def process(self, json_path):
        json_path = Path(json_path)
        
        if not json_path.exists():
            return f"Error: File {json_path} not found."

        try:
            refined_data, refined_json_path = process_with_gemini(str(json_path), self.config)
            
            logger.info(f"Refined JSON saved: {refined_json_path.name}")
            return f"Success: Refined segments saved to {refined_json_path.name} (Detected Title: {refined_data.get('title')})"

        except Exception as e:
            return f"Error in Stage 2 for {json_path.name}: {str(e)}"

class Stage3Wrapper:
    """Pipeline Stage 3: Refined JSON -> Cut Video (Smooth)"""
    def __init__(self, config_path="config.json"):
        os.environ["PATH"] += os.pathsep + os.getcwd()
        self.config = load_config(config_path)

    def process(self, video_path):
        video_path = Path(video_path)
        refined_json_path = video_path.with_name(f"{video_path.stem}_refined.json")
        
        if not refined_json_path.exists():
            return f"Missing refined JSON for {video_path.name}. Run Stage 2 first."

        try:
            with open(refined_json_path, 'r', encoding='utf-8') as f:
                refined_data = json.load(f)
            
            output_dir = Path("output")
            lang = refined_data.get("detected_language", "HE").upper()
            lang_dir = output_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            safe_title = "".join([c for c in refined_data.get("title", "video") if c.isalnum() or c in (' ', '_')]).rstrip()
            safe_title = safe_title.replace(' ', '_')
            final_output = lang_dir / f"{video_path.stem}_{safe_title}_refined.mp4"

            segments = refined_data.get("meaningful_segments", [])
            
            if trim_video_from_segments(str(video_path), segments, str(final_output)):
                # Cleanup: Move source files to processed folder
                processed_root = Path("processed")
                video_folder = processed_root / video_path.stem
                video_folder.mkdir(parents=True, exist_ok=True)

                try:
                    # Move Original Video
                    if video_path.exists():
                        video_path.rename(video_folder / video_path.name)
                    
                    # Move Transcript JSON (Stage 1)
                    transcript_json = video_path.with_suffix('.json')
                    if transcript_json.exists():
                        transcript_json.rename(video_folder / transcript_json.name)

                    # Move Refined JSON (Stage 2)
                    if refined_json_path.exists():
                        refined_json_path.rename(video_folder / refined_json_path.name)
                        
                    return f"Success! Video cut & source files moved to {video_folder}"
                except Exception as e:
                    return f"Success (Cut), but failed to move files: {e}"
            else:
                return f"Cutting failed for {video_path.name}"

        except Exception as e:
            return f"Error in Stage 3 for {video_path.name}: {str(e)}"

# --- Workers ---

def run_stage1_worker(file_path, config_path="config.json"):
    wrapper = Stage1Wrapper(config_path)
    return wrapper.process(file_path)

def run_stage2_worker(json_path, config_path="config.json"):
    wrapper = Stage2Wrapper(config_path)
    return wrapper.process(json_path)

def run_stage3_worker(video_path, config_path="config.json"):
    wrapper = Stage3Wrapper(config_path)
    return wrapper.process(video_path)
