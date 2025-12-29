import argparse
import multiprocessing
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from colorama import Fore, Style, init

import utils

def process_file(video_path, config):
    """
    Stage 3 Logic:
    1. Find refined JSON
    2. Determine output path
    3. Cut video
    4. Move source files to processed
    """
    video_path = Path(video_path)
    refined_json_path = video_path.with_name(f"{video_path.stem}_refined.json")
    
    if not refined_json_path.exists():
        return f"Missing refined JSON for {video_path.name}. Run Stage 2 first."

    try:
        with open(refined_json_path, 'r', encoding='utf-8') as f:
            refined_data = json.load(f)
            
        if isinstance(refined_data, list):
            refined_data = refined_data[0] if refined_data else {}
        
        output_dir = Path("output")
        lang = refined_data.get("detected_language", "HE").upper()
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        safe_title = "".join([c for c in refined_data.get("title", "video") if c.isalnum() or c in (' ', '_')]).rstrip()
        safe_title = safe_title.replace(' ', '_')
        final_output = lang_dir / f"{video_path.stem}_{safe_title}_refined.mp4"

        segments = refined_data.get("meaningful_segments", [])
        
        if utils.trim_video_from_segments(str(video_path), segments, str(final_output)):
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

def wrapper_process(video_path):
    os.environ["PATH"] += os.pathsep + os.getcwd()
    config = utils.load_config("config.json")
    return process_file(video_path, config)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    init(autoreset=True)
    utils.setup_logging(is_main_process=True)

    parser = argparse.ArgumentParser(description="Stage 3 Video Processor (Cutting Only)")
    parser.add_argument("--input", "-i", type=str, default="input")
    parser.add_argument("--workers", "-w", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input)
    all_videos = list(input_path.glob("*.mp4"))
    # Target videos that have a _refined.json transcript
    files = [v for v in all_videos if v.with_name(f"{v.stem}_refined.json").exists()]

    if not files:
        print("No videos with refined transcripts (_refined.json) found in input folder.")
        exit()

    print(f"{Fore.CYAN}Starting Stage 3 (Cutting) for {len(files)} files with {args.workers} workers...{Style.RESET_ALL}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(wrapper_process, f): f for f in files}
        for future in tqdm(futures, total=len(files), desc="Cutting"):
            tqdm.write(future.result())
