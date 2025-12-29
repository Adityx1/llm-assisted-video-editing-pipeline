import argparse
import multiprocessing
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from colorama import Fore, Style, init

import utils

def process_file(file_path, config):
    """
    Stage 1 Logic:
    1. Extract audio
    2. Transcribe
    3. Cleanup audio
    """
    file_path = Path(file_path)
    # print(f"{Fore.CYAN}Stage 1 Processing: {file_path}{Style.RESET_ALL}")
    
    try:
        temp_audio = file_path.with_suffix('.wav')
        
        # 1. Extract
        if not utils.extract_audio(str(file_path), str(temp_audio), config.get("enable_audio_cleaning", False)):
            return f"Failed extract: {file_path.name}"
        
        # 2. Transcribe
        try:
            result, lang = utils.transcribe_audio(str(temp_audio), config.get("model_size", "small"))
        except Exception as e:
            return f"Failed transcribe: {file_path.name} - {e}"

        # 3. Cleanup audio if needed
        if temp_audio.exists():
            try: os.remove(temp_audio)
            except: pass
            
        json_path = file_path.with_suffix('.json')
        return f"Success: Transcript saved to {json_path.name}"

    except Exception as e:
        return f"Error: {e}"

def wrapper_process(file_path):
    """Wrapper to load config inside the worker process."""
    # Ensure FFMPEG in PATH for the worker
    os.environ["PATH"] += os.pathsep + os.getcwd()
    config = utils.load_config("config.json")
    return process_file(file_path, config)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    init(autoreset=True)
    utils.setup_logging(is_main_process=True)

    parser = argparse.ArgumentParser(description="Stage 1 Video Processor (Transcription Only)")
    parser.add_argument("--input", "-i", type=str, default="input")
    parser.add_argument("--workers", "-w", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input)
    files = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.mp4"))

    if not files:
        print("No files found.")
        exit()

    print(f"{Fore.CYAN}Starting Stage 1 processing for {len(files)} files with {args.workers} workers...{Style.RESET_ALL}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(wrapper_process, f): f for f in files}
        for future in tqdm(futures, total=len(files), desc="Processing"):
            tqdm.write(future.result())
