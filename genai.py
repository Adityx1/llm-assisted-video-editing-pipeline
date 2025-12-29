import argparse
import multiprocessing
import json
import logging
import os # For pathsep
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from colorama import Fore, Style, init

import utils
import prompts

logger = logging.getLogger(__name__)

def process_file(json_path, config):
    """
    Stage 2 Logic:
    1. Read transcript
    2. Construct prompt
    3. Call Gemini
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        return f"Error: File {json_path} not found."

    try:
        # Prepare data for prompt
        with open(json_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        simplified_transcript = []
        for seg in transcript_data.get("segments", []):
            simplified_transcript.append({
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "")
            })
            
        # Construct Prompt
        prompt_text = prompts.GEMINI_PROMPT_TEMPLATE.format(
            transcript_json=json.dumps(simplified_transcript, indent=2)
        )
        
        # Gemini processing
        refined_data, refined_json_path = utils.process_with_gemini(str(json_path), config, prompt_text)
        
        logger.info(f"Refined JSON saved: {refined_json_path.name}")
        return f"Success: Refined segments saved to {refined_json_path.name} (Detected Title: {refined_data.get('title')})"

    except Exception as e:
        return f"Error in Stage 2 for {json_path.name}: {str(e)}"

def wrapper_process(json_path):
    config = utils.load_config("config.json")
    return process_file(json_path, config)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    init(autoreset=True)
    utils.setup_logging(is_main_process=True)

    parser = argparse.ArgumentParser(description="Stage 2 GenAI Refiner (Process JSON files with Gemini)")
    parser.add_argument("--input", "-i", type=str, default="input", help="Directory containing .json transcripts")
    parser.add_argument("--workers", "-w", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    
    # Target JSON files, but exclude ones already refined
    all_json = list(input_path.glob("*.json"))
    files = [f for f in all_json if not f.name.endswith("_refined.json")]

    if not files:
        print(f"No transcript files (.json) found in {input_path}")
        exit()

    print(f"{Fore.CYAN}Starting Stage 2 (Gemini Refinement) for {len(files)} JSON files with {args.workers} workers...{Style.RESET_ALL}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(wrapper_process, f): f for f in files}
        for future in tqdm(futures, total=len(files), desc="Processing JSONs"):
            tqdm.write(future.result())
