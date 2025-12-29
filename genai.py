import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from colorama import Fore, Style, init

import utils

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
        futures = {executor.submit(utils.run_stage2_worker, f): f for f in files}
        for future in tqdm(futures, total=len(files), desc="Processing JSONs"):
            tqdm.write(future.result())
