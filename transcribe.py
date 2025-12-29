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
        futures = {executor.submit(utils.run_stage1_worker, f): f for f in files}
        for future in tqdm(futures, total=len(files), desc="Processing"):
            tqdm.write(future.result())
