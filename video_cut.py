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
        futures = {executor.submit(utils.run_stage3_worker, f): f for f in files}
        for future in tqdm(futures, total=len(files), desc="Cutting"):
            tqdm.write(future.result())
