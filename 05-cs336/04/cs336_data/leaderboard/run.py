#!/usr/bin/env python3
import os
import sys
import glob
import time
from multiprocessing import Pool, cpu_count
import psutil
from tqdm import tqdm

# Add the parent directory to sys.path so we can import cs336_data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cs336_data.extract import process_warc_file


def process_single_warc_file(warc_file_path):
    """Process a single WARC file using the extract pipeline"""
    try:
        start_time = time.time()
        num_records, output_file = process_warc_file(file_path=warc_file_path, subsample_count=100)
        elapsed_time = time.time() - start_time
        return warc_file_path, num_records, output_file, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time if "start_time" in locals() else 0
        return warc_file_path, 0, None, elapsed_time


def main():
    # Find all .warc.gz files in the leaderboard/.data directory
    data_dir = os.path.join(os.path.dirname(__file__), ".data")
    warc_pattern = os.path.join(data_dir, "*.warc.gz")
    warc_files = sorted(glob.glob(warc_pattern))

    if not warc_files:
        print(f"No .warc.gz files found in {data_dir}")
        return

    print(f"Found {len(warc_files)} WARC files to process")

    # Get system resources
    num_cpus = cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)

    # Estimate memory usage per process (conservative estimate)
    # WARC files can be memory-intensive, so we'll use ~3GB per process
    memory_per_process_gb = 3
    max_processes_by_memory = int(total_memory_gb * 0.8 / memory_per_process_gb)  # Use 80% of available memory

    # Use the minimum of CPU count and memory-constrained processes
    num_processes = min(num_cpus, max_processes_by_memory, len(warc_files))

    print(f"System specs: {num_cpus} CPUs, {total_memory_gb:.1f}GB RAM")
    print(f"Using {num_processes} parallel processes")

    # Process files in parallel with progress tracking
    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for progress tracking
        results = []
        with tqdm(total=len(warc_files), desc="Processing WARC files", unit="file") as pbar:
            for result in pool.imap_unordered(process_single_warc_file, warc_files):
                results.append(result)
                warc_file, num_records, output_file, file_time = result

                # Update progress bar with file info
                file_name = os.path.basename(warc_file)
                if output_file:
                    pbar.set_postfix(
                        {
                            "records": num_records,
                            "time": f"{file_time:.1f}s",
                            "file": file_name[:15] + "..." if len(file_name) > 15 else file_name,
                        }
                    )
                else:
                    pbar.set_postfix(
                        {"status": "FAILED", "file": file_name[:15] + "..." if len(file_name) > 15 else file_name}
                    )
                pbar.update(1)

    total_time = time.time() - start_time

    # Summary statistics
    total_records = 0
    successful_files = 0
    total_processing_time = 0

    for warc_file, num_records, output_file, file_time in results:
        if output_file:
            total_records += num_records
            successful_files += 1
        total_processing_time += file_time

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Successfully processed: {successful_files}/{len(warc_files)} files")
    print(f"Total records extracted: {total_records:,}")
    print(f"Wall clock time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Total processing time: {total_processing_time:.1f}s")
    print(f"Parallelization efficiency: {total_processing_time / total_time:.1f}x")
    if successful_files > 0:
        print(f"Average time per file: {total_processing_time / successful_files:.1f}s")
        print(f"Average records per file: {total_records / successful_files:.0f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    # 50 files, 5k records, fuck me
    # 151 seconds, 45 seconds per file, with 32 nprocs running, 1 takes an outsized amount, 5 seconds per file
    # 1. filters are too high?
    # ~ like 100 records out of 24k is absurd.
    # [ ] what's cutting the most? is it paloma? most likely? reduce threshold? reduce other thresholds?
    # ~ can't process more than 10TB, current 100/24000, lol 100M files, wtf 100
    # ~ nvm, was not considering 100 * tokens, is not tokens, count again and run again ~ somewhere around 700k chars per file * 50 / 4
    # ~ about 175k tokens per warc file
    # ~ it'd require about 30000 files, I need at least 3 to 5 x more per file
    # ~ if I get a 128 nprocs it should be way way faster.
    # # let's say I keep it at 100 ratio, 30000 files ≈ 30TB ≈ about 3h, not terrible at all, WTF?
    # TODO: yea do it, but first, improve this a bit, save .warc.gz in a separate folder, same for .txt, then maybe download + process then remove file
    # TODO: vibe check results, minor script to move them into a single .txt, and provide some general stats with tokenizer.
    # - yea need to delete them once processed, cause the machine has max 6TB of space
