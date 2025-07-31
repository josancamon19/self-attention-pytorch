#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from multiprocessing import Pool, cpu_count, Process
from concurrent.futures import ThreadPoolExecutor
import psutil
from tqdm import tqdm
from pathlib import Path

# Add the parent directory to sys.path so we can import cs336_data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cs336_data.extract import process_warc_file


def download_process_delete_warc(url_info, delete_downloads=True):
    """Download, process, and optionally delete a single WARC file with caching checks"""
    index, warc_path = url_info
    url = f"https://data.commoncrawl.org/{warc_path}"

    start_time = time.time()

    # Extract filename from warc_path with timestamp to avoid collisions
    # e.g., "crawl-data/CC-MAIN-2025-30/segments/1751905933612.63/warc/CC-MAIN-20250707183638-20250707213638-00000.warc.gz"
    # becomes "20250707213638-00000.warc.gz" (end timestamp + file number)
    full_filename = warc_path.split("/")[-1]  # "CC-MAIN-20250707183638-20250707213638-00000.warc.gz"
    parts = full_filename.split("-")  # ["CC", "MAIN", "20250707183638", "20250707213638", "00000.warc.gz"]
    filename = f"{parts[-2]}-{parts[-1]}"  # "20250707213638-00000.warc.gz"
    base_name = filename.replace(".warc.gz", "")  # "20250707213638-00000"

    # Check paths
    script_dir = Path(__file__).parent
    download_path = script_dir / ".data/warcs" / filename
    processed_dir = script_dir / ".data/processed"

    # Check if already processed by looking for output file
    potential_output = processed_dir / f"{base_name}.txt"
    if potential_output.exists():
        # File already processed, read record count from file
        try:
            with open(potential_output) as f:
                content = f.read()
                # Count records by splitting on <|endoftext|> separator
                num_records = len([x for x in content.split("<|endoftext|>") if x.strip()])
            elapsed_time = time.time() - start_time
            return filename, num_records, str(potential_output), elapsed_time
        except Exception:
            # If we can't read the processed file, reprocess
            pass

    # Check if file is already downloaded
    temp_path = None
    if download_path.exists() and download_path.stat().st_size >= 900 * 1024 * 1024:
        # Use existing downloaded file
        file_to_process = str(download_path)
    else:
        # Download to .data/warcs directory
        download_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = str(download_path)

        try:
            cmd = ["wget", "-O", temp_path, url]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            file_to_process = temp_path
        except subprocess.CalledProcessError:
            elapsed_time = time.time() - start_time
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return filename, 0, None, elapsed_time

    try:
        # Process file with high-memory mode for speed
        target_output_file = processed_dir / f"{base_name}.txt"
        num_records, output_file = process_warc_file(
            file_path=file_to_process,
            target_output_path=str(target_output_file),
            # subsample_count=100,
            low_ram_usage=False,  # Use high-memory mode with 566GB RAM
        )

        elapsed_time = time.time() - start_time
        return filename, num_records, output_file, elapsed_time

    except Exception:
        elapsed_time = time.time() - start_time
        return filename, 0, None, elapsed_time
    finally:
        # Remove downloaded file if it exists to save space (if enabled)
        if delete_downloads and download_path.exists():
            try:
                os.remove(download_path)
            except OSError:
                pass


def download_files_ahead(url_infos, script_dir, start_after=None):
    """Download files ahead of time in a separate process"""
    download_dir = script_dir / ".data/warcs"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Start downloading after the initial batch if specified
    urls_to_download = url_infos[start_after:] if start_after else url_infos

    print(f"[DOWNLOADER] Starting to download {len(urls_to_download)} files ahead...")
    downloaded_count = 0
    skipped_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        for _, warc_path in urls_to_download:
            # Extract filename with timestamp to avoid collisions
            full_filename = warc_path.split("/")[-1]
            parts = full_filename.split("-")
            filename = f"{parts[-2]}-{parts[-1]}"  # "20250707213638-00000.warc.gz"
            base_name = filename.replace(".warc.gz", "")
            download_path = download_dir / filename

            # Check if already processed - skip download entirely
            processed_output = script_dir / ".data/processed" / f"{base_name}.txt"
            if processed_output.exists():
                skipped_count += 1
                continue

            # Skip if already downloaded
            if download_path.exists():
                skipped_count += 1
                continue

            # Download file
            url = f"https://data.commoncrawl.org/{warc_path}"

            def download_single(url, path, filename):
                try:
                    cmd = ["wget", "-O", str(path), url]
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    return True, filename
                except subprocess.CalledProcessError:
                    # Remove failed download
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    return False, filename

            future = executor.submit(download_single, url, download_path, filename)
            futures.append(future)

        # Track completed downloads
        for future in futures:
            success, filename = future.result()
            if success:
                downloaded_count += 1
                if downloaded_count % 10 == 0:  # Log every 10 downloads
                    print(f"[DOWNLOADER] Downloaded {downloaded_count} files ahead (skipped {skipped_count} existing)")

    print(f"[DOWNLOADER] Finished: {downloaded_count} downloaded, {skipped_count} skipped")


def main(target_count: int = 100, delete_downloads: bool = True):
    warc_paths_file = Path(__file__).parent / "data" / "warc.paths"
    with open(warc_paths_file) as f:
        warc_paths = [line.strip() for line in f.readlines()]

    warc_paths = warc_paths[:target_count]
    url_infos = [(i, path) for i, path in enumerate(warc_paths)]
    print(f"Processing {len(url_infos)} WARC files from URLs")

    # Ensure directories exist
    script_dir = Path(__file__).parent
    (script_dir / ".data/warcs").mkdir(parents=True, exist_ok=True)
    (script_dir / ".data/processed").mkdir(parents=True, exist_ok=True)

    # Get system resources
    num_cpus = cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)

    # Optimize for 288 CPUs and 566GB RAM - use more processes
    # 2.5GB per process allows for 215 processes using 537GB RAM
    memory_per_process_gb = 2
    max_processes_by_memory = int(total_memory_gb * 0.95 / memory_per_process_gb)  # Use 95% of available memory

    # Use the minimum of CPU count and memory-constrained processes
    num_processes = min(num_cpus, max_processes_by_memory, len(url_infos))

    print(f"System specs: {num_cpus} CPUs, {total_memory_gb:.1f}GB RAM")
    print(f"Using {num_processes} parallel processes")
    print("Each process will: check cache -> download (if needed) -> process -> cleanup")

    # Start background downloader process after initial batch
    downloader_process = Process(target=download_files_ahead, args=(url_infos, script_dir, num_processes))
    # downloader_process.daemon = True # bunch of race conditions cases to manage
    # downloader_process.start()
    print(f"Started background downloader for files {num_processes}+")

    # Process files in parallel with progress tracking
    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for progress tracking
        results = []
        with tqdm(total=len(url_infos), desc="Download+Process WARC files", unit="file") as pbar:
            # Create partial function with delete_downloads parameter
            from functools import partial

            process_func = partial(download_process_delete_warc, delete_downloads=delete_downloads)
            for result in pool.imap_unordered(process_func, url_infos):
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
    print(f"Successfully processed: {successful_files}/{len(url_infos)} files")
    print(f"Total records extracted: {total_records:,}")
    print(f"Wall clock time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Total processing time: {total_processing_time:.1f}s")
    print(f"Parallelization efficiency: {total_processing_time / total_time:.1f}x")
    if successful_files > 0:
        print(f"Average time per file: {total_processing_time / successful_files:.1f}s")
        print(f"Average records per file: {total_records / successful_files:.0f}")
    print(f"{'=' * 60}")

    # Clean up downloader process
    if downloader_process.is_alive():
        downloader_process.terminate()
        downloader_process.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process WARC files with caching")
    parser.add_argument("--count", type=int, default=100, help="Number of files to process (default: 100)")
    parser.add_argument(
        "--keep-downloads", action="store_true", help="Keep downloaded WARC files instead of deleting them"
    )

    args = parser.parse_args()

    main(target_count=args.count, delete_downloads=not args.keep_downloads)
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
    # TODO: vibe check results, minor script to move them into a single .txt, and provide some general stats with tokenizer.
    # - yea need to delete them once processed, cause the machine has max 6TB of space
    # ~
    # deduplicate call after first 8000 files processed
    # - then, check how many tokens are there, compute some thoughts with it
    # - create a script to tokenize into npy file or however format
    # - trigger a partial run with less steps than mentioned.
    # - extract the remaining needed data
