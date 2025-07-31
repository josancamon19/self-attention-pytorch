import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download_warc_files(target_count: int, max_workers: int = 8) -> None:
    """
    Download WARC files in parallel from warc.paths file.

    Args:
        target_count: Number of files to download
        max_workers: Maximum number of parallel downloads
    """
    warc_paths_file = Path(__file__).parent / "data" / "warc.paths"
    output_dir = Path(__file__).parent / ".data/warcs"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Read WARC paths
    with open(warc_paths_file) as f:
        paths = [line.strip() for line in f.readlines()]

    # Limit to target count
    paths = paths[:target_count]

    print(f"Downloading {len(paths)} WARC files...")

    def download_file(index: int, path: str) -> tuple[int, bool, str]:
        """Download a single WARC file."""
        url = f"https://data.commoncrawl.org/{path}"
        output_file = output_dir / f"{index:05d}.warc.gz"

        try:
            cmd = ["wget", "-O", str(output_file), url]
            subprocess.run(cmd, capture_output=False, text=False, check=True)
            return index, True, f"Downloaded {output_file.name}"
        except subprocess.CalledProcessError as e:
            return index, False, f"Failed to download {output_file.name}: {e}"

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, i, path) for i, path in enumerate(paths)]

        completed = 0
        for future in as_completed(futures):
            index, success, message = future.result()
            completed += 1
            status = "" if success else ""
            print(f"[{completed}/{len(paths)}] {status} {message}")

    print(f"Download complete! Files saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download WARC files from Common Crawl")
    parser.add_argument("count", type=int, help="Number of files to download")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")

    args = parser.parse_args()
    download_warc_files(args.count, args.workers)
