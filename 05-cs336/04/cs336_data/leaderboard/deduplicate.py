from cs336_data.filters.g_deduplication import exact_deduplication, minhash_deduplication
from pathlib import Path


def count_directory_chars(directory_path):
    """Count total characters, records, and files in a directory of .txt files"""
    directory = Path(directory_path)
    if not directory.exists():
        return 0, 0, 0, f"Directory {directory_path} does not exist"

    txt_files = list(directory.glob("*.txt"))
    if not txt_files:
        return 0, 0, 0, f"No .txt files found in {directory_path}"

    total_chars = 0
    total_records = 0
    for txt_file in txt_files:
        try:
            with open(txt_file, encoding="utf-8") as f:
                content = f.read()
                total_chars += len(content)
                # Count records by splitting on <|endoftext|> separator
                records = [x for x in content.split("<|endoftext|>") if x.strip()]
                total_records += len(records)
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")

    return total_chars, total_records, len(txt_files), None


def analyze_deduplication_impact(script_dir):
    """Analyze character counts, records, and token estimates across processed, exact dedup, and fuzzy dedup directories"""
    print(f"\n{'=' * 80}")
    print("DEDUPLICATION IMPACT ANALYSIS")
    print(f"{'=' * 80}")

    directories = [
        (".data/processed", "Original Processed"),
        (".data/exact_deduplicated", "After Exact Dedup"),
        (".data/fuzzy_deduplicated", "After Fuzzy Dedup"),
    ]

    results = []
    for dir_name, label in directories:
        dir_path = script_dir / dir_name
        chars, records, files, error = count_directory_chars(dir_path)

        if error:
            print(f"{label}: {error}")
            results.append((label, 0, 0, 0))
        else:
            tokens_estimate = chars // 4  # Rough estimate: chars / 4 ≈ tokens
            print(f"{label}:")
            print(f"  - {chars:,} characters")
            print(f"  - {records:,} records")
            print(f"  - {files} files")
            print(f"  - ~{tokens_estimate:,} tokens (estimated)")
            print()
            results.append((label, chars, records, files))

    # Calculate reduction percentages
    if len(results) >= 2 and results[0][1] > 0:  # Original has content
        original_chars = results[0][1]
        original_records = results[0][2]

        print("REDUCTION ANALYSIS:")
        for i in range(1, len(results)):
            if results[i][1] > 0:
                char_reduction = (original_chars - results[i][1]) / original_chars * 100
                record_reduction = (
                    (original_records - results[i][2]) / original_records * 100 if original_records > 0 else 0
                )
                remaining_tokens = results[i][1] // 4

                print(f"{results[i][0]}:")
                print(f"  - Character reduction: {char_reduction:.1f}% ({results[i][1]:,} chars remaining)")
                print(f"  - Record reduction: {record_reduction:.1f}% ({results[i][2]:,} records remaining)")
                print(f"  - Estimated tokens remaining: ~{remaining_tokens:,}")
                print()

    print(f"{'=' * 80}")


def run_deduplication(script_dir):
    """Run both exact and fuzzy deduplication on processed files"""
    processed_dir = script_dir / ".data/processed"
    processed_files = list(processed_dir.glob("*.txt"))

    if not processed_files:
        print("No processed files found for deduplication")
        return

    processed_file_paths = [str(f) for f in processed_files]
    print(f"\nRunning deduplication on {len(processed_files)} processed files...")

    # Run exact deduplication
    print("Step 1: Running exact line deduplication...")
    exact_dedup_dir = script_dir / ".data/exact_deduplicated"
    exact_deduplication(processed_file_paths, str(exact_dedup_dir))
    print(f"Exact deduplication complete. Output saved to {exact_dedup_dir}")

    # Run fuzzy deduplication on exact deduplicated files
    exact_dedup_files = list(exact_dedup_dir.glob("*.txt"))
    if exact_dedup_files:
        print("Step 2: Running fuzzy document deduplication...")
        fuzzy_dedup_dir = script_dir / ".data/fuzzy_deduplicated"
        exact_dedup_paths = [str(f) for f in exact_dedup_files]

        # Minhash parameters
        num_hashes = 128
        num_bands = 16
        ngrams = 3
        jaccard_threshold = 0.8

        # minhash_deduplication(exact_dedup_paths, num_hashes, num_bands, ngrams, jaccard_threshold, str(fuzzy_dedup_dir))
        # print(f"Fuzzy deduplication complete. Final output saved to {fuzzy_dedup_dir}")
    else:
        print("No files remaining after exact deduplication")


if __name__ == "__main__":
    # After run.py finished, this should be called
    script_dir = Path(__file__).parent
    run_deduplication(script_dir)
    analyze_deduplication_impact(script_dir)
    # fuzzy deduplication takes too long, skipping for now, want to get rid of this
