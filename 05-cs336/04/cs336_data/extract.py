from cs336_data.filters.a_html import extract_text_from_html_bytes
from cs336_data.filters.b_language_id import language_identification
from cs336_data.filters.c_piid import remove_emails, remove_ip_addresses, remove_phone_numbers
from cs336_data.filters.d_harmful import is_harmful
from cs336_data.filters.e_gopher_heuristics import gopher_filters
from cs336_data.filters.f_quality_classifier import classify_quality as matches_wiki_quality
from cs336_data.leaderboard.processing.classifier import matches_paloma_quality
import os
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import random
from enum import Enum
import re


class QualityProcessingType(Enum):
    GOPHER = "gopher"
    FASTTEXT = "fasttext"
    PALOMA = "paloma"
    NONE = "none"


def _custom_preprocessing(plain_text: str):
    lines = plain_text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.endswith("..."):
            continue

        if line.startswith("• "):
            line = line[2:].strip()
            if not line:
                continue

        # Fast length check first
        if len(line) > 100:
            cleaned_lines.append(line)
            continue

        # Optimized word count check
        word_count = 0
        for word in line.split():
            if len(word) >= 3:
                word_count += 1
                if word_count >= 5:  # Early exit when threshold met
                    break

        if word_count >= 5:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def process_single_record(
    html_content: bytes,
    process_language: bool,
    process_piid: bool,
    process_harmful: bool,
    quality_processing: QualityProcessingType,
    custom_preprocessing: bool,
):
    """Process a single record and return the processed text or None if filtered"""
    plain_text = extract_text_from_html_bytes(html_content)

    if process_language and not language_identification(plain_text, True, 0.6):
        return None

    if custom_preprocessing:
        plain_text = _custom_preprocessing(plain_text)

    if quality_processing == QualityProcessingType.GOPHER and not gopher_filters(plain_text)["pass_filter"]:
        # thought they were filtering too much from my hq wikipedia pages
        # but actually it does make sense what's filtering
        # but also, thinking deduplication should happen before this filter
        # - lots of menu's repeated stuff, that'd clean up further here
        # - but why's on the assignment as f_quality_classifier?
        # print(url)
        # print(sorted([(k, v) for k, v in result["filters"].items() if not v]))
        return None
    elif quality_processing == QualityProcessingType.PALOMA and not matches_paloma_quality(plain_text):
        return None

    if process_harmful and is_harmful(plain_text, 0.8):
        return None

    if process_piid:
        plain_text, _ = remove_emails(plain_text)
        plain_text, _ = remove_ip_addresses(plain_text)
        plain_text, _ = remove_phone_numbers(plain_text)

    return plain_text


def process_record_batch(
    records_batch,
    process_language: bool,
    process_piid: bool,
    process_harmful: bool,
    quality_processing: QualityProcessingType,
    custom_preprocessing: bool,
):
    """Process a batch of records in parallel"""
    results = []
    filtered_by = defaultdict(int)
    for record_data in records_batch:
        url, html_content = record_data
        plain_text = process_single_record(
            html_content, process_language, process_piid, process_harmful, quality_processing, custom_preprocessing
        )
        results.append((url, plain_text))
    return results, filtered_by


def process_warc_file(
    file_path: str,
    target_output_path: str | None = None,
    subsample_count: int = None,
):
    """process and stream to disk directly, faster(?)"""
    assert ".warc.gz" in file_path

    if not target_output_path:
        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name_without_ext = base_name.replace(".warc.gz", "")
        output_path = os.path.join(dir_path, f"{name_without_ext}.txt")
    else:
        output_path = target_output_path

    # Stream and process records directly to disk
    processed_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        with gzip.open(file_path, "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.record_type == WarcRecordType.response:
                    if subsample_count and processed_count >= subsample_count:
                        break

                    html_content = record.reader.read()
                    processed_text = process_single_record(
                        html_content, True, True, True, QualityProcessingType.PALOMA, True
                    )
                    del html_content

                    if processed_text is not None:
                        out_f.write(processed_text)
                        out_f.write("<|endoftext|>")
                        processed_count += 1

    # print(f"Streaming processing completed: {processed_count} records")
    return processed_count, output_path


def process_warc_file_parallel(
    file_path: str = ".data/sample.warc.gz",
    target_output_path: str | None = None,
    process_language: bool = True,
    process_piid: bool = True,
    process_harmful: bool = True,
    quality_processing=QualityProcessingType.GOPHER,
    custom_preprocessing: bool = True,  # not asked, but imo is needed
    subsample_count: int = None,
    valid_urls_patterns: list = [],
    include_metadata: bool = True,
):
    assert ".warc.gz" in file_path

    if not target_output_path:
        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name_without_ext = base_name.replace(".warc.gz", "")
        parsed_text_file = os.path.join(dir_path, f"{name_without_ext}.txt")
    else:
        parsed_text_file = target_output_path

    # Collect all records first
    all_records = []
    skipped_by_url_count = 0
    with gzip.open(file_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                url = record.headers.get("WARC-Target-URI", "Unknown URL")
                if valid_urls_patterns:
                    url_matches = False
                    for pattern in valid_urls_patterns:
                        if re.search(pattern, url, re.IGNORECASE):
                            url_matches = True
                            break

                    if url_matches:
                        html_content = record.reader.read()
                        all_records.append((url, html_content))
                    else:
                        skipped_by_url_count += 1
                else:
                    html_content = record.reader.read()
                    all_records.append((url, html_content))

    print(f"Collected {len(all_records)} records, processing with {cpu_count()} processes")
    if skipped_by_url_count:
        print(f"skipped {skipped_by_url_count} cause not in URL matches")

    if subsample_count:
        all_records = random.sample(all_records, subsample_count)
        print(f"Records subsampled down to {len(all_records)} records")

    # Process records either with or without multiprocessing
    all_results = []
    # Split records into batches for multiprocessing
    batch_size = max(1, len(all_records) // cpu_count())
    batches = [all_records[i : i + batch_size] for i in range(0, len(all_records), batch_size)]

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(
                process_record_batch,
                batch,
                process_language,
                process_piid,
                process_harmful,
                quality_processing,
                custom_preprocessing,
            )
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            batch_results, filtered_by = future.result()
            all_results.extend(batch_results)
            print(f"Batch {i + 1}/{len(batches)} completed: {len(batch_results)} records")
            # print(f"Cleaning stats: {filtered_by}")

    # Write all results to output file
    with open(parsed_text_file, "w", encoding="utf-8") as out_f:
        for i, (url, plain_text) in enumerate(all_results, 1):
            if include_metadata:
                out_f.write(f"=== Record {i} ===\n")
                out_f.write(f"URL: {url}\n")
                out_f.write(f"Content-Length: {len(plain_text)} characters\n")
                out_f.write("-" * 80 + "\n")
                out_f.write(plain_text)
                out_f.write("\n" + "=" * 80 + "\n\n")
            else:
                out_f.write(plain_text)
                if i < len(all_results):  # Don't add separator after last record
                    out_f.write("<|endoftext|>")

    print(f"\nProcessed {len(all_results)} response records.")
    print(f"Results saved to: {parsed_text_file}")
    return len(all_results), parsed_text_file


def check_separate_sample_with_paloma_filtering():
    process_warc_file_parallel(
        file_path="cs336_data/leaderboard/.data/00002.warc.gz",
        quality_processing=QualityProcessingType.PALOMA,
        # subsample_count=100,
    )


if __name__ == "__main__":
    check_separate_sample_with_paloma_filtering()
