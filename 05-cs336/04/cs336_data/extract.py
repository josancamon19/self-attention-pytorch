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
        plain_text = extract_text_from_html_bytes(html_content)

        if process_language and not language_identification(plain_text, True, 0.6):
            filtered_by["language"] += 1
            continue

        if process_harmful and is_harmful(plain_text, 0.8):
            filtered_by["harmful"] += 1
            continue

        if custom_preprocessing:
            lines = plain_text.split("\n")
            del plain_text
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line.endswith("..."):
                    continue  # no at all of this data, and most times cause it's cut here, not really making sense
                if line.startswith("• "):  # TODO: I might add some of this data at some point but prob not
                    line = line[2:].strip()
                if not line:
                    continue
                if len(line) > 100:  # split is too expensive
                    cleaned_lines.append(line)
                    continue

                if sum([len(w) >= 3 for w in line.split()]) < 5:  # some • or random symbols
                    continue
                cleaned_lines.append(line)
            plain_text = "\n".join(cleaned_lines)

        if quality_processing == QualityProcessingType.GOPHER:
            result = gopher_filters(plain_text)
            if not result["pass_filter"]:
                # thought they were filtering too much from my hq wikipedia pages
                # but actually it does make sense what's filtering
                # but also, thinking deduplication should happen before this filter
                # - lots of menu's repeated stuff, that'd clean up further here
                # - but why's on the assignment as f_quality_classifier?
                # print(url)
                # print(sorted([(k, v) for k, v in result["filters"].items() if not v]))
                filtered_by["gopher"] += 1
                continue
        elif quality_processing == QualityProcessingType.FASTTEXT:
            raise NotImplementedError()
        elif quality_processing == QualityProcessingType.PALOMA and not matches_paloma_quality(plain_text):
            continue

        if process_piid:
            plain_text, _ = remove_emails(plain_text)
            plain_text, _ = remove_ip_addresses(plain_text)
            plain_text, _ = remove_phone_numbers(plain_text)

        results.append((url, plain_text))
    return results, filtered_by


def warc_extract_pipeline(
    file_path: str = ".data/sample.warc.gz",
    target_output_path: str | None = None,
    process_language: bool = True,
    process_piid: bool = True,
    process_harmful: bool = True,
    quality_processing=QualityProcessingType.GOPHER,
    custom_preprocessing: bool = True,  # not asked, but imo is needed
    subsample_count: int = None,
    valid_urls_patterns: list = [],
):
    assert ".warc.gz" in file_path

    if not target_output_path:
        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name_without_ext = base_name.replace(".warc.gz", "")
        parsed_text_file = os.path.join(dir_path, f"{name_without_ext}_parsed.txt")
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

    # Split records into batches for multiprocessing
    batch_size = max(1, len(all_records) // cpu_count())
    batches = [all_records[i : i + batch_size] for i in range(0, len(all_records), batch_size)]

    # Process batches in parallel
    all_results = []
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
            out_f.write(f"=== Record {i} ===\n")
            out_f.write(f"URL: {url}\n")
            out_f.write(f"Content-Length: {len(plain_text)} characters\n")
            out_f.write("-" * 80 + "\n")
            out_f.write(plain_text)
            out_f.write("\n" + "=" * 80 + "\n\n")

    print(f"\nProcessed {len(all_results)} response records.")
    print(f"Results saved to: {parsed_text_file}")
    return len(all_results), parsed_text_file


if __name__ == "__main__":
    warc_extract_pipeline(
        file_path="cs336_data/leaderboard/.data/2530-002.warc.gz",
        quality_processing=QualityProcessingType.PALOMA,
        subsample_count=100,
    )
