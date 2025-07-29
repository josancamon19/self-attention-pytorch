from cs336_data.a_html import extract_text_from_html_bytes
from cs336_data.b_language_id import language_identification
from cs336_data.c_piid import remove_emails, remove_ip_addresses, remove_phone_numbers
from cs336_data.d_harmful import is_harmful
from cs336_data.e_gopher_heuristics import gopher_filters
import os
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import queue


def process_record_batch(
    records_batch,
    process_language: bool,
    process_piid: bool,
    process_hamrful: bool,
    process_gopher: bool,
):
    """Process a batch of records in parallel"""
    results = []
    for record in records_batch:
        plain_text = extract_text_from_html_bytes(record)
        if process_gopher and not gopher_filters(plain_text)["pass_filter"]:
            continue
        if process_language and not language_identification(plain_text, True, 0.9):
            continue
        if process_hamrful and is_harmful(plain_text, 0.8):
            continue

        if process_piid:
            plain_text, _ = remove_emails(plain_text)
            plain_text, _ = remove_ip_addresses(plain_text)
            plain_text, _ = remove_phone_numbers(plain_text)
    return results


def pipeline(
    file_path: str = ".data/sample.warc.gz",
    process_language: bool = True,
    process_piid: bool = True,
    process_hamrful: bool = True,
    process_gopher: bool = True,
):
    assert ".warc.gz" in file_path

    dir_path = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    name_without_ext = base_name.replace(".warc.gz", "")
    parsed_text_file = os.path.join(dir_path, f"{name_without_ext}_parsed.txt")

    # decode_and_html_to_text(file_path, parsed_text_file)
    html_data = []
    with gzip.open(file_path, "rb") as stream:
        # Keep track of processed records
        record_count = 0

        with open(parsed_text_file, "w", encoding="utf-8") as out_f:
            for record in ArchiveIterator(stream):
                # Only process respone records (which contain the actual web content)
                if record.record_type == WarcRecordType.response:
                    record_count += 1
                    url = record.headers.get("WARC-Target-URI", "Unknown URL")
                    html_content = record.reader.read()
                    html_data.append(html_content)
                    plain_text = extract_text_from_html_bytes(html_content)

                    if process_gopher and not gopher_filters(plain_text)["pass_filter"]:
                        continue
                    if process_language and not language_identification(plain_text, True, 0.9):
                        continue
                    if process_hamrful and is_harmful(plain_text, 0.8):
                        continue

                    if process_piid:
                        plain_text, _ = remove_emails(plain_text)
                        plain_text, _ = remove_ip_addresses(plain_text)
                        plain_text, _ = remove_phone_numbers(plain_text)

                    # Write to output file
                    out_f.write(f"=== Record {record_count} ===\n")
                    out_f.write(f"URL: {url}\n")
                    out_f.write(f"Content-Length: {len(plain_text)} characters\n")
                    out_f.write("-" * 80 + "\n")
                    out_f.write(plain_text)
                    out_f.write("\n" + "=" * 80 + "\n\n")
                    print(f"Processed record {record_count}: {url}")
    print()
    print(f"\nProcessed {record_count} response records.")
    print(f"Results saved to: {parsed_text_file}")


if __name__ == "__main__":
    pipeline()
