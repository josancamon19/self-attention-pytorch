import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_string = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        detected_encoding = detect_encoding(html_bytes)
        if detected_encoding:
            try:
                html_string = html_bytes.decode(detected_encoding)
            except (UnicodeDecodeError, LookupError):
                html_string = html_bytes.decode("utf-8", errors="replace")
        else:
            html_string = html_bytes.decode("utf-8", errors="replace")
    return extract_plain_text(html_string)


def process():
    # Process the sample.warc.gz file
    input_file = "sample.warc.gz"
    output_file = "sample_warc_parsed.txt"

    print(f"Processing {input_file}...")

    with gzip.open(input_file, "rb") as stream:
        # Keep track of processed records
        record_count = 0

        with open(output_file, "w", encoding="utf-8") as out_f:
            for record in ArchiveIterator(stream):
                # Only process respone records (which contain the actual web content)
                if record.record_type == WarcRecordType.response:
                    record_count += 1
                    url = record.headers.get("WARC-Target-URI", "Unknown URL")
                    html_content = record.reader.read()
                    plain_text = extract_text_from_html_bytes(html_content)

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
    print(f"Results saved to: {output_file}")
