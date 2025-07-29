import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import timeit
import fasttext
import re


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


model = fasttext.load_model("lid.176.bin")


def remove_newlines(text: str) -> str:
    return re.sub(r"\n", "", text)


def langauge_identification(content: str = "Hi, my name is John"):
    # TODO: why does it ask to not use \n
    prediction = model.predict(remove_newlines(content))
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1].item()
    return label, confidence


def process_lang_id():
    """Process the parsed WARC text file and perform language identification."""
    import json
    from collections import defaultdict

    input_file = "sample_warc_parsed.txt"
    output_file = "language_analysis.json"

    print(f"Processing {input_file} for language identification...")

    results = []
    lang_distribution = defaultdict(int)
    english_confidences = []

    with open(input_file, "r", encoding="utf-8") as f:
        current_record = None
        current_url = None
        current_text = []
        reading_content = False

        for line in f:
            if line.startswith("=== Record"):
                # Process previous record if exists
                if current_record and current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:  # Only process non-empty content
                        lang, confidence = langauge_identification(text_content)
                        results.append(
                            {
                                "record": current_record,
                                "url": current_url,
                                "language": lang,
                                "confidence": confidence,
                                "text_length": len(text_content),
                            }
                        )
                        lang_distribution[lang] += 1
                        if lang == "en":
                            english_confidences.append(confidence)

                # Start new record
                current_record = line.strip()
                current_text = []
                reading_content = False

            elif line.startswith("URL:"):
                current_url = line[4:].strip()

            elif line.startswith("-" * 80):
                reading_content = True

            elif line.startswith("=" * 80):
                reading_content = False

            elif reading_content:
                current_text.append(line.rstrip())

        # Don't forget the last record
        if current_record and current_text:
            text_content = "\n".join(current_text).strip()
            if text_content:
                lang, confidence = langauge_identification(text_content)
                results.append(
                    {
                        "record": current_record,
                        "url": current_url,
                        "language": lang,
                        "confidence": confidence,
                        "text_length": len(text_content),
                    }
                )
                lang_distribution[lang] += 1
                if lang == "en":
                    english_confidences.append(confidence)

    # Calculate statistics
    total_records = len(results)
    lang_percentages = {lang: (count / total_records * 100) for lang, count in lang_distribution.items()}

    # Analyze English confidence thresholds
    english_stats = {}
    if english_confidences:
        english_confidences.sort()
        english_stats = {
            "min": min(english_confidences),
            "max": max(english_confidences),
            "mean": sum(english_confidences) / len(english_confidences),
            "median": english_confidences[len(english_confidences) // 2],
            "percentiles": {
                "10th": english_confidences[int(len(english_confidences) * 0.1)],
                "25th": english_confidences[int(len(english_confidences) * 0.25)],
                "50th": english_confidences[int(len(english_confidences) * 0.5)],
                "75th": english_confidences[int(len(english_confidences) * 0.75)],
                "90th": english_confidences[int(len(english_confidences) * 0.9)],
            },
            "threshold_recommendations": {
                "high_confidence": 0.9,  # Very confident English
                "medium_confidence": 0.7,  # Reasonably confident
                "low_confidence": 0.5,  # Barely confident
            },
        }

    # Create summary
    summary = {
        "total_records": total_records,
        "language_distribution": dict(lang_distribution),
        "language_percentages": lang_percentages,
        "english_statistics": english_stats,
        "top_languages": sorted(lang_percentages.items(), key=lambda x: x[1], reverse=True)[:10],
        "detailed_results": results,
    }

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nProcessed {total_records} records")
    print("\nTop 10 Languages:")
    for lang, percentage in summary["top_languages"]:
        print(f"  {lang}: {percentage:.2f}% ({lang_distribution[lang]} records)")

    if english_stats:
        print("\nEnglish Confidence Statistics:")
        print(f"  Min: {english_stats['min']:.4f}")
        print(f"  Max: {english_stats['max']:.4f}")
        print(f"  Mean: {english_stats['mean']:.4f}")
        print(f"  Median: {english_stats['median']:.4f}")
        print("\nRecommended thresholds for English content:")
        print("  - High confidence (>90% certain): 0.9")
        print("  - Medium confidence (>70% certain): 0.7")
        print("  - Low confidence (>50% certain): 0.5")

    print(f"\nDetailed results saved to: {output_file}")
    # TODO:given this stats, what makes sense here?


def remove_emails(text: str) -> tuple[str, int]:
    """
    Remove email addresses from text using regex.

    Matches common email patterns like:
    - user@domain.com
    - first.last@subdomain.domain.co.uk
    - user+tag@domain.org

    Returns:
        tuple[str, int]: (masked text, count of emails removed)
    """
    # Email regex pattern
    # Matches: word characters, dots, hyphens, plus signs @ domain with subdomains
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # Count matches before replacing
    count = len(re.findall(email_pattern, text))

    # Replace emails with [EMAIL]
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)

    return masked_text, count


def remove_phone_numbers(text: str) -> tuple[str, int]:
    """
    Remove phone numbers from text using regex.

    Matches various phone number formats like:
    - (123) 456-7890
    - 123-456-7890
    - 123.456.7890
    - +1 123 456 7890
    - 1234567890

    Returns:
        tuple[str, int]: (masked text, count of phone numbers removed)
    """
    # List of phone number patterns to match different formats
    phone_patterns = [
        # International format: +1 234 567 8900, +12345678900
        r"\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,4}",
        r"\+\d{10,15}",
        # US format with parentheses: (123) 456-7890
        r"\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}",
        # Standard formats: 123-456-7890, 123.456.7890, 123 456 7890
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        # 10-digit number: 1234567890
        r"\b\d{10}\b",
        # Extension formats: 123-456-7890 x1234
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\s?(?:ext|x|extension)\s?\d{1,5}\b",
    ]

    # Combine all patterns with OR operator
    combined_pattern = "|".join(phone_patterns)

    # Count matches before replacing
    count = len(re.findall(combined_pattern, text, flags=re.IGNORECASE))

    # Replace phone numbers with [PHONE]
    masked_text = re.sub(combined_pattern, "|||PHONE_NUMBER|||", text, flags=re.IGNORECASE)

    return masked_text, count


def remove_ip_addresses(text: str) -> tuple[str, int]:
    """
    Remove IP addresses from text using regex.

    Matches:
    - IPv4 addresses: 192.168.1.1
    - IPv6 addresses: 2001:0db8:85a3:0000:0000:8a2e:0370:7334

    Returns:
        tuple[str, int]: (masked text, count of IP addresses removed)
    """
    # IPv4 pattern: 4 groups of 1-3 digits separated by dots
    ipv4_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    # IPv6 pattern (simplified - matches most common formats)
    # Full IPv6: 8 groups of 4 hex digits separated by colons
    # Compressed IPv6: allows :: for consecutive zeros
    ipv6_pattern = r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b|\b(?:[A-Fa-f0-9]{1,4}:)*:(?:[A-Fa-f0-9]{1,4}:)*[A-Fa-f0-9]{1,4}\b"

    # Count matches before replacing
    ipv6_count = len(re.findall(ipv6_pattern, text))
    ipv4_count = len(re.findall(ipv4_pattern, text))
    total_count = ipv6_count + ipv4_count

    # First remove IPv6 (more specific pattern)
    text = re.sub(ipv6_pattern, "|||IP_ADDRESS|||", text)
    # Then remove IPv4
    text = re.sub(ipv4_pattern, "|||IP_ADDRESS|||", text)

    return text, total_count


def process_piid():
    """Process the parsed WARC text file and perform PII detection/removal."""
    import random

    input_file = "sample_warc_parsed.txt"
    output_file = "pii_analysis_sample.txt"

    print(f"Processing {input_file} for PII detection...")

    # Collect all records first
    all_records = []

    with open(input_file, "r", encoding="utf-8") as f:
        current_record = None
        current_url = None
        current_text = []
        reading_content = False

        for line in f:
            if line.startswith("=== Record"):
                # Process previous record if exists
                if current_record and current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:  # Only process non-empty content
                        all_records.append({"record": current_record, "url": current_url, "text": text_content})

                # Start new record
                current_record = line.strip()
                current_text = []
                reading_content = False

            elif line.startswith("URL:"):
                current_url = line[4:].strip()

            elif line.startswith("-" * 80):
                reading_content = True

            elif line.startswith("=" * 80):
                reading_content = False

            elif reading_content:
                current_text.append(line.rstrip())

        # Don't forget the last record
        if current_record and current_text:
            text_content = "\n".join(current_text).strip()
            if text_content:
                all_records.append({"record": current_record, "url": current_url, "text": text_content})

    # Sample 30 records (or all if less than 30)
    sample_size = min(30, len(all_records))
    sampled_records = random.sample(all_records, sample_size)

    # Process sampled records and write to output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PII Detection Analysis Sample\n")
        f.write(f"Sampled {sample_size} records from {len(all_records)} total records\n")
        f.write("=" * 100 + "\n\n")

        total_emails = 0
        total_phones = 0
        total_ips = 0

        for i, record in enumerate(sampled_records, 1):
            f.write(f"{'=' * 50} SAMPLE {i}/{sample_size} {'=' * 50}\n")
            f.write(f"Record: {record['record']}\n")
            f.write(f"URL: {record['url']}\n")
            f.write(f"Original text length: {len(record['text'])} characters\n")
            f.write("-" * 100 + "\n\n")

            # Apply PII removal functions
            text = record["text"]

            # Remove emails
            text_after_emails, email_count = remove_emails(text)
            total_emails += email_count

            # Remove phone numbers
            text_after_phones, phone_count = remove_phone_numbers(text_after_emails)
            total_phones += phone_count

            # Remove IP addresses
            text_after_ips, ip_count = remove_ip_addresses(text_after_phones)
            total_ips += ip_count

            # Write results
            f.write(f"PII Found:\n")
            f.write(f"  - Emails: {email_count}\n")
            f.write(f"  - Phone numbers: {phone_count}\n")
            f.write(f"  - IP addresses: {ip_count}\n")
            f.write(f"  - Total PII items: {email_count + phone_count + ip_count}\n\n")

            # Show original text (truncated if too long)
            f.write("ORIGINAL TEXT (first 1000 chars):\n")
            f.write("-" * 40 + "\n")
            f.write(text[:1000])
            if len(text) > 1000:
                f.write("\n... [TRUNCATED] ...")
            f.write("\n\n")

            # Show masked text (truncated if too long)
            f.write("MASKED TEXT (first 1000 chars):\n")
            f.write("-" * 40 + "\n")
            f.write(text_after_ips[:1000])
            if len(text_after_ips) > 1000:
                f.write("\n... [TRUNCATED] ...")
            f.write("\n\n")

            # If PII was found, show the differences
            if email_count + phone_count + ip_count > 0:
                f.write("DETECTED PII CONTEXTS:\n")
                f.write("-" * 40 + "\n")

                # Find and display contexts where PII was replaced
                # This helps identify false positives/negatives
                pattern = r"\|\|\|[A-Z_]+\|\|\|"
                matches = re.finditer(pattern, text_after_ips)

                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text_after_ips), match.end() + 50)
                    context = text_after_ips[start:end]

                    # Get corresponding original text
                    original_context = text[start:end]

                    f.write(f"Context around {match.group()}:\n")
                    f.write(f"  Original: ...{original_context}...\n")
                    f.write(f"  Masked:   ...{context}...\n\n")

            f.write("\n" + "=" * 100 + "\n\n")

        # Write summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total samples analyzed: {sample_size}\n")
        f.write(f"Total emails found: {total_emails}\n")
        f.write(f"Total phone numbers found: {total_phones}\n")
        f.write(f"Total IP addresses found: {total_ips}\n")
        f.write(f"Total PII items found: {total_emails + total_phones + total_ips}\n")
        f.write(f"Average PII items per record: {(total_emails + total_phones + total_ips) / sample_size:.2f}\n")

        # Records with most PII
        f.write("\nRecords with PII:\n")
        records_with_pii = 0
        for i, record in enumerate(sampled_records, 1):
            text = record["text"]
            _, emails = remove_emails(text)
            _, phones = remove_phone_numbers(text)
            _, ips = remove_ip_addresses(text)
            total = emails + phones + ips
            if total > 0:
                records_with_pii += 1
                f.write(f"  Sample {i}: {total} PII items ({emails} emails, {phones} phones, {ips} IPs)\n")

        f.write(f"\nRecords with PII: {records_with_pii}/{sample_size} ({records_with_pii / sample_size * 100:.1f}%)\n")

    print(f"\nProcessed {sample_size} sample records")
    print(f"Total PII items found: {total_emails + total_phones + total_ips}")
    print(f"  - Emails: {total_emails}")
    print(f"  - Phone numbers: {total_phones}")
    print(f"  - IP addresses: {total_ips}")
    print(f"\nResults saved to: {output_file}")
    print("\nThis file contains original and masked text samples to help identify false positives/negatives.")
    # TODO: how to actually identify false positives
    # TODO: how fasttext models get built
    


if __name__ == "__main__":
    # Time the process function
    execution_time = timeit.timeit(process_piid, number=1)
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    # print(langauge_identification())
