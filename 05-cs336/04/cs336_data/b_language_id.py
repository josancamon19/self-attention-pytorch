import fasttext
import re

langid_model = fasttext.load_model(".models/lid.176.bin")


def language_identification(
    content: str = "Hi, my name is John",
    determine_is_english: bool = False,
    confidence_threshold: float = 0.9,
):
    # TODO: why does it ask to not use \n
    prediction = langid_model.predict(re.sub(r"\n", "", content))
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1].item()
    if determine_is_english:
        return label == "en" and confidence_threshold >= confidence
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
                        lang, confidence = language_identification(text_content)
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
                lang, confidence = language_identification(text_content)
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
