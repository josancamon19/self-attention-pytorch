from cs336_data.b_language_id import language_identification
from cs336_data.c_piid import remove_emails, remove_ip_addresses, remove_phone_numbers
from cs336_data.d_harmful import is_harmful
from cs336_data.e_gopher_heuristics import gopher_filters
import os
from collections import defaultdict
import json
from cs336_data._pipeline import warc_extract_pipeline, QualityProcessingType
from cs336_data.g_deduplication import exact_deduplication, minhash_deduplication
import glob
import shutil


def _check_filters_with_paloma(file_path: str):
    results = []
    filtered_by = defaultdict(int)
    with open(file_path) as f:
        documents = f.read().split("<|endoftext|>")

    for document in documents:
        gopher_result = gopher_filters(document)
        if not gopher_result["pass_filter"]:
            filtered_by["gopher"] += 1
            # gopher_result['alphabetic_word_ratio'] filters too much, at 0.8, put to 0.7, and now takes 377 from 1200
            # 0.6 should reduce to a lot like 50, yea
            # if not gopher_result["filters"]["alphabetic_filter"]:
            #     print(document)
            # word count filter, makes sense, tho in this dataset, those with less < 100 words, have content that make sense
            continue
        # print(language_identification(document))
        # 0.7 threshold seem reasonable, with paloma dataset, is taking out 46/13k records, maybe ones with code or weird formatting
        if not language_identification(document, True, 0.6):
            # print(document)  # mostly skips code, which makes sense, or weird formatting
            filtered_by["language"] += 1
            continue

        if is_harmful(document, 0.8):
            filtered_by["harmful"] += 1
            continue

        document, _ = remove_emails(document)
        document, _ = remove_ip_addresses(document)
        document, _ = remove_phone_numbers(document)
        results.append(document)

    print(len(results), f"{(len(results) / len(documents) * 100):.3f}")
    print(json.dumps(filtered_by, indent=2))


#   Crawling Strategy

#   1. URL-based news prioritization - Smart since you know Reuters/BBC/NYT are your highest-value sources
#   2. Deduplication for template removal - Critical for news sites (headers, footers, navigation)
#   3. Single WARC validation - Excellent iterative approach

#   Implementation Steps

#   1. News domain whitelist from your Paloma analysis (Reuters, BBC, NYT, etc.)
#   2. Template deduplication - Hash common page elements, keep article content
#   3. Quality metrics - Compare token distribution, readability, FastText classifications against Paloma
#   4. Rapid iteration - Process 1 WARC → measure similarity → adjust filters → repeat

#   Success Metrics

#   - FastText AG News classification ~37%
#   - Yahoo Answers classification ~24%
#   - Document length distribution matching (median ~500 tokens)
#   - Readability score ~63.8
#   - Minimal code/spam content

# TODO's
# ✅ 1. update _pipeline.py to make deduplication and remove templated content, ~ fuzzy doesn't remove this for 1 file.
# ✅ 2. line spacing? other factors that don't provide clean content as paloma ~ doing it manually
# ✅ 3. pipeline.py has the option to filter by URL sources
# ✅ 4. run with a basic set of urls (news *blog.com != /product/) ❌ != not included
# c 5. see the data, how does it look like? compared
# ~~~ before out of 27k 2.5k records were parsed, now 1k, and a lot is trimmed as well
# ~~~ it's not great yet, but idk, feels like I'm doing a lot already
# ~~~ [ ] is that why the assignment suggests a model instead? it's a lot to filter this way.
# ✅ 6. run it with models and similar stats that paloma has
# ~~~ Fuck, this is terrible!! my data is still way off
# 7. - [ ] try creating a classifier with the paloma validation dataset
# 8. scale the pipeline
# - retrieve 7B tokens on news articles only ~ try training the model only on this
# - then start incorporating some other tokens, like QA 20%, emails, academic papers, code(?)
# - can I just fail code? ignore it, and focus on high quality text?

# try 10 documents first and see how the overall data/scores look like


def process_data_with_heuristics():
    """
    Main pipeline function that processes all WARC files in .data directory
    with parsing and deduplication. News URL prioritization will be added later.
    """

    # Find all WARC files in leaderboard/.data directory
    data_dir = "cs336_data/leaderboard/.data"
    warc_files = glob.glob(f"{data_dir}/*.warc.gz")

    print(f"Found {len(warc_files)} WARC file(s) to process:")
    for warc_file in warc_files:
        print(f"  - {warc_file}")

    for warc_file in warc_files:
        print(f"\nProcessing: {warc_file}")

        # Generate output filename (replace .warc.gz with .txt)
        base_name = os.path.basename(warc_file).replace(".warc.gz", ".txt")
        parsed_text_file = f"{data_dir}/{base_name}"

        # Extract text from WARC file
        print("  Extracting and filtering text...")
        # Comprehensive news outlets from Paloma analysis
        quality_patterns = [
            # US Major Networks
            r"cnn\.com",
            r"foxnews\.com",
            r"msnbc\.com",
            r"abcnews\.go\.com",
            r"cbsnews\.com",
            r"nbcnews\.com",
            # Major Newspapers
            r"nytimes\.com",
            r"washingtonpost\.com",
            r"wsj\.com",
            r"usatoday\.com",
            r"latimes\.com",
            # International Major
            r"bbc\.(com|co\.uk)",
            r"reuters\.com",
            r"ap\.org",
            r"theguardian\.com",
            r"dailymail\.co\.uk",
            r"telegraph\.co\.uk",
            r"independent\.co\.uk",
            # Business/Finance
            r"bloomberg\.com",
            r"cnbc\.com",
            r"forbes\.com",
            r"ft\.com",
            # Magazines/Opinion
            r"time\.com",
            r"newsweek\.com",
            r"newyorker\.com",
            r"politico\.com",
            r"huffpost\.com",
            r"vox\.com",
            # Public Media
            r"pbs\.org",
            r"c-span\.org",
            # International
            r"aljazeera\.com",
            r"chinadaily\.com\.cn",
            r"timesofindia\.indiatimes\.com",
            # Canadian
            r"theglobeandmail\.com",
            r"cbc\.ca",
            # Blog content
            r"blog",
            r"wordpress\.com",
            r"medium\.com",
            r"substack\.com",
        ]
        warc_extract_pipeline(
            file_path=warc_file,
            target_output_path=parsed_text_file,
            process_language=True,
            process_piid=True,
            process_harmful=True,
            quality_processing=QualityProcessingType.GOPHER,
            custom_preprocessing=True,
            valid_urls_patterns=quality_patterns,
        )

        if not os.path.exists(parsed_text_file):
            print(f"  Warning: No text extracted to {parsed_text_file}")
            continue

        print(f"  Text extracted to: {parsed_text_file}")

        # Run exact deduplication
        print("  Running exact deduplication...")
        temp_dir = f"{data_dir}/temp_exact"
        exact_deduplication([parsed_text_file], temp_dir)
        temp_exact_file = f"{temp_dir}/{base_name}"

        # Run fuzzy deduplication with corrected parameters
        print("  Running fuzzy deduplication...")
        temp_dir2 = f"{data_dir}/temp_fuzzy"
        minhash_deduplication(
            paths=[temp_exact_file],
            output_dir=temp_dir2,
            num_hashes=100,
            num_bands=20,  # Fixed LSH parameters
            ngrams=5,
            jaccard_threshold=0.65,  # Lower threshold for template detection
        )
        temp_fuzzy_file = f"{temp_dir2}/{base_name}"

        # Overwrite original file with deduplicated version
        if os.path.exists(temp_fuzzy_file):
            print("  Overwriting original with deduplicated version...")
            os.rename(temp_fuzzy_file, parsed_text_file)
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_dir2, ignore_errors=True)
            print(f"  ✓ Completed: {parsed_text_file}")
        else:
            print(f"  ✗ Error: Deduplication failed for {warc_file}")

    print(f"\n🎉 Pipeline completed! Processed {len(warc_files)} WARC file(s)")


if __name__ == "__main__":
    process_data_with_heuristics()
