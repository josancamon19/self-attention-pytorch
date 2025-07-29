# ==== QUALITY CLASSIFIER =====
# - this also applies to search engines
# - in general the rule is, hq pages link other hq pages
# - heuristics, gpt2 reddit karma >= 3
# - or links that are only in wikipedia
# - 40GB of text, this is too small, but you can take it as example, and train a classifier
# -- with this as good examples, common crawl as negative examples. Giving you a quality score
# -- implement a classifier using a subset of hq pages + use some of your filters.


# This file contains a list of 43.5M external links
# found on Wikipedia pages in the English language as of April of 2024, but we expect you to subsample these
# URLs to get positive examples of “high-quality” text for training your classifier. Note that these positive
# examples may still contain undesirable content, so it may be useful to apply the other primitives you’ve built
# (lang id, filtering rules, ,etc)

# Scrape contents of an URL
# wget–-timeout=5 \
# -i subsampled_positive_urls.txt \
# --warc-file=subsampled_positive_urls.warc \
# -O /dev/null

#  Train a quality classifier that, given text, returns a numeric quality score.
# Deliverable: A quality classifier for use in the next subproblem.
# (b) Write a function that labels a page as high or low-quality, and provides a confidence score in the
# label.
from cs336_data._pipeline import warc_extract_pipeline
import random
import fasttext
import re

subsample_size = 10000
negative_sampling = subsample_size * 2  # rate of discard is ≈ 2x as high
dataset_path = ".data/wikipedia_HQ_urls.txt"
subsample_path = f".data/wikipedia_subsampled_{subsample_size}_urls.txt"
subsample_warc_path = subsample_path.replace("_urls.txt", ".warc.gz")
classifier_train_path = ".data/classifier.train.txt"
classifier_valid_path = ".data/classifier.valid.txt"
apply_gopher_filter = (False, True)  # negatives, positives
fasttext_model_path = ".models/quality_classifier.bin"


def subsample():
    import random
    from pathlib import Path

    wiki_file = Path(dataset_path)
    if not wiki_file.exists():
        raise FileNotFoundError(f"Wikipedia URLs file not found: {wiki_file}")

    with open(wiki_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    print(f"Found {total_lines} total URLs in Wikipedia file")

    if subsample_size >= total_lines:
        print(f"Target count ({subsample_size}) >= total lines ({total_lines}), returning all URLs")
        sampled_lines = all_lines
    else:
        sampled_lines = random.sample(all_lines, subsample_size)

    output_file = Path(subsample_path)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    print(f"Subsampled {len(sampled_lines)} URLs to {output_file}")
    return output_file


def urls_into_warc():
    # had to be faster than warc
    import asyncio
    import aiohttp
    import gzip
    from datetime import datetime
    from pathlib import Path

    async def fetch_url(session, url, semaphore):
        async with semaphore:
            try:
                async with session.get(url.strip(), timeout=aiohttp.ClientTimeout(total=5)) as response:
                    content = await response.read()
                    return {
                        "url": url.strip(),
                        "status": response.status,
                        "content": content,
                        "headers": dict(response.headers),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception as e:
                return {
                    "url": url.strip(),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }

    async def download_all_urls():
        with open(subsample_path, "r") as f:
            urls = f.readlines()

        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        async with aiohttp.ClientSession(
            connector=connector, headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"}
        ) as session:
            tasks = [fetch_url(session, url, semaphore) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Write to WARC format
        warc_path = Path(subsample_warc_path)
        with gzip.open(warc_path, "wt", encoding="utf-8") as f:
            for result in results:
                if isinstance(result, dict) and result.get("content"):
                    try:
                        content_text = result["content"].decode("utf-8", errors="ignore")
                    except:
                        content_text = str(result["content"])

                    # Proper WARC record format
                    warc_record = f"""WARC/1.0\r
WARC-Type: response\r
WARC-Target-URI: {result["url"]}\r
WARC-Date: {result["timestamp"]}\r
Content-Type: text/html\r
Content-Length: {len(content_text.encode("utf-8"))}\r
\r
{content_text}\r
\r
"""
                    f.write(warc_record)

        successful = sum(1 for r in results if isinstance(r, dict) and r.get("content"))
        print(f"Downloaded {successful}/{len(urls)} URLs successfully to {warc_path}")

    try:
        asyncio.run(download_all_urls())
    except ImportError:
        print("aiohttp not installed. Install with: pip install aiohttp")
        print(f"Fallback: wget –-timeout=5 -i {subsample_path} --warc-file={subsample_warc_path} -O /dev/null")


def create_dataset():
    pcount, ppath = warc_extract_pipeline(
        subsample_warc_path,
        process_gopher=apply_gopher_filter[0],
        subsample_count=int(subsample_size * [0.25, 1][apply_gopher_filter[0]]),
    )
    ncount, npath = warc_extract_pipeline(
        ".data/sample.warc.gz",
        subsample_count=int(negative_sampling * [0.25, 1][apply_gopher_filter[1]]),
        process_gopher=apply_gopher_filter[1],
    )
    print("create_dataset retrieved positives, negatives:", pcount, ncount)
    print(ppath, npath)
    print()

    # Convert to fastText format
    convert_to_fasttext_format(ppath, npath)


def convert_to_fasttext_format(positive_path, negative_path, split: float = 0.8):
    def clean_text_for_fasttext(text):
        text = re.sub(r"^\d+ ===\n", "", text)  # Remove "1 ===\n"
        text = re.sub(r"^URL: .*?\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Content-Length: .*?\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"^-{80,}\n", "", text, flags=re.MULTILINE)  # 80+ dashes
        text = re.sub(r"^={80,}\n*", "", text, flags=re.MULTILINE)  # 80+ equals

        text = text.replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", text).strip()

    # Collect all samples first
    all_samples = []

    # Process positive examples (high quality)
    with open(positive_path, encoding="utf-8") as f:
        content = f.read()
        records = content.split("=== Record ")[1:]
        for record in records:
            text = clean_text_for_fasttext(record)
            if text:
                all_samples.append(f"__label__high_quality {text}")

    # Process negative examples (low quality)
    with open(negative_path, encoding="utf-8") as f:
        content = f.read()
        records = content.split("=== Record ")[1:]
        for record in records:
            text = clean_text_for_fasttext(record)
            if text:
                all_samples.append(f"__label__low_quality {text}")

    # Shuffle all samples
    random.shuffle(all_samples)

    # Split into train/validation
    split_idx = int(len(all_samples) * split)
    train_samples = all_samples[:split_idx]
    valid_samples = all_samples[split_idx:]

    # Write train set
    with open(classifier_train_path, "w", encoding="utf-8") as train_f:
        for sample in train_samples:
            train_f.write(f"{sample}\n")

    # Write validation set
    with open(classifier_valid_path, "w", encoding="utf-8") as valid_f:
        for sample in valid_samples:
            valid_f.write(f"{sample}\n")

    print(f"FastText train dataset: {classifier_train_path} ({len(train_samples)} samples)")
    print(f"FastText valid dataset: {classifier_valid_path} ({len(valid_samples)} samples)")


def train_quality_classifier():
    model = fasttext.train_supervised(input=classifier_train_path, epoch=25, lr=0.1, wordNgrams=2, dim=100)
    model.save_model(fasttext_model_path)
    print(f"Model saved to: {fasttext_model_path}")
    print(model.test(classifier_valid_path))  # count, precision, f1
    # 1000, train:valid ≈ 280:70, (72, 0.4166666666666667, 0.4166666666666667)
    # so the model is not learning anything
    # wait, shouldn't I be applying gopher rules?
    # I kinda think so, this seems like an alternative to gopher?
    # - cause both are high quality?
    # ok so try the following TODO:
    # 1. train applying gopher filters to high quality data, not to lowq
    # - (67, 0.582089552238806, 0.582089552238806)
    # 2. not applying gopher to either
    # - (70, 0.5857142857142857, 0.5857142857142857)
    
    # - I think applying gopher to positives make sense
    # -- trying now with 10k samples


def classify_quality(text):
    # Load model
    model = fasttext.load_model(fasttext_model_path)

    # Clean text same way as training data ~ clean the same way we parsed for training
    import re

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Predict
    labels, probabilities = model.predict(text, k=2)
    print(labels, probabilities)

    # Return label and confidence
    top_label = labels[0].replace("__label__", "")
    confidence = probabilities[0]

    return {"label": top_label, "confidence": float(confidence), "is_high_quality": top_label == "high_quality"}


if __name__ == "__main__":
    subsample()
    urls_into_warc()
    create_dataset()
    train_quality_classifier()
    # print(classify_quality("asdasdasdasdasdfiuqodnoqiwdj"))
