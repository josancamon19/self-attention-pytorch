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

import random
import fasttext
import re
from pathlib import Path
from cs336_data.utils.warc import retrieve_multiple_urls_to_warc

subsample_size = 10000
negative_sampling = subsample_size * 2  # rate of discard is ≈ 2x as high
dataset_path = ".data/wikipedia_HQ_urls.txt"
subsample_path = f".data/wikipedia_subsampled_{subsample_size}_urls.txt"
subsample_warc_path = subsample_path.replace("_urls.txt", ".warc.gz")
classifier_train_path = ".data/classifier.train.txt"
classifier_valid_path = ".data/classifier.valid.txt"
fasttext_model_path = ".models/quality_classifier.bin"


def subsample_urls():
    with open(Path(dataset_path), encoding="utf-8") as f:
        all_lines = f.readlines()

    sampled_lines = random.sample(all_lines, subsample_size)
    output_file = Path(subsample_path)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    print(f"Found {len(all_lines)}, subsampled {len(sampled_lines)} URLs to {output_file}")
    return output_file


def create_dataset():
    from cs336_data.extract import warc_extract_pipeline, QualityProcessingType
    from cs336_data._fasttext_util import clean_text_for_fasttext

    pcount, positive_path = warc_extract_pipeline(
        subsample_warc_path,
        quality_processing=QualityProcessingType.GOPHER,
    )
    ncount, negative_path = warc_extract_pipeline(
        ".data/sample.warc.gz",
        quality_processing=QualityProcessingType.NONE,
        subsample_count=int(negative_sampling * 0.25),
    )
    print("create_dataset retrieved positives, negatives:", pcount, ncount)

    all_samples = []
    # Process positive examples (high quality)
    with open(positive_path, encoding="utf-8") as f:
        content = f.read()
        records = content.split("=== Record ")[1:]
        for record in records:
            text = clean_text_for_fasttext(record)
            if text:
                all_samples.append(f"__label__wiki {text}")

    # Process negative examples (low quality)
    with open(negative_path, encoding="utf-8") as f:
        content = f.read()
        records = content.split("=== Record ")[1:]
        for record in records:
            text = clean_text_for_fasttext(record)
            if text:
                all_samples.append(f"__label__cc {text}")

    # Shuffle all samples
    random.shuffle(all_samples)

    # Split into train/validation
    split_idx = int(len(all_samples) * 0.8)
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
    model = fasttext.train_supervised(input=classifier_train_path, epoch=50, lr=1.0, wordNgrams=3, dim=100)
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
    # - (666, 0.8678678678678678, 0.8678678678678678)
    # - changed hyperparams and moved it to 0.9


def classify_quality(text):
    model = fasttext.load_model(fasttext_model_path)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    labels, probabilities = model.predict(text, k=2)
    # print(labels, probabilities)

    # Return label and confidence
    top_label = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    return top_label, confidence


if __name__ == "__main__":
    subsample_urls()
    retrieve_multiple_urls_to_warc(subsample_path, subsample_warc_path)
    create_dataset()
    train_quality_classifier()
