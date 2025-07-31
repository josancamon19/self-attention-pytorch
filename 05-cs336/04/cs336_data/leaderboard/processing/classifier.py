from cs336_data.utils.fasttext_util import clean_text_for_fasttext
import os
import random
import fasttext


def create_low_quality_dataset():
    from cs336_data._pipeline import warc_extract_pipeline, QualityProcessingType

    _dir = "cs336_data/leaderboard/.data/"
    count, path1 = warc_extract_pipeline(
        file_path=f"{_dir}00000.warc.gz", quality_processing=QualityProcessingType.NONE
    )
    count2, path2 = warc_extract_pipeline(
        file_path=f"{_dir}00001.warc.gz", quality_processing=QualityProcessingType.NONE
    )

    # Extract documents from both parsed files
    negative_samples = []

    for path in [path1, path2]:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found, skipping...")
            continue

        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Split by record separator
        records = content.split("=== Record ")
        print(f"Found {len(records) - 1} records in {path}")

        for record in records[1:]:  # Skip first empty split
            lines = record.split("\n")
            doc_lines = []

            # Skip metadata lines and extract document content
            for line in lines:
                # Skip URL line
                if line.startswith("URL:"):
                    continue
                # Skip Content-Length line
                if line.startswith("Content-Length:"):
                    continue
                # Skip separator line
                if line.startswith("-" * 80):
                    continue
                # Skip empty lines at start
                if not doc_lines and line.strip() == "":
                    continue
                # Stop at record end
                if line.startswith("=" * 80):
                    break

                doc_lines.append(line)

            # Join document content
            doc_text = "\n".join(doc_lines).strip()
            if doc_text:
                # Clean text for FastText format
                cleaned_text = clean_text_for_fasttext(doc_text)
                if cleaned_text and len(cleaned_text.split()) > 10:
                    negative_samples.append(f"__label__low_quality {cleaned_text}")

    print(f"Created {len(negative_samples)} negative samples from WARC files")

    output_file = "cs336_data/leaderboard/.data/classifier_negative_samples.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(negative_samples))

    print(f"Saved {len(negative_samples)} negative samples to {output_file}")
    return negative_samples  # cs336_data/leaderboard/.data/classifier_negative_samples.txt


def create_fasttext_classifier():
    paloma_file = "cs336_data/leaderboard/.data/paloma_c4_100_domains_validation.txt"
    negative_file = "cs336_data/leaderboard/.data/classifier_negative_samples.txt"
    classifier_train_path = "cs336_data/leaderboard/.data/classifier_train.txt"
    classifier_valid_path = "cs336_data/leaderboard/.data/classifier_valid.txt"
    fasttext_model_path = "cs336_data/leaderboard/.models/paloma_classifier.bin"
    os.makedirs("cs336_data/leaderboard/.models", exist_ok=True)

    # Read positive samples (Paloma)
    with open(paloma_file, encoding="utf-8") as f:
        paloma_content = f.read()

    paloma_documents = paloma_content.split("<|endoftext|>")
    paloma_documents = [doc.strip() for doc in paloma_documents if doc.strip()]

    positive_samples = []
    for text in paloma_documents:
        cleaned_text = clean_text_for_fasttext(text)
        if cleaned_text and len(cleaned_text.split()) > 10:
            positive_samples.append(f"__label__quality {cleaned_text}")

    print(f"Created {len(positive_samples)} positive samples from Paloma data")

    # Read negative samples
    negative_samples = []
    if os.path.exists(negative_file):
        with open(negative_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.startswith("__label__low_quality"):
                    negative_samples.append(line)
        print(f"Loaded {len(negative_samples)} negative samples from file")
    else:
        print(f"Warning: Negative samples file {negative_file} not found!")
        return None

    # Balance datasets - use minimum of both
    min_samples = min(len(positive_samples), len(negative_samples))
    positive_samples = positive_samples[:min_samples]
    negative_samples = negative_samples[:min_samples]

    print(f"Balanced datasets: {len(positive_samples)} positive, {len(negative_samples)} negative")

    # Combine and shuffle samples
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    print(f"Total samples for training: {len(all_samples)}")

    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    valid_samples = all_samples[split_idx:]

    # Write training file
    with open(classifier_train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(sample + "\n")

    # Write validation file
    with open(classifier_valid_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(sample + "\n")

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(valid_samples)}")
    print(f"Training file: {classifier_train_path}")
    print(f"Validation file: {classifier_valid_path}")

    # Train FastText model
    print("Training FastText model...")
    model = fasttext.train_supervised(input=classifier_train_path, epoch=50, lr=1.0, wordNgrams=3, dim=100)

    # Save model
    model.save_model(fasttext_model_path)
    print(f"Model saved to: {fasttext_model_path}")

    # Test model on validation set
    print("Testing model on validation set:")
    test_results = model.test(classifier_valid_path)
    print(f"Samples: {test_results[0]}, Precision: {test_results[1]:.3f}, Recall: {test_results[2]:.3f}")

    return model


try:
    fasttext_model_path = "cs336_data/leaderboard/.models/paloma_classifier.bin"
    model = fasttext.load_model(fasttext_model_path)
except:  # noqa: E722
    pass


def matches_paloma_quality(text: str, quality_threshold: float = 0.7) -> bool:
    cleaned_text = clean_text_for_fasttext(text)
    if not cleaned_text or len(cleaned_text.split()) < 10:
        return False
    prediction = model.predict(cleaned_text, k=2)
    return prediction[0][0] == "__label__quality" and prediction[1][0] > quality_threshold


if __name__ == "__main__":
    create_low_quality_dataset()
    create_fasttext_classifier()
    # in extract.py, call check_separate_sample_with_paloma_filtering and double check the actual filter
