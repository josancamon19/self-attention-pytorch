from cs336_data.f_quality_classifier import clean_text_for_fasttext
from cs336_data._pipeline import warc_extract_pipeline, QualityProcessingType
import os
import fasttext


def create_low_quality_samples():
    # wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/segments/1751905933612.63/warc/CC-MAIN-20250707183638-20250707213638-00001.warc.gz
    # wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/segments/1751905933612.63/warc/CC-MAIN-20250707183638-20250707213638-00002.warc.gz
    count, path1 = warc_extract_pipeline(
        file_path="cs336_data/leaderboard/.data/2530-000.warc.gz",
        quality_processing=QualityProcessingType.NONE,
        custom_preprocessing=True,  # match structure so classifier diff are mainly on semantics
    ) # count, path1= 7110, "cs336_data/leaderboard/.data/2530-000_parsed.txt"
    count2, path2 = warc_extract_pipeline(
        file_path="cs336_data/leaderboard/.data/2530-001.warc.gz",
        quality_processing=QualityProcessingType.NONE,
        custom_preprocessing=True,  # match structure so classifier diff are mainly on semantics
    )

    # Extract documents from both parsed files
    negative_samples = []

    for path in [path1, path2]:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found, skipping...")
            continue

        with open(path, "r", encoding="utf-8") as f:
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
    return negative_samples


def create_fasttext_classifier():
    data_file = "cs336_data/leaderboard/.data/paloma_c4_100_domains_validation.txt"
    classifier_train_path = "cs336_data/leaderboard/.data/classifier_train.txt"
    classifier_valid_path = "cs336_data/leaderboard/.data/classifier_valid.txt"
    fasttext_model_path = "cs336_data/leaderboard/.models/paloma_classifier.bin"
    os.makedirs("cs336_data/leaderboard/.models", exist_ok=True)

    with open(data_file, encoding="utf-8") as f:
        content = f.read()

    documents = content.split("<|endoftext|>")
    documents = [doc.strip() for doc in documents if doc.strip()]

    # Prepare training samples
    all_samples = []
    for text in documents:
        # Clean text for FastText format
        cleaned_text = clean_text_for_fasttext(text)
        if cleaned_text and len(cleaned_text.split()) > 10:  # Minimum word count
            all_samples.append(f"__label__paloma {cleaned_text}")

    print(f"Created {len(all_samples)} training samples from {len(documents)} documents")

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


def matches_paloma_quality(text: str) -> bool:
    fasttext_model_path = "cs336_data/leaderboard/.models/paloma_classifier.bin"
    model = fasttext.load_model(fasttext_model_path)
    cleaned_text = clean_text_for_fasttext(text)

    # Basic quality checks
    if not cleaned_text or len(cleaned_text.split()) < 10:
        return False

    # Predict with model
    prediction = model.predict(cleaned_text, k=1)
    label = prediction[0][0]
    confidence = prediction[1][0]

    # Return True if classified as paloma with high confidence
    return label == "__label__paloma" and confidence > 0.7


if __name__ == "__main__":
    create_low_quality_samples()
