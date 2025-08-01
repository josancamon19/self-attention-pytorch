#!/usr/bin/env python3

import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from google.cloud import storage
from transformers import AutoTokenizer


def upload_to_gcp(file_path, bucket_name, blob_name):
    """Upload a file to Google Cloud Storage."""
    client = storage.Client.from_service_account_json("google-credentials.json")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"Uploading {file_path} to gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(file_path)
    print(f"Upload complete: gs://{bucket_name}/{blob_name}")


tokenizer = None


def init_worker():
    """Initialize tokenizer once per worker process."""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="cs336_data/leaderboard/.cache")
    tokenizer.model_max_length = 10000000  # to avoid warning


def tokenize_file(file_path):
    print("tokenize_file", file_path)
    """Tokenize a single file in chunks and return the token IDs."""
    # Initialize tokenizer in each process

    global tokenizer

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return token_ids


def process_exact_deduplicated_to_npy(exact_dedup_dir, output_npy_path):
    """Convert all files in exact_deduplicated directory to a single npy file using max CPU cores."""
    print(f"Processing {exact_dedup_dir} to {output_npy_path}")

    # Collect all text files
    text_files = []
    for root, dirs, files in os.walk(exact_dedup_dir):
        for file in files:
            if file.endswith((".txt",)):
                text_files.append(os.path.join(root, file))

    print(f"Found {len(text_files)} text files to process")

    if not text_files:
        print("No text files found!")
        return

    # Use all available CPU cores (288 in your case)
    # init_worker()
    # for f in text_files:
    #     tokenize_file(f)
    # raise Exception()
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} CPU cores for maximum speed")

    # Process each file on a separate core
    with multiprocessing.Pool(num_processes, initializer=init_worker) as pool:
        print("Starting parallel tokenization of all files...")
        results = list(
            tqdm(pool.imap(tokenize_file, text_files, chunksize=1), total=len(text_files), desc="Tokenizing files")
        )

    print("Merging results in main thread...")

    # Merge all results into a single list
    all_ids = []
    total_tokens = 0
    for file_tokens in tqdm(results, desc="Merging tokenized results"):
        all_ids.extend(file_tokens)
        total_tokens += len(file_tokens)

    print(f"Total tokenized tokens: {total_tokens}")

    # Convert to numpy array and save
    print("Converting to numpy array and saving...")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_npy_path)
    print(f"Saved tokenized data to {output_npy_path}")


def main():
    # Configuration
    # tar -cf - cs336_data/leaderboard/.data/exact_deduplicated | pigz -9 -p 288 > exact_deduplicated.tar.gz
    # tar -cf - cs336_data/leaderboard/.data/processed | pigz -9 -p 288 > processed.tar.gz

    # upload_to_gcp("deduplicated.tar.gz", "test-joan1", "deduplicated.tar.gz")
    upload_to_gcp("processed.tar.gz", "test-joan1", "processed.tar.gz")

    # AutoTokenizer.from_pretrained("gpt2", cache_dir="cs336_data/leaderboard/.cache") ~ cache
    # npy_output = os.path.join(output_dir, "tokenized_data.npy")
    # process_exact_deduplicated_to_npy(os.path.join(DATA_DIR, "exact_deduplicated"), npy_output)
    # upload_to_gcp(npy_output, BUCKET_NAME, "tokenized_data.npy")


if __name__ == "__main__":
    main()
