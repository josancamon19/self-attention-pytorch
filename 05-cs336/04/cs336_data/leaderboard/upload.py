#!/usr/bin/env python3

import os
import tarfile
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from google.cloud import storage
from transformers import AutoTokenizer


def compress_directory(dir_path, output_path):
    """Compress a directory into a tar.gz file."""
    print(f"Compressing {dir_path} to {output_path}")
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))
    print(f"Compression complete: {output_path}")


def upload_to_gcp(file_path, bucket_name, blob_name):
    """Upload a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"Uploading {file_path} to gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(file_path)
    print(f"Upload complete: gs://{bucket_name}/{blob_name}")


def tokenize_chunk(args):
    """Tokenize a chunk of text."""
    text_chunk, tokenizer = args
    return tokenizer.encode(text_chunk)


def process_exact_deduplicated_to_npy(exact_dedup_dir, output_npy_path):
    """Convert all files in exact_deduplicated directory to a single npy file."""
    print(f"Processing {exact_dedup_dir} to {output_npy_path}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Collect all text files
    text_files = []
    for root, dirs, files in os.walk(exact_dedup_dir):
        for file in files:
            if file.endswith((".txt", ".jsonl", ".json")):
                text_files.append(os.path.join(root, file))

    print(f"Found {len(text_files)} text files to process")

    # Read all content from all files as continuous text
    all_text = ""
    for file_path in tqdm(text_files, desc="Reading files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Total text length: {len(all_text)} characters")

    # Split text into chunks for multiprocessing (to avoid memory issues)
    chunk_size = 100000  # 100k characters per chunk
    text_chunks = [all_text[i : i + chunk_size] for i in range(0, len(all_text), chunk_size)]

    print(f"Split into {len(text_chunks)} chunks for processing")

    # Tokenize using multiprocessing
    pool = multiprocessing.Pool(1)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 1

    # Prepare arguments for multiprocessing
    args_list = [(chunk, tokenizer) for chunk in text_chunks]

    results = []
    for result in tqdm(
        pool.imap(tokenize_chunk, args_list, chunksize=chunksize),
        total=len(text_chunks),
        desc="Tokenizing chunks",
    ):
        results.append(result)

    pool.close()
    pool.join()

    # Flatten the list of ids and convert to numpy array
    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized into {len(all_ids)} tokens")

    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_npy_path)
    print(f"Saved tokenized data to {output_npy_path}")


def main():
    # Configuration
    DATA_DIR = "cs336_data/leaderboard/.data"
    BUCKET_NAME = "test-joan1"

    processed_dir = os.path.join(DATA_DIR, "processed")
    exact_dedup_dir = os.path.join(DATA_DIR, "exact_deduplicated")

    # Check if directories exist
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} does not exist")
        return

    if not os.path.exists(exact_dedup_dir):
        print(f"Error: {exact_dedup_dir} does not exist")
        return

    # Create output directory for compressed files
    output_dir = DATA_DIR + "/compressed_uploads"
    os.makedirs(output_dir, exist_ok=True)

    # Compress directories
    processed_tar = os.path.join(output_dir, "processed.tar.gz")
    exact_dedup_tar = os.path.join(output_dir, "exact_deduplicated.tar.gz")

    compress_directory(processed_dir, processed_tar)
    compress_directory(exact_dedup_dir, exact_dedup_tar)

    # Upload to GCP
    try:
        upload_to_gcp(processed_tar, BUCKET_NAME, "processed.tar.gz")
        upload_to_gcp(exact_dedup_tar, BUCKET_NAME, "exact_deduplicated.tar.gz")
    except Exception as e:
        print(f"Error uploading to GCP: {e}")
        print("Make sure you have set up GCP authentication and updated BUCKET_NAME")

    # Convert exact_deduplicated to npy
    npy_output = os.path.join(output_dir, "exact_deduplicated_tokenized.npy")
    try:
        process_exact_deduplicated_to_npy(exact_dedup_dir, npy_output)

        # Upload the npy file as well
        upload_to_gcp(npy_output, BUCKET_NAME, "tokenized_data.npy")
    except Exception as e:
        print(f"Error processing to npy: {e}")


if __name__ == "__main__":
    main()
