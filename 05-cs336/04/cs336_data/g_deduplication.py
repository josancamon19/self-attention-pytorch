# Part 1
# websites with UGC, have menu all over, links, etc.
# heuristic, keep only unique lines in the corpus
# use hashing to save memory, 1 pass to count unique, 1 pass to remove dups

import hashlib
import os
from pathlib import Path
from collections import defaultdict
import random
import re
import unicodedata
from typing import List, Set, Tuple  # noqa: UP035


def exact_deduplication(paths: list[str], output_dir: str):
    """
    Performs exact line deduplication across a corpus of files.

    First pass: Count frequency of each line across all files using hashing to save memory.
    Second pass: Rewrite each file keeping only lines that appear exactly once in the corpus.

    Args:
        paths: List of file paths to process
        output_dir: Directory to write deduplicated files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # First pass: Count line frequencies using hashes to save memory
    line_hash_counts = defaultdict(int)

    for file_path in paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Hash the line to save memory
                line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                line_hash_counts[line_hash] += 1

    # Second pass: Rewrite files keeping only unique lines
    for file_path in paths:
        input_path = Path(file_path)
        output_path = Path(output_dir) / input_path.name

        with open(file_path, "r", encoding="utf-8") as input_f, open(output_path, "w", encoding="utf-8") as output_f:
            for line in input_f:
                line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                # Only keep lines that appear exactly once across the entire corpus
                if line_hash_counts[line_hash] == 1:
                    output_f.write(line)


# fuzzy document-level deduplication,
# we will use minhash with locality sensitive hashing (LSH)
# templated content
# Jaccard similarty, between docs ngrams, JS between S and T, is defined as
# |S ∩ T |/|S ∪ T |
# 1. you can't do n^2 checks, neither store n-grams per doc, too expensive

# we use hash/signatures of a doc, that approx how similar those 2 signatures are
# minhash
# does it really matter to understand this in much detail? doubt it tbh
# claude implementing


def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing, removing punctuation, normalizing whitespaces,
    removing accents, and applying NFD unicode normalization.
    """
    # Apply NFD unicode normalization
    text = unicodedata.normalize("NFD", text)

    # Remove accents (combining characters)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and normalize whitespaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def generate_ngrams(text: str, n: int) -> Set[str]:
    """Generate n-grams from text (word-level)."""
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()

    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.add(ngram)

    return ngrams


def hash_function(text: str, seed: int) -> int:
    """Generate a hash function with a given seed."""
    return hash((text, seed)) & 0x7FFFFFFF  # Keep positive


def compute_minhash_signature(ngrams: Set[str], num_hashes: int) -> List[int]:
    """Compute minhash signature for a set of n-grams."""
    signature = []

    for i in range(num_hashes):
        min_hash = float("inf")
        for ngram in ngrams:
            h = hash_function(ngram, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash if min_hash != float("inf") else 0)

    return signature


def get_bands(signature: List[int], num_bands: int) -> List[Tuple[int, ...]]:
    """Divide signature into bands for LSH."""
    r = len(signature) // num_bands  # rows per band
    bands = []

    for i in range(num_bands):
        start = i * r
        end = start + r
        band = tuple(signature[start:end])
        bands.append(band)

    return bands


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union


def find_connected_components(edges: List[Tuple[int, int]], num_docs: int) -> List[List[int]]:
    """Find connected components using Union-Find."""
    parent = list(range(num_docs))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build union-find structure
    for i, j in edges:
        union(i, j)

    # Group by root parent
    components = defaultdict(list)
    for i in range(num_docs):
        components[find(i)].append(i)

    return list(components.values())


def minhash_deduplication(
    paths: List[str],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_dir: str,
):
    """
    Performs fuzzy document deduplication using MinHash and LSH.

    Args:
        paths: List of file paths to process
        num_hashes: Number of hash functions for minhash signatures
        num_bands: Number of bands for LSH
        ngrams: N-gram length (in words) for computing signatures
        jaccard_threshold: Jaccard similarity threshold for marking duplicates
        output_dir: Directory to write deduplicated files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read and normalize all documents
    documents = []
    normalized_texts = []
    document_ngrams = []

    for file_path in paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)

            # Normalize text
            normalized = normalize_text(text)
            normalized_texts.append(normalized)

            # Generate n-grams
            ngrams_set = generate_ngrams(normalized, ngrams)
            document_ngrams.append(ngrams_set)

    # Compute minhash signatures
    signatures = []
    for ngrams_set in document_ngrams:
        signature = compute_minhash_signature(ngrams_set, num_hashes)
        signatures.append(signature)

    # Apply LSH to find candidate pairs
    band_buckets = defaultdict(list)

    for doc_idx, signature in enumerate(signatures):
        bands = get_bands(signature, num_bands)
        for band_idx, band in enumerate(bands):
            # Use band content and band index as bucket key
            bucket_key = (band_idx, band)
            band_buckets[bucket_key].append(doc_idx)

    # Collect candidate pairs
    candidate_pairs = set()
    for bucket_docs in band_buckets.values():
        if len(bucket_docs) > 1:
            # Add all pairs from this bucket
            for i in range(len(bucket_docs)):
                for j in range(i + 1, len(bucket_docs)):
                    candidate_pairs.add((bucket_docs[i], bucket_docs[j]))

    # Compute true Jaccard similarity for candidates and find duplicates
    duplicate_edges = []
    for i, j in candidate_pairs:
        similarity = jaccard_similarity(document_ngrams[i], document_ngrams[j])
        if similarity >= jaccard_threshold:
            duplicate_edges.append((i, j))

    # Find connected components (clusters of duplicates)
    clusters = find_connected_components(duplicate_edges, len(documents))

    # Randomly select one document from each cluster to keep
    docs_to_keep = set()
    for cluster in clusters:
        if len(cluster) == 1:
            # Single document, keep it
            docs_to_keep.add(cluster[0])
        else:
            # Multiple documents, randomly select one
            selected = random.choice(cluster)
            docs_to_keep.add(selected)

    # Write deduplicated files
    for i, file_path in enumerate(paths):
        if i in docs_to_keep:
            input_path = Path(file_path)
            output_path = Path(output_dir) / input_path.name

            with open(output_path, "w", encoding="utf-8") as output_f:
                output_f.write(documents[i])
