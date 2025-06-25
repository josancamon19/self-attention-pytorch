from collections import defaultdict
import os
from typing import BinaryIO
import time
from functools import wraps
import regex as re


# Global dictionary to track execution statistics for each function
execution_stats = {}


def timeit(func):
    """Decorator to measure execution time of a function and track cumulative statistics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time

        # Initialize stats for this function if not exists
        if func.__name__ not in execution_stats:
            execution_stats[func.__name__] = {
                "total_time": 0.0,
                "call_count": 0,
                "min_time": float("inf"),
                "max_time": 0.0,
            }

        # Update statistics
        stats = execution_stats[func.__name__]
        stats["total_time"] += execution_time
        stats["call_count"] += 1
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)

        # print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result

    return wrapper


def print_execution_summary():
    """Print a summary of all function execution statistics."""
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)

    for func_name, stats in execution_stats.items():
        avg_time = stats["total_time"] / stats["call_count"] if stats["call_count"] > 0 else 0
        print(f"{func_name}:")
        print(f"  Total calls: {stats['call_count']}")
        print(f"  Total time: {stats['total_time']:.4f}s")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Min time: {stats['min_time']:.4f}s")
        print(f"  Max time: {stats['max_time']:.4f}s")
        print()


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(text: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, text)


def init_vocabulary():
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab, set(vocab.values())


# ========================================
# ========================================
# ========================================


def update_pretokenized_with_new_merge(
    _merge: tuple[bytes, bytes],
    sample: list[list[bytes]],  # sentence, each word, as a list of bytes
) -> list[list[bytes]]:
    merge_first, merge_second = _merge
    merge = merge_first + merge_second

    pretokenized_merged = []
    for word in sample:
        if len(word) == 1:
            # this stupid thing, reduced 1 whole second!!
            pretokenized_merged.append(word)
            continue

        merged_word = []
        i = 0
        while i < len(word):
            # 0.3 seconds reduced, instead of (word[i] + word[j]) == merge, creating a new byte
            if i + 1 < len(word) and (word[i] == merge_first and word[i + 1] == merge_second):
                merged_word.append(merge)
                i += 2
            else:
                merged_word.append(word[i])
                i += 1
        pretokenized_merged.append(merged_word)
        # print(merged_word)

    return pretokenized_merged


@timeit
def merge_step(
    vocab: set,
    vocab_dict: dict,
    pretokenized: list[list[list[bytes]]],
) -> tuple[bytes, bytes]:
    """
    return merge step made,
    pretokenized: list of sentences, each one split with the pretokenized
    """

    @timeit
    def count_bigrams():  # 99% of the function time
        common = defaultdict(int)
        for sample in pretokenized:
            for word in sample:
                for i in range(len(word) - 1):
                    common[(word[i], word[i + 1])] += 1
        return common

    common = count_bigrams()

    if not common:
        return None
    merge = max(common.items(), key=lambda x: x[1])[0]
    vocab_dict[len(vocab)] = merge[0] + merge[1]
    vocab.add(merge[0] + merge[1])
    return merge


@timeit
def update_pretokenized_with_merge(
    merge: tuple[bytes, bytes],
    pretokenized: list[list[list[bytes]]],
):
    print("update_pretokenized_with_merge")
    return [update_pretokenized_with_new_merge(merge, s) for s in pretokenized]


@timeit
def get_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 350,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """returns: vocab, merges"""

    assert "<|endoftext|>" in special_tokens

    chunks = []
    with open(input_text_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token

    vocab_dict, vocab = init_vocabulary()
    for st in special_tokens:
        vocab_dict[len(vocab)] = st.encode("utf-8")
        vocab.add(st.encode("utf-8"))

    @timeit
    def pretokenize_all(chunk) -> list[list[list[bytes]]]:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        chunk_samples = re.split(pattern, chunk)  # [:1]
        return [
            [[c.encode("utf-8") for c in match.group()] for match in pretokenize(sample)] for sample in chunk_samples
        ]

    chunk = chunks[0]
    pretokenized = pretokenize_all(chunk)
    merges = []
    while len(vocab) < target_vocab_size:
        merge = merge_step(vocab, vocab_dict, pretokenized)
        if not merge:
            break
        merges.append(merge)
        pretokenized = update_pretokenized_with_merge(merge, pretokenized)

    return vocab_dict, merges


# get_tokenizer()
