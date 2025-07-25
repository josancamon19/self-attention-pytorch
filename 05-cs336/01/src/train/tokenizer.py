from collections import defaultdict
import json
from line_profiler import profile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import regex as re
import heapq
import time
from functools import wraps
from typing import BinaryIO
import os

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


# ======================================== #
# ======================================== #
# ======================================== #

@timeit
@profile
def initialize(text: str, special_tokens: list[str]):
    special_tokens = sorted(special_tokens, key=len, reverse=True)  # overlapping issue
    split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
    strings = re.split(split_special_tokens, text)  # [:1]
    # print("initialize:", len(text), len(strings))
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pairs_count = defaultdict(int)
    pretokens_to_split = {}  # b"hi": [b"h", b"i"]
    pretokens_counts = defaultdict(int)
    pairs_to_pretokens = defaultdict(set)

    for string in strings:
        for match in PAT.finditer(string):
            pretoken_bytes = match.group().encode("utf-8")
            if len(pretoken_bytes) == 1:
                continue

            pretoken_split = []
            for j in range(len(pretoken_bytes)):
                pretoken_split.append(pretoken_bytes[j : j + 1])

                if j + 1 < len(pretoken_bytes):
                    pair = (pretoken_bytes[j : j + 1], pretoken_bytes[j + 1 : j + 2])
                    pairs_count[pair] += 1
                    pairs_to_pretokens[pair].add(pretoken_bytes)

            pretokens_to_split[pretoken_bytes] = pretoken_split
            pretokens_counts[pretoken_bytes] += 1

    return pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens


def _initiallize_parallel(file_path: str, f_start, f_end, special_tokens: list[str]):
    with open(file_path, "rb") as f:
        f.seek(f_start)
        text = f.read(f_end - f_start).decode("utf-8", errors="ignore")

    special_tokens = sorted(special_tokens, key=len, reverse=True)  # overlapping issue
    split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
    strings = re.split(split_special_tokens, text)  # [:1]
    print(file_path, f_start, f_end, len(text), len(strings))
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pairs_count = defaultdict(int)
    pretokens_to_split = {}  # b"hi": [b"h", b"i"]
    pretokens_counts = defaultdict(int)
    pairs_to_pretokens = defaultdict(set)

    for string in strings:
        for match in PAT.finditer(string):
            pretoken_bytes = match.group().encode("utf-8")
            if len(pretoken_bytes) == 1:
                continue

            pretoken_split = []
            for j in range(len(pretoken_bytes)):
                pretoken_split.append(pretoken_bytes[j : j + 1])

                if j + 1 < len(pretoken_bytes):
                    pair = (pretoken_bytes[j : j + 1], pretoken_bytes[j + 1 : j + 2])
                    pairs_count[pair] += 1
                    pairs_to_pretokens[pair].add(pretoken_bytes)

            pretokens_to_split[pretoken_bytes] = pretoken_split
            pretokens_counts[pretoken_bytes] += 1

    return pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens


@timeit
def initialize_parallel(input_text_file, boundaries, special_tokens):
    pairs_count = defaultdict(int)
    pretokens_to_split = {}  # b"hi": [b"h", b"i"]
    pretokens_counts = defaultdict(int)
    pairs_to_pretokens = defaultdict(set)

    with ProcessPoolExecutor(mp.cpu_count()) as pool:
        futures = []
        for start, end in boundaries:
            futures.append(pool.submit(_initiallize_parallel, input_text_file, start, end, special_tokens))

        # Collect results and merge them
        for future in futures:
            pairs_count_result, pretokens_to_split_result, pretokens_counts_result, pairs_to_pretokens_result = (
                future.result()
            )

            # Merge pairs_count
            for pair, count in pairs_count_result.items():
                pairs_count[pair] += count

            # Merge pretokens_to_split
            pretokens_to_split.update(pretokens_to_split_result)

            # Merge pretokens_counts
            for pretoken, count in pretokens_counts_result.items():
                pretokens_counts[pretoken] += count

            # Merge pairs_to_pretokens
            for pair, pretokens in pairs_to_pretokens_result.items():
                pairs_to_pretokens[pair].update(pretokens)
    return pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens


@timeit
@profile
def get_max_priority_queue(priority_queue: list):
    # Pop all items with minimum count
    min_items = []
    if not priority_queue:
        return None, None

    # Get the minimum count
    min_count = priority_queue[0][0]

    # Pop all items with the same minimum count
    while priority_queue and priority_queue[0][0] == min_count:
        min_items.append(heapq.heappop(priority_queue))

    max_item = max(min_items, key=lambda x: x[1])

    for item in min_items:
        if item != max_item:
            heapq.heappush(priority_queue, item)

    count = -max_item[0]
    return count, max_item[1]


@timeit
@profile
def update_pairs_count_after_merge(priority_queue, new_created_pairs_count, affected_pairs_count):
    for pair, count in new_created_pairs_count.items():
        priority_queue.append((-count, pair))

    for i, item in enumerate(priority_queue):
        count, pair = -item[0], item[1]
        if pair in affected_pairs_count:
            new_count = -(count - affected_pairs_count[pair])
            priority_queue[i] = (new_count, pair)
    heapq.heapify(priority_queue)


@timeit
@profile
def train_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 300,
    special_tokens: list[str] = ["<|endoftext|>"],
    save_results: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_text_file, "rb") as f:
        # text = f.read().decode("utf-8", errors="ignore")
        # f.seek(0)  # Reset file pointer to beginning
        boundaries = find_chunk_boundaries(f, mp.cpu_count(), b"<|endoftext|>")
        boundaries = zip(boundaries[:-1], boundaries[1:])
        # print("len(text)", len(text))

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    vocab_set = set(vocab.values())

    # pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens = initialize(text, special_tokens)
    pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens = initialize_parallel(
        input_text_file, boundaries, special_tokens
    )
    # return
    priority_queue = [(-count, pair) for pair, count in pairs_count.items()]

    heapq.heapify(priority_queue)
    merges = []

    # print("priority_queue:", priority_queue)

    pbar = tqdm(total=target_vocab_size - len(vocab), desc="Training Tokenizer")
    while len(vocab) < target_vocab_size:
        pcount, pair = get_max_priority_queue(priority_queue)
        pair_bytes = pair[0] + pair[1]
        # print(f"merge {len(merges) + 1}:", pcount, pair)
        merges.append(pair)
        vocab[len(vocab)] = pair_bytes
        vocab_set.add(pair_bytes)

        new_created_pairs_count = defaultdict(int)
        affected_pairs_count = defaultdict(int)

        #  ==== MERGE ====
        matching_pretokens = pairs_to_pretokens[pair]
        affected_pairs_count[pair] = pcount  # remove it all

        for pretoken in matching_pretokens:
            split = pretokens_to_split[pretoken]
            count = pretokens_counts[pretoken]

            i = 0
            updated_split = []
            last_merge = False
            while i < len(split):
                if i + 1 < len(split) and split[i] == pair[0] and split[i + 1] == pair[1]:
                    updated_split.append(pair_bytes)
                    if i > 0:
                        affected_pairs_count[(split[i - 1], split[i])] += count
                        new_pair = (updated_split[-2], updated_split[-1])
                        new_created_pairs_count[new_pair] += count
                        pairs_to_pretokens[new_pair].add(pretoken)

                    last_merge = True
                    i += 2
                else:
                    if last_merge:
                        affected_pairs_count[(split[i - 1], split[i])] += count
                        new_pair = (updated_split[-1], split[i])
                        new_created_pairs_count[new_pair] += count
                        pairs_to_pretokens[new_pair].add(pretoken)
                        last_merge = False

                    updated_split.append(split[i])
                    i += 1

            pretokens_to_split[pretoken] = updated_split

        update_pairs_count_after_merge(
            priority_queue,
            new_created_pairs_count,
            affected_pairs_count,
        )
        pbar.update(1)
    pbar.close()

    if save_results:
        merges_path = input_text_file.replace(".txt", "-merges.json")
        vocab_path = input_text_file.replace(".txt", "-vocab.json")
        with open(merges_path, "w") as f:
            json.dump(merges, f, indent=2, default=str)

        with open(vocab_path, "w") as f:
            json.dump(vocab, f, indent=2, default=str)
    return vocab, merges


if __name__ == "__main__":
    train_tokenizer(
        # input_text_file="data/TinyStoriesV2-GPT4-train.txt",
        input_text_file="data/owt_train.txt",
        target_vocab_size=32000,
        save_results=True,
    )
    print_execution_summary()

# initialize 705 seconds
# train_tokenizer 755 seconds
# most expensive 95% is initialize, lol.
