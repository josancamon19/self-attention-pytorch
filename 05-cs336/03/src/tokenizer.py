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
        avg_time = (
            stats["total_time"] / stats["call_count"] if stats["call_count"] > 0 else 0
        )
        print(f"{func_name}:")
        print(f"  Total calls: {stats['call_count']}")
        print(f"  Total time: {stats['total_time']:.4f}s")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Min time: {stats['min_time']:.4f}s")
        print(f"  Max time: {stats['max_time']:.4f}s")
        print()


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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
    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

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
    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

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
            futures.append(
                pool.submit(
                    _initiallize_parallel, input_text_file, start, end, special_tokens
                )
            )

        # Collect results and merge them
        for future in futures:
            (
                pairs_count_result,
                pretokens_to_split_result,
                pretokens_counts_result,
                pairs_to_pretokens_result,
            ) = future.result()

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
def update_pairs_count_after_merge(
    priority_queue, new_created_pairs_count, affected_pairs_count
):
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
    pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens = (
        initialize_parallel(input_text_file, boundaries, special_tokens)
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
                if (
                    i + 1 < len(split)
                    and split[i] == pair[0]
                    and split[i + 1] == pair[1]
                ):
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


from collections.abc import Iterable, Iterator
from line_profiler import profile
import regex as re
import json
import torch
import numpy as np
from tqdm import tqdm


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        special_tokens = special_tokens or []
        special_tokens = sorted(special_tokens, key=len, reverse=True)

        self.vocab_reversed = {v: k for k, v in vocab.items()}

        for st in special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.vocab_reversed:
                continue

            vocab[len(vocab)] = st_bytes
            self.vocab_reversed[st_bytes] = len(vocab)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_size = len(vocab)
        self.pad_id = (
            self.vocab_reversed[special_tokens[-1].encode("utf-8")]
            if special_tokens
            else None
        )
        # --
        self.split_special_tokens = "|".join(
            re.escape(token) for token in self.special_tokens
        )
        self.PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Create merge priority map for O(1) lookup instead of O(n) iteration
        self.merge_priority = {merge: i for i, merge in enumerate(merges)}

    # TODO: the following 2 methods were initially imp by me, but were a bottleneck
    # - claude implemented this ones
    # - WOW, gotta review this code, reduced time here 1/20th
    @profile
    def encode_batched(
        self,
        batch: list[str],
        truncation: bool = False,
        max_sequence_length: int = 0,
        padding: bool = True,
    ):
        encoded = []
        for item in batch:
            if (
                truncation
                and max_sequence_length
                and len(item) > max_sequence_length * 4
            ):
                item = item[: max_sequence_length * 4]  # Rough char->token ratio

            item_enc = self.encode(item)

            if truncation and max_sequence_length:
                item_enc = item_enc[:max_sequence_length]

            encoded.append(item_enc)

        if not encoded:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
            }

        # Find max length for padding
        max_length = (
            max(len(seq) for seq in encoded)
            if padding
            else max_sequence_length or max(len(seq) for seq in encoded)
        )

        batch_size = len(encoded)
        input_ids = torch.full((batch_size, max_length), self.pad_id, dtype=torch.long)
        attention_mask = (
            torch.zeros((batch_size, max_length), dtype=torch.long)
            if padding
            else torch.ones((batch_size, max_length), dtype=torch.long)
        )

        # Fill tensors efficiently - avoid creating individual tensors
        for i, seq in enumerate(encoded):
            seq_len = len(seq)
            if seq_len > 0:
                # Convert to tensor once and use slice assignment
                input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
                if padding:
                    attention_mask[i, :seq_len] = (
                        1  # Real tokens get 1, padding stays 0
                    )
                else:
                    attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @profile
    def encode(self, input_text: str, show_progress: bool = True) -> list[int]:
        if self.special_tokens:
            strings = re.split(self.split_special_tokens, input_text)
            special_tokens_sep = [
                m.group().encode("utf-8")
                for m in re.finditer(self.split_special_tokens, input_text)
            ]
        else:
            strings, special_tokens_sep = [input_text], []

        tokenized = []
        for si, string in tqdm(
            enumerate(strings),
            desc="Tokenizer.encode",
            total=len(strings),
            disable=not show_progress,
        ):
            for match in self.PAT.finditer(string):
                pretoken_bytes = match.group().encode("utf-8")

                # Optimized BPE merge using priority queue
                tokens = [bytes([b]) for b in pretoken_bytes]

                while len(tokens) >= 2:
                    # Find all possible pairs and their priorities
                    pairs = []
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        if pair in self.merge_priority:
                            pairs.append((self.merge_priority[pair], i, pair))

                    if not pairs:
                        break

                    # Get the highest priority merge (lowest index)
                    priority, pos, (first, second) = min(pairs)

                    # Apply the merge
                    tokens[pos] = first + second
                    del tokens[pos + 1]

                # Convert to token IDs
                tokenized.extend(self.vocab_reversed[token] for token in tokens)

            # Add special token between strings
            if si < len(strings) - 1:
                tokenized.append(self.vocab_reversed[special_tokens_sep[si]])

        return tokenized

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            token_ids = self.encode(chunk)
            yield from token_ids

    def decode(self, ids: list[int]):
        # print("[Tokenizer.decode] ids:", ids)
        decoded_bytes = b"".join([self.vocab[_id] for _id in ids])
        decoded = decoded_bytes.decode("utf-8")
        # print("[Tokenizer.decode] decoded:", decoded)
        return decoded

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=["<|endoftext|>"],
    ):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {int(k): eval(v) for k, v in vocab.items()}

        merges = []
        with open(merges_filepath) as f:
            merges = json.load(f)
            merges = [(eval(b1), eval(b2)) for b1, b2 in merges]

        return cls(vocab, merges, special_tokens)


if __name__ == "__main__":
    _type = "train"
    vocab = "data/slimpajama_sample_100M-vocab.json"
    merges = "data/slimpajama_sample_100M-merges.json"
    text_data = "data/slimpajama_sample_100M.txt"
    output_path = "data/slimpajama_sample_100M.npy"

    tokenizer = Tokenizer.from_files(vocab, merges)
    with open(text_data, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")

    output = tokenizer.encode(content)
    output_np = np.array(output, dtype=np.uint16)
    np.save(output_path, output_np)
    print(f"Saved tokenized output with shape {output_np.shape} to {output_path}")
