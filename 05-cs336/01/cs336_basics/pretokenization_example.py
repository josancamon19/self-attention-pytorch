from collections import defaultdict
import os
from typing import BinaryIO, Tuple


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


import multiprocessing  # noqa: E402
import regex as re  # noqa: E402
import time
from functools import wraps

num_processes = multiprocessing.cpu_count()


def timeit(func):
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result

    return wrapper


def pretokenize(text: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, text)


def init_vocabulary():
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab, set(vocab.values())


def pretokenized_to_vocab_bytes(vocab: set, pretokenized: list[str]) -> list[list[bytes]]:
    # print(f"get_word_in_vocab_bytes: vocab {len(vocab)} pretokenized: {len(pretokenized)}")
    pretokenized_bytes = []
    for word in pretokenized:
        i, j = 0, len(word)
        vocab_word = []
        while i < j:
            subword = word[i:j].encode("utf-8")
            if subword in vocab:
                vocab_word.append(subword)
                i = j
                j = len(word)
            else:
                j -= 1

        pretokenized_bytes.append(vocab_word)
    return pretokenized_bytes


@timeit
def merge_step(
    vocab: set,
    vocab_dict: dict,
    pretokenized: list[list[str]],
) -> tuple[bytes, bytes]:
    """
    return merge step made,
    pretokenized: list of sentences, each one split with the pretokenized
    """
    pretokenized_bytes = [pretokenized_to_vocab_bytes(vocab, sent) for sent in pretokenized]
    # print(f"pretokenized_bytes: {pretokenized_bytes}")
    common = defaultdict(int)
    for sentence in pretokenized_bytes:
        for word in sentence:
            for i in range(len(word) - 1):
                common[(word[i], word[i + 1])] += 1
    if not common:
        return None
    max_value = max(common.values())
    new_vocab_item = max([k for k, v in common.items() if v == max_value])
    print("new_vocab_item", new_vocab_item, "count_value", max_value)
    vocab_dict[len(vocab)] = new_vocab_item[0] + new_vocab_item[1]
    vocab.add(new_vocab_item[0] + new_vocab_item[1])
    return new_vocab_item


def get_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 300,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """returns: vocab, merges"""

    assert "<|endoftext|>" in special_tokens

    chunks = []
    with open(input_text_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
            # Run pre-tokenization on your chunk and store the counts for each pre-token
    print(f"split_chunks: {len(chunks)} chunks")

    vocab_dict, vocab = init_vocabulary()
    for st in special_tokens:
        vocab_dict[len(vocab)] = st
        vocab.add(st)

    def pretokenize_all():
        # removing/splitting on any special token
        pattern = "|".join(re.escape(token) for token in special_tokens)
        chunk_samples = re.split(pattern, chunks[0])
        print(chunk_samples[0])
        return [[match.group() for match in pretokenize(sample)] for sample in chunk_samples]

    pretokenized = pretokenize_all()
    merges = []
    while len(vocab) < target_vocab_size:
        # TODO: parallelizing, suggesting to do bytes pretoken from here, then merge separate
        merged = merge_step(vocab, vocab_dict, pretokenized)
        if not merged:
            break
        merges.append(merged)
    
    # TODO: how to in parallel?

    return vocab_dict, merges


get_tokenizer()
