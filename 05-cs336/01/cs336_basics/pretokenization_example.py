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

num_processes = multiprocessing.cpu_count()


def pretokenize(text: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, text)


def init_vocabulary():
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab, set(vocab.values()), len(vocab)


def get_word_in_vocab_bytes(vocab: set, pretokenized: list[str]) -> list[list[bytes]]:
    # print(f"get_word_in_vocab_bytes: vocab {len(vocab)} pretokenized: {len(pretokenized)}")
    pretokenized_bytes = []

    for word in pretokenized:
        i, j = 0, 1
        vocab_word = []
        while j <= len(word):
            subword = word[i:j].encode("utf-8")
            if subword not in vocab:
                vocab_word.append(word[i : j - 1].encode("utf-8"))
                i = j - 1
                # wtf, models think here I should put j = i+1 ???
            elif j == len(word):
                vocab_word.append(subword)
                j += 1
            else:
                j += 1

        pretokenized_bytes.append(vocab_word)
    return pretokenized_bytes


def merge_step(
    vocab: set,
    vocab_dict: dict,
    pretokenized: list[list[str]],
) -> tuple[bytes, bytes]:
    """
    return merge step made,

    pretokenized: list of sentences, each one split with the pretokenized
    """
    pretokenized_bytes = [get_word_in_vocab_bytes(vocab, sent) for sent in pretokenized]
    # print(f"pretokenized_bytes: {pretokenized_bytes}")
    common = defaultdict(int)
    for sentence in pretokenized_bytes:
        for word in sentence:
            for i in range(len(word) - 1):
                common[(word[i], word[i + 1])] += 1

    max_value = max(common.values())
    new_vocab_item = max([k for k, v in common.items() if v == max_value])
    print("new_vocab_item", new_vocab_item, "count_value", max_value)
    vocab.add(new_vocab_item[0] + new_vocab_item[1])
    vocab_dict[len(vocab) + 1] = new_vocab_item[0] + new_vocab_item[1]
    return new_vocab_item


def get_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 1000,
    special_tokens: list[str] = [],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """returns: vocab, merges"""
    # TODO: remove special tokens before regex pretokenization, avoid merging outside the margin it puts ??

    sep = "<|endoftext|>"
    chunks = []
    with open(input_text_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, sep.encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # print(start, end, len(chunk.split(sep)))
            chunks.append(chunk)
            # Run pre-tokenization on your chunk and store the counts for each pre-token
    print(f"split_chunks: {len(chunks)} chunks")
    print(f"split_chunks len(chunks[0].split(sep)): {len(chunks[0].split(sep))}")
    print(f'split_chunks chunks[0].split(sep)[0]: "{chunks[0].split(sep)[0]}"')
    # TODO: should \n be cleaned by trimming?

    # ----
    # sample = chunks[0].split(sep)[0]
    # matches = pretokenize(sample)
    # pretokenized = [match.group() for match in matches]
    # print(f"pretokenized: {pretokenized}")

    # chunk_0 = [[m.group() for m in pretokenize(sample)] for sample in chunks[0].split(sep)]
    # chunk_0 += ["👀", "は"]
    # vocab = init_vocabulary([chunk_0])
    vocab_dict, vocab, vocab_size = init_vocabulary()
    # while vocab_size < target_vocab_size:
    #     merge_step(vocab, vocab_dict, [pretokenized])
    #     break
    # merges = []
    # for _ in range(12):
    #     merged: tuple[bytes, bytes] = merge_step(vocab, vocab_dict, [pretokenized])
    #     merges.append(merged)
    vocab.add(b"o")
    vocab.add(b"le")
    vocab.add(b"ole")
    pretokenized_bytes = get_word_in_vocab_bytes(vocab, ["mole"])
    print(f"pretokenized_bytes: {pretokenized_bytes}")

    # now with chunks and parallelization what?
    # -- am I processing the vocabulary separately?
    # special tokens and secondary stuff


get_tokenizer()
