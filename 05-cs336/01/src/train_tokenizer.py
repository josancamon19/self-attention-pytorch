from collections import defaultdict
import json
import os
from line_profiler import profile
from tqdm import tqdm
from src.shared import timeit, print_execution_summary
import regex as re
import heapq


@timeit
@profile
def initialize(text: str, special_tokens: list[str]):
    special_tokens = sorted(special_tokens, key=len, reverse=True)  # overlapping issue
    split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
    strings = re.split(split_special_tokens, text)  # [:1]
    print("initialize len(strings)", len(strings))
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
        text = f.read().decode("utf-8", errors="ignore")

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    vocab_set = set(vocab.values())

    pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens = initialize(text, special_tokens)
    # pairs_count[(b"n", b"a")] = 0
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


# train_tokenizer(input_text_file="data/TinyStoriesV2-GPT4-train.txt", target_vocab_size=10000, save_results=True)
# print_execution_summary()

# initialize 705 seconds
# train_tokenizer 755 seconds
# most expensive 95% is initialize, lol.