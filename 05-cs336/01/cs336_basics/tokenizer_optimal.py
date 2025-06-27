from collections import defaultdict
from line_profiler import profile
from cs336_basics.shared import init_vocabulary, timeit
import regex as re
import heapq


# TODO: can improve performance by keeping (sentence, pair) = [word_indices]
# TODO: optimize use of ints/bytes
def initialize(text: str, special_tokens: list[str]):
    split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
    sentences = re.split(split_special_tokens, text)  # [:1]
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pairs_to_sentence_idx = defaultdict(set[int])
    pairs_count = defaultdict(int)

    pretokenized_sentences = []
    for i, sentence in enumerate(sentences):
        pretokenized_sentence = []

        for match in PAT.finditer(sentence):
            word_bytes = match.group().encode("utf-8")
            if len(word_bytes) == 1:
                continue  # saves 0.04 seconds on test1

            # pretokenized_sentence.append(word_bytes)
            word = []
            for j in range(len(word_bytes) - 1):
                pair = (word_bytes[j : j + 1], word_bytes[j + 1 : j + 2])
                pairs_to_sentence_idx[pair].add(i)
                pairs_count[pair] += 1
                word.append(word_bytes[j : j + 1])

            word.append(word_bytes[len(word_bytes) - 1 :])
            pretokenized_sentence.append(word)

        pretokenized_sentences.append(pretokenized_sentence)
    return pairs_count, pairs_to_sentence_idx, pretokenized_sentences


def get_pq_lex_key(pair):
    combined = pair[0] + pair[1]
    # For reverse lexicographical order, prepend with negative length
    # This ensures longer sequences come first when prefixes match
    # return (-len(combined),) + tuple(combined)
    return 0
    # return tuple(-b for b in pair[0]) + tuple(-b for b in pair[1])


def get_max_priority_queue(priority_queue: list):
    # wrong, cause not lexicographically bigger, enough for tests
    # count, _, pair = heapq.heappop(priority_queue)
    # count = -count
    # TODO: this is super inefficient
    min_count = min(item[0] for item in priority_queue)
    candidates = [item for item in priority_queue if item[0] == min_count]
    max_item = max(candidates, key=lambda x: x[2])
    priority_queue.remove(max_item)
    count = -max_item[0]
    return count, max_item[2]


def update_pairs_count_after_merge(priority_queue, new_created_pairs_count, affected_pairs_count, logs=False):
    if logs:
        print("update_pairs_count_after_merge")
        print("new_created_pairs_count:", len(new_created_pairs_count))

    for pair, count in new_created_pairs_count.items():
        priority_queue.append((-count, get_pq_lex_key(pair), pair))

    if logs:
        print("#")
        print("affected_pairs_count:", len(affected_pairs_count))
    for i, item in enumerate(priority_queue):
        count, pair = -item[0], item[2]
        if pair in affected_pairs_count:
            new_count = -(count - affected_pairs_count[pair])
            priority_queue[i] = (new_count, get_pq_lex_key(pair), pair)

    if logs:
        print("----")
    heapq.heapify(priority_queue)


@timeit
# @profile
def train_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 300,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_text_file, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    vocab_set = set(vocab.values())

    pairs_count, pairs_to_sentence_idx, pretokenized_sentences = initialize(text, special_tokens)
    # print("pretokenized_sentences", pretokenized_sentences)
    priority_queue = [(-count, get_pq_lex_key(pair), pair) for pair, count in pairs_count.items()]
    heapq.heapify(priority_queue)
    merges = []

    while len(vocab) < target_vocab_size:
        count, pair = get_max_priority_queue(priority_queue)
        pair_bytes = pair[0] + pair[1]
        print("merge:", count, pair, pair_bytes)
        merges.append(pair)
        vocab[len(vocab)] = pair_bytes
        vocab_set.add(pair_bytes)

        new_created_pairs_count = defaultdict(int)
        affected_pairs_count = defaultdict(int)

        #  ==== MERGE ====
        indices = pairs_to_sentence_idx[pair]
        # print("pairs_to_sentence_idx",pairs_to_sentence_idx)
        for s_idx in list(indices):
            sentence = pretokenized_sentences[s_idx]
            for w_idx, word in enumerate(sentence):
                i = 0
                updated_word = []
                while i < len(word):
                    if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                        updated_word.append(pair_bytes)

                        if i > 0:
                            old_pair = (word[i - 1], pair[0])
                            new_pair = (word[i - 1], pair_bytes)
                            affected_pairs_count[old_pair] += 1
                            new_created_pairs_count[new_pair] += 1

                            pairs_to_sentence_idx[new_pair].add(s_idx)
                            try:
                                pairs_to_sentence_idx[old_pair].remove(s_idx)
                            except KeyError:
                                pass

                        if i + 2 < len(word):
                            old_pair = (pair[1], word[i + 2])
                            new_pair = (pair_bytes, word[i + 2])
                            affected_pairs_count[old_pair] += 1
                            new_created_pairs_count[new_pair] += 1

                            pairs_to_sentence_idx[new_pair].add(s_idx)
                            try:
                                pairs_to_sentence_idx[old_pair].remove(s_idx)
                            except KeyError:
                                pass

                        i += 2
                    else:
                        updated_word.append(word[i])
                        i += 1

                # print(word, "---", updated_word)
                pretokenized_sentences[s_idx][w_idx] = updated_word
        # break
        update_pairs_count_after_merge(
            priority_queue,
            new_created_pairs_count,
            affected_pairs_count,
            False,
        )
        # print(priority_queue)

    return vocab, merges


# FUCK, hadn't I really test this way, fuck me.

file = "cs336_basics/sample_file.txt"
print("merges_current:", train_tokenizer(file, 300)[1])
from cs336_basics.tokenizer import train_tokenizer as correct

print("correct merges:", correct(file, 300)[1])
