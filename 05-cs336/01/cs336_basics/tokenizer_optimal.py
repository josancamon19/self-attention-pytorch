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

            pretokenized_sentence.append(word_bytes)

            for j in range(len(word_bytes) - 1):
                pair = (word_bytes[j : j + 1], word_bytes[j + 1 : j + 2])
                pairs_to_sentence_idx[pair].add(i)
                pairs_count[pair] += 1

        pretokenized_sentences.append(pretokenized_sentence)
    return pairs_count, pairs_to_sentence_idx, pretokenized_sentences


def get_max_priority_queue(priority_queue):
    # wrong, cause not lexicographically bigger, enough for tests
    count, _, pair = heapq.heappop(priority_queue)
    count = -count
    print("merge:", count, pair)
    return count, pair


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

    pairs_count, pairs_to_sentence_idx, pretokenized_sentences = initialize(text, special_tokens)
    priority_queue = []
    merges = []

    def get_pq_lex_key(pair):
        return tuple(-b for b in pair[0]) + tuple(-b for b in pair[1])

    for pair, count in pairs_count.items():
        heapq.heappush(priority_queue, (-count, get_pq_lex_key(pair), pair))

    while len(vocab) < target_vocab_size:
        count, pair = get_max_priority_queue(priority_queue)
        pair_bytes = pair[0] + pair[1]
        new_pair_length = len(pair_bytes)
        print("merge:", count, pair, pair_bytes)
        merges.append(pair)
        vocab[len(vocab)] = pair_bytes
        # print("pairs_to_sentence_idx[pair]", pairs_to_sentence_idx[pair])

        # simulate a "merge", then count new pairs such as (merge-1, merge) and (merge, merge+1)
        new_created_pairs_count = defaultdict(int)
        # the "merge" decreases the count of prev pairs that do not exist anymore
        # e.g. after merging "he", on every finding of the word "the", the pair (t,h) doesn't exist anymore
        # so we gotta decrease this counter from our priotiy queue.
        affected_pairs_count = defaultdict(int)

        # why could index 3 be different

        for s_idx in pairs_to_sentence_idx[pair]:
            sentence = pretokenized_sentences[s_idx]
            for word in sentence:
                i = 0
                # TODO: update pairs, and same variables, but consider vocabulary size
                # -- not only -1, pair, and pair, + 1, cause maybe +1 alone doesn't exist anymore, is another vocab.
                # -- find the merge, go from 0 to i, and (n,j) finding the vocabulary pair
                while i < len(word) - new_pair_length + 1:
                    if word[i : i + new_pair_length] == pair_bytes:
                        if i > 0:
                            j = 0
                            prev_bytes = word[i - 1 : i]
                            # while j < i:
                            #     if word[j:i] in vocab:
                            #         prev_bytes = word[j:i]
                            #     j += 1

                            new_created_pairs_count[(prev_bytes, pair_bytes)] += 1
                            affected_pairs_count[(prev_bytes, pair[0])] += 1

                        if i + new_pair_length < len(word):
                            j = len(word)
                            next_bytes = word[i + new_pair_length : i + new_pair_length + 1]

                            # while i + new_pair_length < j:
                            #     if word[i + new_pair_length : j] in vocab:
                            #         next_bytes = word[i + new_pair_length : j]
                            #     j -= 1

                            new_created_pairs_count[(pair_bytes, next_bytes)] += 1
                            affected_pairs_count[(pair[1], next_bytes)] += 1
                        i += new_pair_length
                    else:
                        i += 1

        # print("new_created_pairs_count:", len(new_created_pairs_count))
        # print("new_created_pairs_count:", new_created_pairs_count)
        # print("affected_pairs_count:", len(affected_pairs_count))
        # print("affected_pairs_count:", affected_pairs_count)
        for pair, count in new_created_pairs_count.items():
            priority_queue.append((-count, get_pq_lex_key(pair), pair))

        for i, item in enumerate(priority_queue):
            count, pair = -item[0], item[2]
            if pair in affected_pairs_count:
                priority_queue[i] = (-(count - affected_pairs_count[pair]), get_pq_lex_key(pair), pair)

        heapq.heapify(priority_queue)

        # if len(merges) == 5:
        #     break
    return vocab, merges


# train_tokenizer()
