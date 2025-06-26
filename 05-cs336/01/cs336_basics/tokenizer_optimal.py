from collections import defaultdict
from line_profiler import profile
from cs336_basics.shared import init_vocabulary, timeit
import regex as re
import heapq


@timeit
# @profile
def train_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 300,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_text_file, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    vocab, _ = init_vocabulary()
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # TODO: optimize use of ints/bytes
    split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
    sentences = re.split(split_special_tokens, text)  # [:1]
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pairs_to_sentence_idx = defaultdict(set[int])
    # TODO: can improve performance by keeping (sentence, pair) = [word_indices]
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

    priority_queue = []  # 0.8 seconds
    for pair, count in pairs_count.items():
        heapq.heappush(priority_queue, (-count, pair))
    merges = []

    while len(vocab) < target_vocab_size:
        # print("priority_queue", priority_queue)
        # wrong, cause not lexicographically bigger, enough for tests
        count, pair = heapq.heappop(priority_queue)
        count = -count
        print("merge:", count, pair)
        new_pair = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = new_pair

        # print("pairs_to_sentence_idx[pair]", pairs_to_sentence_idx[pair])
        new_created_pairs_count = defaultdict(int)
        affected_pairs_count = defaultdict(int)

        for s_idx in pairs_to_sentence_idx[pair]:
            sentence = pretokenized_sentences[s_idx]
            for word in sentence:
                i = 0
                while i < len(word) - 1:
                    if word[i : i + 2] == new_pair:
                        curr_pair_0 = word[i : i + 1]
                        curr_pair_1 = word[i + 1 : i + 2]

                        if i > 0:
                            prev_byte = word[i - 1 : i]
                            new_created_pairs_count[(prev_byte, new_pair)] += 1
                            affected_pairs_count[(prev_byte, curr_pair_0)] += 1
                        if i + 2 < len(word):
                            next_byte = word[i + 2 : i + 3]
                            new_created_pairs_count[(new_pair, next_byte)] += 1
                            affected_pairs_count[(curr_pair_1, next_byte)] += 1
                        i += 2
                    else:
                        i += 1

        print("new_created_pairs_count:", len(new_created_pairs_count))
        print("new_created_pairs_count:", new_created_pairs_count)
        print("affected_pairs_count:", len(affected_pairs_count))
        print("affected_pairs_count:", affected_pairs_count)
        for pair, count in new_created_pairs_count.items():
            heapq.heappush(priority_queue, (-count, pair))

        for i, item in enumerate(priority_queue):
            count = -item[0]
            pair = item[1]

            if pair in affected_pairs_count:
                priority_queue[i] = (-(count - affected_pairs_count[pair]), pair)

        heapq.heapify(priority_queue)

        if len(merges) == 5:
            break
    return vocab, merges


# train_tokenizer()
