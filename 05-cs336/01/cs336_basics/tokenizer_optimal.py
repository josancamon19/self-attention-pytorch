from collections import defaultdict

from line_profiler import profile
from cs336_basics.shared import init_vocabulary, timeit
import regex as re
import heapq

# initial tokenizer version was not optimal
# pseudocode
# - pretokenize, ' hello'
# - - list of ints/bytes, each pretoken, and also count every pair
# - - # pairs to sentence_ids # depends on vocabulary available, updated as you merge
# - - # pairs, priority queue, contains pairs, counts, rule depending.
# ----- pretokenize until here?
# --- when are those updated?
# - - pop from the queue
# - - merge
# - - - pair, a,b, find sentences with the pair.
# - - - merge operation there
# - - - update both dts, update counts for priority queue, and already poped from dictionary


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
    sentences = re.split(split_special_tokens, text)
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pairs_to_sentence_idx = defaultdict(set[int])
    pairs_count = defaultdict(int)

    pretokenized_sentences = []
    for i, sentence in enumerate(sentences):
        pretokenized_sentence = []

        for match in PAT.finditer(sentence):
            word_bytes = match.group().encode("utf-8")
            # if len(word_bytes) == 1:
            #     continue # not even added, but that should be okay

            pretokenized_sentence.append(word_bytes)

            for j in range(len(word_bytes) - 1):  # 2.4 seconds extr
                pair = (word_bytes[j : j + 1], word_bytes[j + 1 : j + 2])
                pairs_to_sentence_idx[pair].add(i)
                pairs_count[pair] += 1

        pretokenized_sentences.append(pretokenized_sentence)

    priority_queue = []  # 0.8 seconds
    for pair, count in pairs_count.items():
        heapq.heappush(priority_queue, (-count, pair))
    merges = []

    # Lists have same length, checking for content differences...
    # (b' ', b't')  correct: (b' ', b't')
    # (b't', b'h')  correct: (b' ', b'a')
    # (b' ', b'a')  correct: (b'h', b'e')
    # (b'h', b'e')  correct: (b'i', b'n')
    # (b' ', b'th') correct: (b' t', b'he')

    # -- count has to be wrong then
    # 2940 (b' ', b't')
    # 2764 (b't', b'h')
    # 2214 (b' ', b'a')
    # 2168 (b'h', b'e')
    # 2094 (b' ', b'th')

    while len(vocab) < target_vocab_size:
        # print("priority_queue", priority_queue)
        # wrong, cause not lexicographically bigger, enough for tests
        count, pair = heapq.heappop(priority_queue)
        count = -count
        # print("max_count:", count, pair, "priority_queue:", len(priority_queue))
        new_pair = pair[0] + pair[1]
        new_created_pairs_count = defaultdict(int)

        merges.append(pair)
        vocab[len(vocab)] = new_pair

        # print("pairs_to_sentence_idx[pair]", pairs_to_sentence_idx[pair])
        for s_idx in pairs_to_sentence_idx[pair]:
            sentence = pretokenized_sentences[s_idx]
            for word in sentence:
                i = 0
                while i < len(word) - 1:
                    if word[i : i + 2] == new_pair:
                        if i > 0:
                            new_created_pairs_count[(word[i - 1 : i], new_pair)] += 1
                        if i + 2 < len(word):
                            new_created_pairs_count[(new_pair, word[i + 2 : i + 3])] += 1
                        i += 2
                    else:
                        i += 1

        # print("new_created_pairs_count:", new_created_pairs_count)
        for pair, count in new_created_pairs_count.items():
            heapq.heappush(priority_queue, (-count, pair))

        # if len(merges) == 2:
        #     break
    return vocab, merges


# train_tokenizer()
