# print(f'chr(0): "{chr(0)}"')
# print(f"chr(0): {chr(0).__repr__()}")

# ----

# print("this is a test" + chr(0) + "string")

# unicode standard 150k vocab
# unicode encoding, takes a character into bytes, utf-8 (dominant on internet)

# print("--------")
# test_string = "hello! こんにちは!"
# utf8_encoded = test_string.encode("utf-8")
# print(utf8_encoded)
# print(type(utf8_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
# list(utf8_encoded)
# [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
# One byte does not necessarily correspond to one Unicode character!
# print(len(test_string))
# print(len(utf8_encoded))
# print(utf8_encoded.decode("utf-8"))

# vocabulary size of 256, bytes can have 256 possible values
# 1 byte = 8 bits [0 or 1 at each state] 2^8, 256

# why utf-8 vs 16/32
# -- utf instead of unicodes, limited vocab size, and composability, utf-16/32 have 2/4x the size, most internet


# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])


# print('"にち".encode("utf-8"):', "にち".encode("utf-8"))
# print("bytes([b]): ", bytes(["にち".encode("utf-8")[0]]))
# output = decode_utf8_bytes_to_str_wrong("にち".encode("utf-8"))  # noqa: UP012
# print("decode_utf8_bytes_to_str_wrong:", output)

# when a character requires more than 1 byte it fails.

# C, 2 byte sequence that doesn't decode to any unicode char ??
# -- any 2 bytes that go independently?

# ----

# print("-------")
# BPE, as a midpoint between byte level tokenizer and word level tokenizer.

# import regex as re  # noqa: E402
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# re.findall(PAT, "some text that i'll pre-tokenize")


# def bpe_example(merges: int = 5):
#     from collections import Counter, defaultdict  # noqa: E402

#     corpus = """
#     low low low low low
#     lower lower widest widest widest
#     newest newest newest newest newest newest
#     """.replace("\n", " ")
#     pretokenization = [w.strip() for w in corpus.split(" ") if w.strip()]
#     count = Counter(pretokenization)
#     print("pretokenized counter:", count)
#     vocabulary = set([bytes([b]) for w in set(pretokenization) for b in w.encode("utf-8")])
#     print("vocabulary:", vocabulary)
#     print()

#     def merge():
#         pretokenization_bytes = []
#         for w in pretokenization:
#             w = [bytes([b]) for b in w.encode("utf-8")]
#             word = []
#             i, j = 0, 1
#             while j <= len(w):
#                 subw = b"".join(w[i:j])
#                 if subw not in vocabulary:
#                     word.append(b"".join(w[i : j - 1]))
#                     i = j - 1
#                 elif j == len(w):
#                     word.append(b"".join(w[i:j]))
#                     j += 1
#                 else:
#                     j += 1
#             pretokenization_bytes.append(word)
#         print("pretokenization_bytes:", pretokenization_bytes, "...")
#         print("pretokenization_bytes[0][0]:", pretokenization_bytes[0][0], type(pretokenization_bytes[0][0]))

#         common_pairs = defaultdict(int)
#         for w in pretokenization_bytes:
#             for i in range(len(w)):
#                 if i == len(w) - 1:
#                     break
#                 j = i + 1
#                 if w[i] + w[j] not in vocabulary:
#                     common_pairs[w[i] + w[j]] += 1

#         if not common_pairs:
#             return True
#         print("common_pairs:", common_pairs)
#         max_count = max(common_pairs.values())
#         max_pairs = [pair for pair, count in common_pairs.items() if count == max_count]
#         print(f"Most common pairs (count={max_count}): {max_pairs}")
#         print(f"New Vocab Word: {max(max_pairs)}")
#         vocabulary.add(max(max_pairs))
#         print("updated vocabulary:", vocabulary)
#         print("-")

#     for i in range(merges):
#         if merge():
#             print(f"nothing else to merge at {i} iter")
#             break


# bpe_example(6)


# 2.7!!

# import time
# from src.tokenizer import Tokenizer


# tinystories_tokenizer = Tokenizer.from_files(
#     ".tokenizer/TinyStoriesV2-GPT4-train-vocab.json",
#     ".tokenizer/TinyStoriesV2-GPT4-train-merges.json",
# )

# owt_tokenizer = Tokenizer.from_files(
#     ".tokenizer/owt_train-vocab.json",
#     ".tokenizer/owt_train-merges.json",
# )

# with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
#     tinystories_dataset = f.read().decode("utf-8", errors="ignore")

# with open("data/owt_valid.txt", "rb") as f:
#     owt_dataset = f.read().decode("utf-8", errors="ignore")

# tinystories_samples = tinystories_dataset.split("<|endoftext|>")
# owt_samples = owt_dataset.split("<|endoftext|>")

# total_bytes, total_encoded = 0, 0
# for sample in tinystories_samples[:10]:
#     sample_bytes = sample.encode("utf-8")
#     encoded = tinystories_tokenizer.encode(sample, False)
#     total_bytes += len(sample_bytes)
#     total_encoded += len(encoded)

# print("tinystories tokenizer has a compresion ratio:", total_bytes / total_encoded, "bytes/token")

# total_bytes, total_encoded = 0, 0
# for sample in owt_samples[:10]:
#     sample_bytes = sample.encode("utf-8")
#     encoded = owt_tokenizer.encode(sample, False)
#     total_bytes += len(sample_bytes)
#     total_encoded += len(encoded)

# print("owt tokenizer has a compresion ratio:", total_bytes / total_encoded, "bytes/token")

# owt_sample = owt_samples[0]
# owt_sample_bytes = owt_sample.encode("utf-8")
# sample_tinystories_encoded = tinystories_tokenizer.encode(owt_sample, False)
# sample_owt_encoded = owt_tokenizer.encode(owt_sample, False)

# print(f"owt sample (len:{len(owt_sample_bytes)}) compared compression ratio")
# print(f"tinystories tokenizer: {len(owt_sample_bytes) / len(sample_tinystories_encoded)}")
# print(f"owt tokenizer: {len(owt_sample_bytes) / len(sample_owt_encoded)}")

# print("----")
# target_mbs = 1 * 1024 * 1024
# owt_big_sample = owt_dataset[:target_mbs]
# owt_big_sample_bytes = owt_big_sample.encode("utf-8")
# print(f"owt sample size: {len(owt_big_sample_bytes) / (1024 * 1024)} MB")
# start = time.time()
# encoded = owt_tokenizer.encode(owt_big_sample, True)
# print(f"encoding {len(owt_big_sample_bytes)} bytes took {time.time() - start} seconds")

# print("----")
# token_id, token_bytes = max(owt_tokenizer.vocab.items(), key=lambda kv: len(kv[1]))
# print(f"longest item in owt tokenizer is {token_id}, length: {len(token_bytes)}")
# print(f"longest item token_bytes: {token_bytes}")
# print(f"longest item contents: {token_bytes.decode('utf-8')}")


# 3.6 transformer resource accounting

# vocab_size : 50,257
# context_length : 1,024
# num_layers : 48
# d_model : 1,600
# num_heads : 25
# d_ff : 6400

# Embedding: vocab_size * d_model = 80411200
# 1 Layer = 40963200
# - Attention = 10240000 (4 * d_model^2)
# -- Q,K,V: 3 * d_model * d_model
# -- W_O = d_model * d_model
# - 2 RMS Norm = 2 * d_model = 3200
# - PosWise FFN = 30720000 (3*d_model*dff)
# -- w1, w3 = d_model * dff
# -- w2 = dff * d_model
# output norm: 1 * d_model
# output: d_model * vocab_size = 80411200

# 1. trainable parameters?
# 80411200 +
# 48 * 40963200 = 1,966,233,600
# + 80411200
# = 2,126,902,400

# 2. memory in FP32 (single precision) with XL
# parameters * bytes? = 8,507,609,600 = 8.5GB to load.

# 3. matmuls, forwardpass
# $seq_length

# matmuls
#  A: (m, k), B: (k, n) → Output: (m, n)
# - FLOPs = m × n × k × 2

# - x = seq_length, d_model
# - 1 block
# -- attention
# - x @ q,k,v = 3 * seq_length * d_model * d_model
# - scores (q @ k) = seq_length * d_model * seq_length
# - scores @ v = seq_length * d_model * seq_length
# - wo = seq_length * d_model * d_model
# - = 3 * seq_length * d_model^2 + 2 * seq_length^2 * d_model
# - = seq_length * 7,680,000 + 3200 * seq_length^2
# -- poswise = 3 * seq_length * d_model * dff = 30720000 * seq_length
# - x @ w1,w3 = 2 * seq_length * dff * d_model
# - above @ w2 = seq_length * d_model * dff
# -- output = seq_length * vocab_size * d_model = 80411200

# 48*(15360000 * seq_length + 6400 * seq_length ^ 2 + 61,440,000 * seq_length) + seq_length * 160822400
def compute_flops(seq_length: int):
    attention = 48 * (15360000 * seq_length + 6400 * seq_length**2)
    mlp = 48 * (61440000 * seq_length)
    output = seq_length * 160822400

    total_flops = attention + mlp + output

    attention_pct = (attention / total_flops) * 100
    mlp_pct = (mlp / total_flops) * 100
    output_pct = (output / total_flops) * 100

    print(f"🔍 Sequence Length: {seq_length:,}")
    print(f"⚡ Attention FLOPs: {attention:,} ({attention_pct:.1f}%)")
    print(f"🧠 MLP FLOPs: {mlp:,} ({mlp_pct:.1f}%)")
    print(f"📤 Output FLOPs: {output:,} ({output_pct:.1f}%)")
    print(f"🎯 Total FLOPs: {total_flops:,}")
    print("=" * 50)

    return total_flops


# compute_flops(1)
# compute_flops(10)
# compute_flops(100)
# compute_flops(1000)
# compute_flops(10000)
# compute_flops(16384)


# --------
# Forward activations
# - sizes at each
# q,k,v = 3 * num_heads * seq_length * head_size = 4800 * seq_length values
# q @ k = num_heads * seq_length * seq_length = 25 * seq_length ^ 2
# softmax = same as above = 25 * seq_length ^ 2
# attention weights @ v = num_heads * seq_length * head_size =1600 * seq_length
# W_O = seq_length * d_model 1600 * seq_length

# = 8000 * seq_length + 50 * seq_length ** 2

# - residual = activations sum ocuppy the same?
# - MLP
# -- w1x = seq_length, dff
# -- activation = ""
# -- w3x = seq_length, dff
# -- w2x = seq_length, d_model

#

# - Do you store the norms as well and activations as same size?
