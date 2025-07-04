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
        self.pad_id = self.vocab_reversed[special_tokens[-1].encode("utf-8")] if special_tokens else None
        # --
        self.split_special_tokens = "|".join(re.escape(token) for token in self.special_tokens)
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
            if truncation and max_sequence_length and len(item) > max_sequence_length * 4:
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
            max(len(seq) for seq in encoded) if padding else max_sequence_length or max(len(seq) for seq in encoded)
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
                    attention_mask[i, :seq_len] = 1  # Real tokens get 1, padding stays 0
                else:
                    attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @profile
    def encode(self, input_text: str, show_progress: bool = True) -> list[int]:
        if self.special_tokens:
            strings = re.split(self.split_special_tokens, input_text)
            special_tokens_sep = [m.group().encode("utf-8") for m in re.finditer(self.split_special_tokens, input_text)]
        else:
            strings, special_tokens_sep = [input_text], []

        tokenized = []
        for si, string in tqdm(
            enumerate(strings), desc="Tokenizer.encode", total=len(strings), disable=not show_progress
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
    # dataset = "tinystories"
    dataset = "owt"
    if dataset == "tinystories":
        vocab = ".tokenizer/TinyStoriesV2-GPT4-train-vocab.json"
        merges = ".tokenizer/TinyStoriesV2-GPT4-train-merges.json"
        text_data = f"data/TinyStoriesV2-GPT4-{_type}.txt"
        output_path = f".tokenizer/TinyStoriesV2-GPT4-{_type}-encoded.npy"
    else:
        vocab = ".tokenizer/owt_train-vocab.json"
        merges = ".tokenizer/owt_train-merges.json"
        text_data = f"data/owt_{_type}.txt"
        output_path = f".tokenizer/owt_{_type}-encoded.npy"

    tokenizer = Tokenizer.from_files(vocab, merges)
    with open(text_data, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")

    output = tokenizer.encode(content)
    output_np = np.array(output, dtype=np.uint16)
    np.save(output_path, output_np)
    print(f"Saved tokenized output with shape {output_np.shape} to {output_path}")
