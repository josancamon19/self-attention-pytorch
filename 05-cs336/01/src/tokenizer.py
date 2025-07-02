from collections.abc import Iterable, Iterator
import regex as re
import json

import torch


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        padding_token: str = None,
    ):
        # print(
        #     f"[Tokenizer.__init__] vocab_size: {len(vocab)}, "
        #     f"merges_size: {len(merges)}, special_tokens: {special_tokens}, pad_token: {padding_token}",
        # )
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

    def encode_batched(
        self,
        batch: list[str],
        truncation: bool = False,
        max_sequence_length: int = 0,
        padding: bool = True,
    ):
        # Encode all sequences
        encoded = []
        for item in batch:
            # Only truncate text if it's extremely long (char-level optimization)
            if truncation and max_sequence_length and len(item) > max_sequence_length * 4:
                item = item[: max_sequence_length * 4]  # Rough char->token ratio

            item_enc = self.encode(item)

            # Truncate tokens if needed
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

        # Pre-allocate tensors for efficiency
        batch_size = len(encoded)
        input_ids = torch.full((batch_size, max_length), self.pad_id, dtype=torch.long)
        attention_mask = (
            torch.zeros((batch_size, max_length), dtype=torch.long)
            if padding
            else torch.ones((batch_size, max_length), dtype=torch.long)
        )

        # Fill tensors
        for i, seq in enumerate(encoded):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            if padding:
                attention_mask[i, :seq_len] = 1  # Real tokens get 1, padding stays 0
            else:
                attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, input_text: str) -> list[int]:
        if self.special_tokens:
            strings = re.split(self.split_special_tokens, input_text)
            special_tokens_sep = [m.group().encode("utf-8") for m in re.finditer(self.split_special_tokens, input_text)]
        else:
            strings, special_tokens_sep = [input_text], []

        tokenized = []
        for si, string in enumerate(strings):
            for match in self.PAT.finditer(string):
                pretoken_bytes = match.group().encode("utf-8")

                # Fast BPE merge algorithm
                tokens = [bytes([b]) for b in pretoken_bytes]

                for merge in self.merges:
                    if len(tokens) < 2:
                        break

                    # Find all merge positions in one pass
                    merge_positions = []
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                            merge_positions.append(i)
                            i += 2  # Skip the pair to avoid overlapping merges
                        else:
                            i += 1

                    # Apply merges from right to left to preserve indices
                    for pos in reversed(merge_positions):
                        tokens[pos] = merge[0] + merge[1]
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
        special_tokens=None,
    ):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {int(k): eval(v) for k, v in vocab.items()}

        merges = []
        with open(merges_filepath) as f:
            merges = json.load(f)
            merges = [(eval(b1), eval(b2)) for b1, b2 in merges]

        return cls(vocab, merges, special_tokens)


# Tokenizer.from_files("data/TinyStoriesV2-GPT4-valid-vocab.json", "data/TinyStoriesV2-GPT4-valid-merges.json")

# TODO: parallelize encode
# TODO: train on tinystories dataset, vocabsize 10k (store to disk)
# TODO: profile the code
# TODO: parallelize training

# TODO: train on openwebtext dataset.
# TODO: 2.7 experiments.
