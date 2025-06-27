from collections.abc import Iterable, Iterator
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # print(
        #     f"[Tokenizer.__init__] vocab_size: {len(vocab)}, merges_size: {len(merges)}, special_tokens: {special_tokens}",
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
        # --
        self.split_special_tokens = "|".join(re.escape(token) for token in self.special_tokens)
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, input_text: str) -> list[int]:
        # print("[Tokenizer.encode] input_text:", input_text)
        if self.special_tokens:
            strings = re.split(self.split_special_tokens, input_text)
            special_tokens_sep = [m.group().encode("utf-8") for m in re.finditer(self.split_special_tokens, input_text)]
            # print("[Tokenizer.encode] special_tokens_sep:", special_tokens_sep)
        else:
            strings, special_tokens_sep = [input_text], []

        pretokens = set()
        pretokenized_strings = []
        for si, string in enumerate(strings):
            pretokenized = []
            for match in self.PAT.finditer(string):
                pretoken_bytes = match.group().encode("utf-8")
                pretokens.add(pretoken_bytes)
                pretokenized.append(pretoken_bytes)
            pretokenized_strings.append(pretokenized)

        pretokens_map = {}
        for pretoken in pretokens:
            pretoken_bytes = [bytes([b]) for b in pretoken]
            for merge in self.merges:
                i = 0
                while i < len(pretoken_bytes) - 1:
                    if pretoken_bytes[i] == merge[0] and pretoken_bytes[i + 1] == merge[1]:
                        pretoken_bytes = pretoken_bytes[:i] + [merge[0] + merge[1]] + pretoken_bytes[i + 2 :]
                    else:
                        i += 1

            pretokens_map[pretoken] = [self.vocab_reversed[pb] for pb in pretoken_bytes]

        # print(pretokens_map)
        tokenized = []
        for si, string in enumerate(pretokenized_strings):
            for pretoken_bytes in string:
                tokenized.extend(pretokens_map[pretoken_bytes])

            if si < len(strings) - 1:
                tokenized.append(self.vocab_reversed[special_tokens_sep[si]])

        # print("[Tokenizer.encode] tokenized:", tokenized)
        # print("[Tokenizer.encode] tokenized.pre:", [self.vocab[i] for i in tokenized])
        return tokenized

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError()

    def decode(self, ids: list[int]):
        # print("[Tokenizer.decode] ids:", ids)
        decoded_bytes = b"".join([self.vocab[_id] for _id in ids])
        decoded = decoded_bytes.decode("utf-8")
        # print("[Tokenizer.decode] decoded:", decoded)
        return decoded

    @staticmethod
    def from_files(vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError()
