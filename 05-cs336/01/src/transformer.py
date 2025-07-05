import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from transformers import GPT2Tokenizer


# We expect you to build these components from scratch. In particular, you may not
# use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
# • torch.nn.Parameter
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)1
# • The torch.optim.Optimizer base class


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((vocab_size, embed_dim)))
        nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class Linear(nn.Module):
    def __init__(self, inp: int, out: int, bias: bool = False, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.inp = inp
        self.out = out
        self.weights = nn.Parameter(torch.empty((out, inp), device=device, dtype=dtype))
        self.bias = None if not bias else nn.Parameter(torch.zeros((out,), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.inp + self.out))
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        if self.bias is not None:
            return x @ self.weights.T + self.bias
        return x @ self.weights.T


def get_positional_encodings(embed_dim, max_positions=4):
    positions = torch.arange(max_positions).unsqueeze(1)
    dim_indices = torch.arange(0, embed_dim, 2)  # TODO: not sure if this part is clear
    div_term = 10000 ** (dim_indices / embed_dim)

    pe = torch.empty((max_positions, embed_dim))
    pe[:, 0::2] = torch.sin(positions / div_term)
    pe[:, 1::2] = torch.cos(positions / div_term)
    return pe


class RotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        theta: float = 10000,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        assert d_k % 2 == 0
        i = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        theta_i = theta ** (-2 * i / d_k)
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)

        thetas = positions * theta_i
        cosines = torch.cos(thetas)
        sines = torch.sin(thetas)
        rope_cache = torch.stack([cosines, sines], dim=-1)
        self.register_buffer("rope_cache", rope_cache)

    def forward(self, x, token_positions=None):
        # print("RoPE.forward x.shape", x.shape, self.rope_cache.shape)
        original_shape = x.shape
        seq_length, d_k = original_shape[-2:]
        x = x.view(-1, seq_length, d_k)
        # print("x.shape", x.shape)
        batch_size = x.shape[0]

        # if token_positions.dim() == 1:
        #     token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        # token_positions_flat = token_positions.view(-1, seq_length)
        rope_selected = self.rope_cache[:seq_length]
        x_pairs = x.view(batch_size, seq_length, -1, 2)
        # print("x_pairs.shape", x_pairs.shape)
        # print("rope_selected.shape", rope_selected.shape)

        # some black magic where complex numbers i,j parts can be computed like this
        # TODO: this part below is definitely not clear
        x_complex = torch.view_as_complex(x_pairs)
        rope_complex = torch.view_as_complex(rope_selected)
        rotated_complex = x_complex * rope_complex
        rotated_real = torch.view_as_real(rotated_complex)
        result = rotated_real.view(-1, seq_length, d_k)
        return result.view(original_shape)


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        # TODO: understand reasoning, not only implement
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gain = Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        # TODO: gpt recommends formula is not clear, review
        # --- identify variance/means comp and name properly
        x_dtype = x.dtype
        x = x.to(torch.float32)
        tsum = torch.sum(torch.pow(x, 2), dim=-1)
        div_term = torch.sqrt((1 / self.embedding_dim) * tsum + self.eps).unsqueeze(-1)
        result = torch.divide(x, div_term) * self.gain
        return result.to(x_dtype)


class PosWiseFFN(nn.Module):
    # TODO: understand the reasoning of SiLU and SwiGLU
    # We offer no explanation as to why these architectures seem to work; we attribute their success,
    # as all else, to divine benevolence
    def __init__(self, embedding_dim: int):
        super().__init__()
        # round to 64 closest value
        value = 8 * embedding_dim / 3
        dff = round(value / 64) * 64
        self.W1 = Linear(embedding_dim, dff)
        self.W2 = Linear(dff, embedding_dim)
        self.W3 = Linear(embedding_dim, dff)

    @staticmethod
    def silu(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        silu = PosWiseFFN.silu(self.W1(x))
        return self.W2(silu * self.W3(x))


def softmax(tensor: torch.Tensor, dim: int = 0):
    max_vals = torch.max(tensor, dim=dim, keepdim=True)[0]
    num_part = torch.exp(tensor - max_vals)
    div_term = torch.sum(num_part, dim=dim, keepdim=True)
    return num_part / div_term


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, max_sequence_length):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.embedding_dim = embedding_dim

        # TODO: what's the most optimal way of implementing this?
        self.Q = Linear(embedding_dim, self.head_size * self.num_heads)
        self.K = Linear(embedding_dim, self.head_size * self.num_heads)
        self.V = Linear(embedding_dim, self.head_size * self.num_heads)
        self.rope = RotaryPositionalEncoding(self.head_size, max_sequence_length)

        self.W_O = Linear(embedding_dim, embedding_dim)

    def _reshape_to_heads(self, batch, seq_length, tensor):
        return tensor.view(batch, seq_length, self.num_heads, self.head_size).transpose(2, 1)

    def forward(self, x, padding_mask):
        # print(x)
        batch, seq_length = x.shape[0], x.shape[1]  # , embedding_dim
        q = self._reshape_to_heads(batch, seq_length, self.Q(x)).contiguous()
        k = self._reshape_to_heads(batch, seq_length, self.K(x)).contiguous()
        v = self._reshape_to_heads(batch, seq_length, self.V(x))

        if self.rope:  # test logic
            q = self.rope(q)
            k = self.rope(k)

        attention_scores = q @ k.transpose(-2, -1)  # b, num_heads, seq_length, seq_length
        attention_scores = attention_scores / math.sqrt(self.head_size)

        mask = torch.tril(torch.ones((seq_length, seq_length))).to(q.device)
        if padding_mask is not None:
            # print("mask.shape:", mask.shape)
            mask = (mask * padding_mask.unsqueeze(1)).unsqueeze(1)
            # print("mask.shape:", mask.shape)
            # print("attention_scores.shape:", attention_scores.shape)

        attention_scores = torch.masked_fill(attention_scores, mask == 0, -float("inf"))
        attention_weights = softmax(attention_scores, dim=-1)
        x = attention_weights @ v
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch, seq_length, -1)
        wo = self.W_O(x)
        # print("MultiHeadSelfAttention.forward wo.shape:", wo.shape)
        return wo


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, max_sequence_length):
        super().__init__()
        self.attention_norm = RMSNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, max_sequence_length)
        self.pos_wise_norm = RMSNorm(embedding_dim)
        self.pos_wise = PosWiseFFN(embedding_dim)

    def forward(self, x, padding_mask):
        attention = self.attention(self.attention_norm(x), padding_mask) + x
        output = self.pos_wise(self.pos_wise_norm(attention)) + attention
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        # pe = get_positional_encodings(embedding_dim, max_sequence_length)
        # self.register_buffer("pe", pe)

        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads, max_sequence_length) for _ in range(num_layers)
        )
        self.pre_output_norm = RMSNorm(embedding_dim)
        self.output = nn.Parameter(torch.empty(embedding_dim, vocab_size))
        nn.init.normal_(self.output, std=0.02)

    def forward(self, input_ids, padding_mask):
        tokens = self.embeddings(input_ids)  #  + self.pe[: input_ids.shape[-1], :]
        for block in self.blocks:
            tokens = block(tokens, padding_mask)

        tokens = self.pre_output_norm(tokens)
        output = tokens @ self.output
        return output  # output logits


# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
# max_sequence_length = 100
# embedding_dim = 128
# num_layers = 1
# num_heads = 8

# tokenizer.pad_token = "[PAD]"
# tokenized = tokenizer(
#     ["Hi there, this is a test", "hey"],
#     return_tensors="pt",
#     padding=True,
#     truncation=True,
# )
# model = Transformer(tokenizer.vocab_size, max_sequence_length, embedding_dim, num_layers, num_heads)
# model(tokenized["input_ids"], tokenized["attention_mask"])
