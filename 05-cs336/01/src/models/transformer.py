from enum import Enum
import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from src.models.tokenizer import Tokenizer
from src.utils import softmax

# from transformers import GPT2Tokenizer


# We expect you to build these components from scratch. In particular, you may not
# use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
# • torch.nn.Parameter
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)1
# • The torch.optim.Optimizer base class


class PosEmbeddingType(Enum):
    ROPE = "rope"
    NOPE = "nope"
    SINUSOIDAL = "sinusoidal"


class NormType(Enum):
    RMS = "rms"
    LAYER = "layer"
    NONE = "none"


class RMSNormGainType(Enum):
    SCALAR = "scalar"
    ELEMENTWISE = "elementwise"  # ~ only using elementwise
    NONE = "none"


class QKNormType(Enum):
    l2 = "l2"
    rms = "rms"


class NormPosition(Enum):
    PRE = "pre"
    POST = "post"


class FFNType(Enum):
    RELU2 = "relu2"
    SWIGLU = "swiglu"
    SILU = "silu"


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, weight_tying: bool = False):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((vocab_size, embed_dim)))
        # no weight tying 4.2 loss
        # with weight tying
        if weight_tying:
            nn.init.xavier_uniform_(self.embeddings)
        else:
            # nn.init.normal_(self.embeddings, mean=0, std=0.02)
            nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)  # 8.2 loss
        # nn.init.normal_(self.embeddings, mean=0, std=0.02) # 6.2 loss
        # nn.init.xavier_uniform_(self.embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class Linear(nn.Module):
    def __init__(
        self,
        inp: int,
        out: int,
        bias: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
        # num_layers: int = 1 # https://arxiv.org/abs/1908.11365
        is_out_proj: bool = False,
    ):
        super().__init__()
        self.inp = inp
        self.out = out
        self.weights = nn.Parameter(torch.empty((out, inp), device=device, dtype=dtype))
        self.bias = None if not bias else nn.Parameter(torch.zeros((out,), device=device, dtype=dtype))
        self.reset_parameters(is_out_proj)

    def reset_parameters(self, is_out_proj: bool) -> None:
        if is_out_proj:
            nn.init.normal_(self.weights, mean=0, std=0.02)
        else:
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
        original_shape = x.shape
        seq_length, d_k = original_shape[-2:]
        x = x.view(-1, seq_length, d_k)
        batch_size = x.shape[0]

        rope_selected = self.rope_cache[:seq_length]
        x_pairs = x.view(batch_size, seq_length, -1, 2)

        # some black magic where complex numbers i,j parts can be computed like this
        # TODO: this part below is definitely not clear
        # if x_pairs.dtype == torch.bfloat16:
        #     original_dtype = x_pairs.dtype
        #     x_complex = torch.view_as_complex(x_pairs.float())
        #     rope_complex = torch.view_as_complex(rope_selected.float())
        #     rotated_complex = x_complex * rope_complex
        #     rotated_real = torch.view_as_real(rotated_complex).to(original_dtype)
        # else:
        #     # Direct path for FP32
        #     x_complex = torch.view_as_complex(x_pairs)
        #     rope_complex = torch.view_as_complex(rope_selected)
        #     rotated_complex = x_complex * rope_complex
        #     rotated_real = torch.view_as_real(rotated_complex)

        # ===== No Complex Num =====
        cos_vals = rope_selected[..., 0]  # shape: [seq_length, d_k//2]
        sin_vals = rope_selected[..., 1]  # shape: [seq_length, d_k//2]

        x0 = x_pairs[..., 0]  # even indices: [batch, seq_length, d_k//2]
        x1 = x_pairs[..., 1]  # odd indices:  [batch, seq_length, d_k//2]
        # Apply rotation: [x0', x1'] = [x0*cos - x1*sin, x0*sin + x1*cos]
        rotated_x0 = x0 * cos_vals - x1 * sin_vals
        rotated_x1 = x0 * sin_vals + x1 * cos_vals
        rotated_real = torch.stack([rotated_x0, rotated_x1], dim=-1)
        # ===== No Complex Num =====

        result = rotated_real.view(-1, seq_length, d_k)
        return result.view(original_shape)


def rms_norm(x, eps: float = 1e-5):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return torch.divide(x, rms)


class RMSNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
        gain_type: RMSNormGainType = RMSNormGainType.ELEMENTWISE,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gain_type = gain_type

        if gain_type == RMSNormGainType.SCALAR:
            self.gain = Parameter(torch.ones(1))
        elif gain_type == RMSNormGainType.ELEMENTWISE:
            self.gain = Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        normalized = rms_norm(x, self.eps)

        if self.gain is not None:
            normalized = normalized * self.gain

        return normalized.to(x_dtype)


class PosWiseFFN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_type: FFNType = FFNType.SWIGLU,
    ):
        super().__init__()
        # round to 64 closest value

        self.ffn_type = ffn_type
        value = 8 * embedding_dim / 3
        dff = round(value / 64) * 64
        if self.ffn_type == FFNType.SWIGLU:
            self.W1_W3 = Linear(embedding_dim, 2 * dff)  # fused op
            self.W2 = Linear(dff, embedding_dim)
        elif self.ffn_type == FFNType.RELU2:
            self.W1 = Linear(embedding_dim, dff * 4)
            self.W2 = Linear(dff * 4, embedding_dim)
        else:
            self.W1 = Linear(embedding_dim, dff * 4)
            self.W2 = Linear(dff * 4, embedding_dim)

    @staticmethod
    def silu(x):
        return torch.nn.functional.silu(x)
        # return x * torch.sigmoid(x)

    @staticmethod
    def relu2(x):
        return torch.clamp(x, min=0) ** 2

    def forward(self, x):
        if self.ffn_type == FFNType.SWIGLU:
            w1_w3_out = self.W1_W3(x)
            w1_out, w3_out = w1_w3_out.chunk(2, dim=-1)
            return self.W2(self.silu(w1_out) * w3_out)
        elif self.ffn_type == FFNType.RELU2:
            return self.W2(self.relu2(self.W1(x)))
        else:  # SILU
            return self.W2(self.silu(self.W1(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
        pos_embedding: PosEmbeddingType = PosEmbeddingType.ROPE,
        qk_norm: bool = False,
        qk_norm_type: QKNormType = QKNormType.l2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type

        self.QKV = Linear(embedding_dim, 3 * self.head_size * self.num_heads)

        self.register_buffer("causal_mask", torch.tril(torch.ones(max_sequence_length, max_sequence_length)))
        self.scale = 1.0 / math.sqrt(self.head_size)

        # if self.qk_norm:
            # self.qk_scale = nn.Parameter(torch.ones(self.head_size))
        self.qk_scale = nn.Parameter(torch.ones(1))

        # if pos_embedding == PosEmbeddingType.ROPE:
        self.rope = RotaryPositionalEncoding(self.head_size, max_sequence_length)
        # else:
        #     self.rope = None

        self.W_O = Linear(embedding_dim, embedding_dim)

    def _reshape_to_heads(self, batch, seq_length, tensor):
        return tensor.view(batch, seq_length, self.num_heads, self.head_size).transpose(2, 1)

    def forward(self, x, _):
        batch, seq_length = x.shape[0], x.shape[1]  # , embedding_dim
        qkv = self.QKV(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._reshape_to_heads(batch, seq_length, q).contiguous()
        k = self._reshape_to_heads(batch, seq_length, k).contiguous()
        v = self._reshape_to_heads(batch, seq_length, v)

        # if self.rope:  # test logic
        q = self.rope(q)
        k = self.rope(k)

        # if self.qk_norm:
        #     if self.qk_norm_type == QKNormType.l2:
        #         q = F.normalize(q, dim=-1)
        #         k = F.normalize(k, dim=-1)
        #     else:  # rms
        q = rms_norm(q)
        k = rms_norm(k)
        attention_scores = q @ k.transpose(-2, -1)
        attention_scores *= self.qk_scale
        # else:
        #     attention_scores = (q @ k.transpose(-2, -1)) * self.scale  # b, num_heads, seq_length, seq_length

        mask = self.causal_mask[:seq_length, :seq_length]
        # if padding_mask is not None:
        #     mask = (mask * padding_mask.unsqueeze(1)).unsqueeze(1)

        attention_scores = torch.masked_fill(attention_scores, mask == 0, -float("inf"))

        attention_weights = softmax(attention_scores, dim=-1)
        # if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
        #     print("[DEBUG] Attention weights have NaN/Inf after softmax!")

        x = attention_weights @ v
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch, seq_length, -1)
        wo = self.W_O(x)
        return wo


def get_norm_class(norm_type: NormType, dim: int, rms_gain_type=RMSNormGainType.ELEMENTWISE):
    if norm_type == NormType.RMS:
        return RMSNorm(dim, gain_type=rms_gain_type)
    elif norm_type == NormType.LAYER:
        return nn.LayerNorm(dim)
    elif norm_type == NormType.NONE:
        return nn.Identity()


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        max_sequence_length,
        pos_embedding: PosEmbeddingType = PosEmbeddingType.ROPE,
        norm_type: NormType = NormType.RMS,
        norm_position: NormPosition = NormPosition.PRE,
        ffn_type: FFNType = FFNType.SWIGLU,
        qk_norm: bool = False,
        qk_norm_type: QKNormType = QKNormType.l2,
    ):
        super().__init__()

        self.norm_position = norm_position
        self.attention_norm = get_norm_class(norm_type, embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim,
            num_heads,
            max_sequence_length,
            pos_embedding,
            qk_norm,
            qk_norm_type,
        )
        self.pos_wise_norm = get_norm_class(norm_type, embedding_dim)
        self.pos_wise = PosWiseFFN(embedding_dim, ffn_type)

    def forward(self, x, padding_mask):
        # TODO: implement norm position changes
        # if self.norm_position == NormPosition.PRE:
        attention = self.attention(self.attention_norm(x), padding_mask) + x
        output = self.pos_wise(self.pos_wise_norm(attention)) + attention
        # else:  # POST
        #     attention = x + self.attention_norm(self.attention(x, padding_mask))
        #     output = attention + self.pos_wise_norm(self.pos_wise(attention))
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        pos_embedding: PosEmbeddingType = PosEmbeddingType.ROPE,
        norm_type: NormType = NormType.RMS,
        norm_position: NormPosition = NormPosition.PRE,
        ffn_type: FFNType = FFNType.SWIGLU,
        weight_tying: bool = False,
        qk_norm: bool = False,
        qk_norm_type: QKNormType = QKNormType.l2,
    ):
        super().__init__()
        self.pos_embedding = pos_embedding
        self.norm_position = norm_position
        self.weight_tying = weight_tying

        self.embeddings = Embedding(vocab_size, embedding_dim, weight_tying)

        # if self.pos_embedding == PosEmbeddingType.SINUSOIDAL:
        #     pe = get_positional_encodings(embedding_dim, max_sequence_length)
        #     self.register_buffer("pe", pe)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                embedding_dim,
                num_heads,
                max_sequence_length,
                pos_embedding,
                norm_type,
                norm_position,
                ffn_type,
                qk_norm,
                qk_norm_type,
            )
            for _ in range(num_layers)
        )
        # if self.norm_position == NormPosition.PRE:
        self.pre_output_norm = get_norm_class(norm_type, embedding_dim)

        # if not self.weight_tying:
        self.output = Linear(embedding_dim, vocab_size, is_out_proj=True)

    @classmethod
    def from_args(cls, args, return_tokenizer: bool = False):
        tokenizer = Tokenizer.from_files(
            args.tokenizer_vocab_path,
            args.tokenizer_merges_path,
            ["<|endoftext|>"],
        )
        model = cls(
            # TODO: else defaults if args.$param doesn't exists when inference
            tokenizer.vocab_size,
            args.seq_length,
            args.embedding_dim,
            args.num_layers,
            args.num_heads,
            pos_embedding=PosEmbeddingType(args.pos_embedding.lower()),
            norm_type=NormType(args.norm_type.lower()),
            norm_position=NormPosition(args.norm_position.lower()),
            ffn_type=FFNType(args.ffn_type.lower()),
            qk_norm=args.qk_norm,
            qk_norm_type=args.qk_norm_type,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Transformer.from_args]: {total_params} parameters")

        if return_tokenizer:
            return model, tokenizer
        return model

    def forward(self, input_ids, padding_mask):
        tokens = self.embeddings(input_ids)

        # if self.pos_embedding == PosEmbeddingType.SINUSOIDAL:
        #     tokens += self.pe[: input_ids.shape[-1], :]

        for block in self.blocks:
            tokens = block(tokens, padding_mask)

        # if self.norm_position == NormPosition.PRE:
        tokens = self.pre_output_norm(tokens)

        # if self.weight_tying:
        #     output = tokens @ self.embeddings.embeddings.T
        # else:
        output = self.output(tokens)
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
