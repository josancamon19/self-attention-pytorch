import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

nn.Linear
# We expect you to build these components from scratch. In particular, you may not
# use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
# • torch.nn.Parameter
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)1
# • The torch.optim.Optimizer base class


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((vocab_size, embed_dim)))
        nn.init.uniform_(self.embeddings)  # -0.1, 0.1
        # print(self.embeddings)

    def forward(self, token_ids: torch.tensor):
        # How does this operation work
        return self.embeddings[token_ids]


# emb = EmbeddingLayer(5, 2)
# s1 = torch.arange(0, 5)
# s2 = torch.ones(5, dtype=torch.int)
# s3 = torch.tensor([1, 2, 3, 4, -1])

# single = emb(s1)
# batched = emb(torch.stack([s1, s2, s3]))
# print("single.shape", single.shape)
# print("batched.shape", batched.shape)
# print("equal:", torch.equal(single, batched[0]))


def get_positional_encodings(embed_dim, max_positions=3):
    positions = torch.arange(max_positions).unsqueeze(1)
    # print(positions)
    dim_indices = torch.arange(0, embed_dim, 2)
    # print(dim_indices)
    div_term = 10000 ** (dim_indices / embed_dim)
    # print(div_term)
    # print(positions / div_term)

    pe = torch.empty((max_positions, embed_dim))
    pe[:, 0::2] = torch.sin(positions / div_term)
    pe[:, 1::2] = torch.cos(positions / div_term)
    # print(pe)
    return pe


# get_positional_encodings(6, 5)

embedding_dim = 128
sequence_length = 5

seq = torch.tensor([25, 32, 40, 41, 41])
seq_batched = torch.stack([seq, seq, seq])

# TODO: padding mask
emb = EmbeddingLayer(1000, embedding_dim)
pos_enc = get_positional_encodings(embedding_dim, 100)
emb_pos_enc = emb(seq) + pos_enc[: len(seq)]
# print(emb_pos_enc.shape) # 5, embedding_dim
emb_pos_enc = emb(seq_batched) + pos_enc[: len(seq)]
print("emb_pos_enc.shape:", emb_pos_enc.shape)  # 3, 5, embedding_dim


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        shape = (embedding_dim, head_size)
        # print("SelfAttention.__init__:", shape)
        self.Q = Parameter(torch.zeros(shape))
        self.K = Parameter(torch.zeros(shape))
        self.V = Parameter(torch.zeros(shape))

    def forward(self, x):
        # batch, seq_length, embedding_dim
        q = x @ self.Q
        k = x @ self.K
        v = x @ self.V

        # TODO: Causal Mask
        # print("SelfAttention.forward q.shape:", q.shape)
        attention_scores = q @ k.transpose(1, 2)
        # print("SelfAttention.forward attention_scores:", attention_scores.shape)
        attention_weights = torch.softmax(attention_scores / math.sqrt(k.shape[-1]), dim=-1)
        # print("SelfAttention.forward attention_weights:", attention_weights.shape)
        output = attention_weights @ v
        # print("SelfAttention.forward output:", output.shape)
        return output


# sa = SelfAttention(128, 8)
# sa(emb_pos_enc)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        # TODO: No need for SelfAttention block, just the other one is more annoying on dimensions, use this one for now
        self.attention = nn.Sequential(*[SelfAttention(embedding_dim, num_heads) for _ in range(num_heads)])
        self.W_O = nn.Parameter(torch.zeros((embedding_dim, embedding_dim)))

    def forward(self, x):
        # print(x.shape)
        output = [head(x) for head in self.attention]
        output = torch.cat(output, dim=-1)
        wo = output @ self.W_O
        print("MultiHeadSelfAttention.forward wo.shape:", wo.shape)
        return wo


mhsa = MultiHeadSelfAttention(embedding_dim, 8)
mhsa(emb_pos_enc)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embedding_dim, num_heads)
        #  3,5,128
        # pass each seq input through a ffn, `128, 128`?
        self.mlp_in = nn.Parameter(torch.zeros((embedding_dim, 4 * embedding_dim)))
        self.mlp_in_bias = nn.Parameter(torch.zeros((embedding_dim * 4,)))
        self.relu = F.relu  # TODO: should be implemented manually I guess, and uses swiGLU iirc
        self.mlp_out = nn.Parameter(torch.zeros((4 * embedding_dim, embedding_dim)))
        self.mlp_out_bias = nn.Parameter(torch.zeros((embedding_dim,)))

    def forward(self, x):
        # TODO: normalize
        attention = self.mhsa(x) + x
        # TODO: normalize
        proj_in = attention @ self.mlp_in + self.mlp_in_bias
        proj_in_activated = self.relu(proj_in)
        proj_out = proj_in_activated @ self.mlp_out + self.mlp_out_bias
        output = proj_out + attention
        return output


# block = TransformerBlock(embedding_dim, 8)
# block(emb_pos_enc)


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
        self.embeddings = EmbeddingLayer(vocab_size, embedding_dim)
        self.pe = get_positional_encodings(embedding_dim, max_sequence_length)
        # TODO: register pytorch buffer, device movement
        self.blocks = nn.ModuleList(TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers))
        # norm
        self.output = nn.Parameter(torch.zeros(embedding_dim, vocab_size))

    def forward(self, input_ids):
        print(self.pe[: len(input_ids), :].shape)
        tokens = self.embeddings(input_ids) + self.pe[: input_ids.shape[-1], :]
        for block in self.blocks:
            tokens = block(tokens)

        output = tokens @ self.output
        print("output.shape:", output.shape)
        # return torch.softmax(output, dim=-1)
        return output  # output logits


tf = Transformer(1000, 100, embedding_dim, 2, 8)
prob_dist = tf(seq_batched)
