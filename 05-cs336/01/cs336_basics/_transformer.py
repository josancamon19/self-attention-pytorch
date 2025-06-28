import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

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
    dim_indices = torch.arange(0, embed_dim, 2)
    div_term = 10000 ** (dim_indices / embed_dim)

    # at every position, compute sin/cos
    pe = torch.zeros((max_positions, embed_dim))
    pe[:, 0::2] = torch.sin(positions / div_term[::2])
    pe[:, 1::2] = torch.cos(positions / div_term[1::2])
    print(pe)
    return pe


get_positional_encodings(4, 5)


class SelfAttention(nn.Module):
    def __init__(self, batch_size, embed_dim, head_count):
        super().__init__()
        self.Q = Parameter(nn.init.xavier_uniform(torch.zeros((batch_size,))))
        self.K = Parameter()
        self.V = Parameter()

    def forward(self, x):
        q = self.Q @ x
        k = self.K @ x
        v = self.V @ x

        attention_scores = torch.matmul(q, k)
        attention_weights = torch.softmax(attention_scores / x.shape[-1], dim=-1)

        return attention_weights @ v


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        pass
