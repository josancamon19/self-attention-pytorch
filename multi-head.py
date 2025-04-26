# words related in longer sentences
# parallel attention heads

# each head computes it's own k,q,v
# each head is connected after using an MLP, with last dimension of d_model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):
        q = self.W_q(encodings_q)
        k = self.W_k(encodings_k)
        v = self.W_v(encodings_v)

        similarity_scores = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        scaled_similarity_scores = similarity_scores / math.sqrt(k.size(self.col_dim))
        if mask is not None:
            scaled_similarity_scores = scaled_similarity_scores.masked_fill(
                mask == 0, float("-inf")
            )

        # percentage of influence of each token on each other token
        attention_weights = F.softmax(scaled_similarity_scores, dim=self.col_dim)
        # attention scores
        attention_output = torch.matmul(attention_weights, v)

        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1, num_heads=1):
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim) for _ in range(num_heads)]
        )
        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v):
        ## run the data through all of the attention heads
        return torch.cat(
            [
                head(encodings_for_q, encodings_for_k, encodings_for_v)
                for head in self.heads
            ],
            dim=self.col_dim,
        )


if __name__ == "__main__":
    encodings_q = torch.tensor([[1.16, 0.23], [0.23, 0.34], [0.34, 0.45]])
    encodings_k = torch.tensor([[1.16, 0.23], [0.23, 0.34], [0.34, 0.45]])
    encodings_v = torch.tensor([[1.16, 0.23], [0.23, 0.34], [0.34, 0.45]])

    torch.manual_seed(42)

    # mask = torch.tril(torch.ones(3, 3))
    mha = MultiHeadAttention(d_model=2, num_heads=3)
    print(mha(encodings_q, encodings_k, encodings_v))


# - understand properly this class (smth is odd)
# - expand on encoder decoder only
# - can fully use it? can expand with cursor, and train it? and make it generate?
