import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# doesn't now the tokens after and get's trained to predict them
# Decoder only model

# new matrix M
# MaskedAttention(Q,K,V,M) = SoftMax((Q*K^T/sqrt(d_model)) + M) * V * M

# M is a matrix of -inf where the values are masked
# so if "I am human" is the input, the Q*K^T should have -inf masked for "am" and "human"
# because we don't want to give away the future tokens


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encoding, mask):
        # token_encodings = word embeddings + positional encodings for each token
        q = self.W_q(token_encoding)
        k = self.W_k(token_encoding)
        v = self.W_v(token_encoding)

        similarity_scores = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        scaled_similarity_scores = similarity_scores / math.sqrt(k.size(self.col_dim))
        scaled_similarity_scores = scaled_similarity_scores.masked_fill(
            mask == 0, float("-inf")
        )

        # percentage of influence of each token on each other token
        attention_weights = F.softmax(scaled_similarity_scores, dim=self.col_dim)
        # attention scores
        attention_output = torch.matmul(attention_weights, v)

        return attention_output


if __name__ == "__main__":
    encodings_matrix = torch.tensor([[1.16, 0.23], [0.23, 0.34], [0.34, 0.45]])
    mask = torch.tril(torch.ones(3, 3))
    masked_self_attention = MaskedSelfAttention()
    print(masked_self_attention(encodings_matrix, mask))
