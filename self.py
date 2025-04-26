import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        # d_model = n word embedding values per token ~ size of weight matrices

        super().__init__()
        # why attention paper doesn't use bias, fuck around and find out?
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encoding):
        # token_encodings = word embeddings + positional encodings for each token
        q = self.W_q(token_encoding)
        k = self.W_k(token_encoding)
        v = self.W_v(token_encoding)

        similarity_scores = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        scaled_similarity_scores = similarity_scores / math.sqrt(k.size(self.col_dim))

        # percentage of influence of each token on each other token
        attention_weights = F.softmax(scaled_similarity_scores, dim=self.col_dim)
        # attention scores
        attention_output = torch.matmul(attention_weights, v)

        return attention_output


if __name__ == "__main__":
    encodings_matrix = torch.tensor([[1.16, 0.23], [0.23, 0.34], [0.34, 0.45]])
    self_attention = SelfAttention()
    print(self_attention(encodings_matrix))

    # creates context aware embeddings as output 
    # Encoder only 
