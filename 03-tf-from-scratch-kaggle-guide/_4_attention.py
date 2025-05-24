import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _1_tokenization import tokenize_input
from _2_embedding import embed, config
from _3_positional_encoding import add_positional_encoding


def scaled_dot_product_attention(query, key, value):
    # attention formula basically
    dim_k = query.size(-1)  # d_k
    scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)  # W_Q
        self.k = nn.Linear(embed_dim, head_dim)  # W_K
        self.v = nn.Linear(embed_dim, head_dim)  # W_V

    def forward(self, hidden_state):
        # self.q(hidden_state) is doing X * W_Q, projecting the input into Q
        # this projection is basically what takes input of 512 dim into 1/n_heads dim of that.
        # Each head learns to project differently
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embedding_dimensions
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        # hidden_state is just the input embedding (after positional encoding)
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        # x here, is simply the output of each head, concatenated together
        x = self.output_linear(x)
        # combines all head ouptuts, to have a rich understanding, optimally mixing info
        
        # # Maybe it learns:
        # - "Combine 30% syntax + 50% semantics + 20% position"
        # - "For nouns, emphasize semantic info more"
        # - "For verbs, emphasize syntax patterns more"
        
        # referred as `W_O`, or output projection matrix.
        return x


if __name__ == "__main__":
    text = "Hi, this is a test"
    input_sequence = tokenize_input(text)
    embedding_output = embed(input_sequence)
    pos_enc_output = add_positional_encoding(embedding_output)

    multihead_attn = MultiHeadAttention(config).to(config.device)
    atn_output = multihead_attn(pos_enc_output)
    print("Shape of output:", atn_output.size())
    print("Number of heads:", len(MultiHeadAttention(config).heads))
    multihead_attn.heads[:2]
