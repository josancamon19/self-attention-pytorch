import torch
import torch.nn as nn

from _0_tokenization import tokenize_input
from _1_config import config
from _2_embedding import embed


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        pe = torch.zeros(config.max_tokens, config.embedding_dimensions)
        position = torch.arange(0, config.max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (
            10000
            ** (
                torch.arange(0, config.embedding_dimensions, 2).float()
                / config.embedding_dimensions
            )
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = self.pe.to(config.device)

    def forward(self, x):
        return x + self.pe[:, 0]
    
    def get_dimensions(self):
        return self.pe.size()


def add_positional_encoding(embedding_output: torch.Tensor, visualize: bool = False):
    # Confirm this module is working as intended
    positional_encoding = PositionalEncoding(config).to(config.device)
    pos_enc_output = positional_encoding(embedding_output)
    if visualize:
        print("Shape of output:", pos_enc_output.size())
        # View the difference between the two layers
        # These differences can be checked with our heatmap visualization
        diff = pos_enc_output - embedding_output
        print("\nTensor at position 0 first 20 values:")
        print(diff[0, 0][:20])
        print("\nTensor at position 0 last 20 values:")
        print(diff[0, 0][-20:])

        print("\nTensor at position 50 first 20 values:")
        print(diff[0, 50][:20])
        print("\nTensor at position 50 last 20 values:")
        print(diff[0, 50][-20:])
    return pos_enc_output


if __name__ == "__main__":
    text = "Hi, this is a test"
    input_sequence = tokenize_input(text)
    embedding_output = embed(input_sequence)
    add_positional_encoding(embedding_output, visualize=True)
