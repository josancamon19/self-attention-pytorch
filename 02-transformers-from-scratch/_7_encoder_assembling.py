# assembling the encoder
# - multihead attention
# - residual connection (add + norm)
# - feed forward pos wise

import torch.nn as nn
from _0_tokenization import tokenize_input
from _1_config import config
from _2_embedding import embed
from _3_positional_encoding import add_positional_encoding
from _4_attention import MultiHeadAttention
from _6_pos_wise_ffn import FeedForward


class Encoder(nn.Module):
    "Attention is all you need uses Post Layer Normalization, most literature contains pre-LN"

    # Post-LN is harder to train. [ ] Expand on this why?

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimensions)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimensions)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # First perform layer normalization
        hidden_state = self.layer_norm1(x)
        # Then apply attention + skip connection
        x = x + self.attention(hidden_state)

        # Apply layer normalization before inputting to the FFN
        hidden_state = self.layer_norm2(x)
        # Apply FNN + skip connection
        x = x + self.feed_forward(hidden_state)
        return x

    def get_dimensions(self):
        return {
            "attention": self.attention.get_dimensions(),
            "ffn": self.feed_forward.get_dimensions(),
        }

    def get_params_count(self):
        return {
            "attention": self.attention.get_params_count(),
            "ffn": self.feed_forward.get_params_count(),
        }


if __name__ == "__main__":
    text = "Hi, this is a test"
    input_sequence = tokenize_input(text)
    embedding_output = embed(input_sequence)
    pos_enc_output = add_positional_encoding(embedding_output)
    encoder = Encoder(config).to(config.device)
    encoder_output = encoder(pos_enc_output)
    print("Shape of output:", encoder_output.size())
