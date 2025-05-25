import torch.nn as nn
from _1_tokenization import tokenize_input
from _2_embedding import embed, config
from _3_positional_encoding import add_positional_encoding
from _4_attention import MultiHeadAttention


# First, we get our positional encoder output
# x = pos_enc_output

# Instantiate the layer normalization
# layer_norm = nn.LayerNorm(config.embedding_dimensions).to(config.device)

# Our output is then defined as the normalized output of the attention block
# plus the ouput of the positional encoder (skip connection)
# add_norm_output = layer_norm(x + multihead_attn(x))


def residual(x, attn_output):
    # x = pos_enc_output
    layer_norm = nn.LayerNorm(config.embedding_dimensions).to(config.device)
    return layer_norm(x + attn_output)  # add + norm


if __name__ == "__main__":
    text = "Hi, this is a test"
    input_sequence = tokenize_input(text)
    embedding_output = embed(input_sequence)
    pos_enc_output = add_positional_encoding(embedding_output)

    multihead_attn = MultiHeadAttention(config).to(config.device)
    attn_output = multihead_attn(pos_enc_output)
    residual(pos_enc_output, attn_output)
