import torch.nn as nn
from _1_tokenization import tokenize_input
from _2_embedding import embed, config
from _3_positional_encoding import add_positional_encoding
from _4_attention import MultiHeadAttention
from _5_residual import residual


class FeedForward(nn.Module):
    # comparing to a normal ffn, we take the input, flatten it, and send it through the nn
    # in transformers, we take each token, and send it through the nn individually.
    # that's why is position wise, because we maintain the position of the tokens.
    # - great for parallelization, because each token is independent, and we can send them through the nn in parallel.
    # - maintains sequence structure
    # - the network learns patterns for any token in any position.
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embedding_dimensions, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.embedding_dimensions)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        # give the model a larger space to work with, finding more complex patterns
        x = self.linear_1(x)
        # non-linearity, finding more complex patterns
        x = self.gelu(x)
        # compress back to original space, compressing insights
        x = self.linear_2(x)
        # dropout, to prevent overfitting
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    text = "Hi, this is a test"
    input_sequence = tokenize_input(text)
    embedding_output = embed(input_sequence)
    pos_enc_output = add_positional_encoding(embedding_output)

    multihead_attn = MultiHeadAttention(config).to(config.device)
    attn_output = multihead_attn(pos_enc_output)
    residual_output = residual(pos_enc_output, attn_output)
    feed_forward = FeedForward(config).to(config.device)
    ffn_output = feed_forward(residual_output)
    print("Shape of output:", ffn_output.size())
