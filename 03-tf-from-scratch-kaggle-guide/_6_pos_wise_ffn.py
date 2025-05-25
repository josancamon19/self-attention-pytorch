import torch.nn as nn


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
