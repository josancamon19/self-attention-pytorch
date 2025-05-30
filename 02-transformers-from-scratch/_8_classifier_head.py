# Our classifier head first flattens the output of the encoder block, then passes it through two dense layers with ReLU activation in the middle. Since we're going to use this transformer for binary classification, we then pass our final output through a sigmoid activation function.

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            config.embedding_dimensions,  # config.max_tokens *
            2 * config.embedding_dimensions,
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * config.embedding_dimensions, 1)

    def forward(self, x):
        # x = self.flatten(x)
        x = x.mean(dim=1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return torch.sigmoid(x)  # Sigmoid activation for binary classification

    def get_dimensions(self):
        return [
            self.linear1.weight.size(),
            self.linear2.weight.size(),
        ]

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())
