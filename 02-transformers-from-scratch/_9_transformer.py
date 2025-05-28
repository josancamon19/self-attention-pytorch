import torch.nn as nn
from _2_embedding import TokenEmbedding
from _3_positional_encoding import PositionalEncoding
from _7_encoder_assembling import Encoder
from _8_classifier_head import ClassifierHead


class Transformer(nn.Module):
    "In attention is all you need, 6 encoder blocks were used, example has 2"
    def __init__(self, config):
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.encoders = nn.ModuleList(
            [Encoder(config) for _ in range(config.num_encoder_layers)]
        )
        self.classifier_head = ClassifierHead(config)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for encoder in self.encoders:
            x = encoder(x)
        return self.classifier_head(x)
