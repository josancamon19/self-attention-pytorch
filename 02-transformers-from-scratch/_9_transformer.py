import json
import torch.nn as nn
from _2_embedding import TokenEmbedding
from _3_positional_encoding import PositionalEncoding
from _7_encoder_assembling import Encoder
from _8_classifier_head import ClassifierHead
from _1_config import config


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

    def get_dimensions(self):
        return {
            "embedding": self.embedding.get_dimensions(),
            "positional_encoding": self.positional_encoding.get_dimensions(),
            "encoders": [encoder.get_dimensions() for encoder in self.encoders],
            "classifier_head": self.classifier_head.get_dimensions(),
        }


if __name__ == "__main__":
    model = Transformer(config)
    print(json.dumps(model.get_dimensions(), indent=2))
    # Quick note, the 3rd dim in many of the dimensions is batch_size
    # hidden_state (output of positional encoding) has shape (batch_size, sequence_length, embed_dim). Let's assume batch_size=1 and sequence_length is variable. For the example "Hi, this is a test", after tokenization and embedding, let's say sequence_length=6. So hidden_state is (1, 6, 128).
    # TODO: rules of dimensions matching when nxn, if n > 2
