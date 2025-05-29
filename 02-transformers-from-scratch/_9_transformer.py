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

    def get_params_count(self):
        embedding_params = self.embedding.get_params_count()
        encoder_params = [encoder.get_params_count() for encoder in self.encoders]
        encoder_params_attention, encoder_params_ffn = (
            encoder_params[0]["attention"],
            encoder_params[0]["ffn"],
        )
        classifier_params = self.classifier_head.get_params_count()
        return {
            "embedding": embedding_params,
            "encoder": {
                "layers": len(self.encoders),
                "1_attention": encoder_params_attention,
                "1_ffn": encoder_params_ffn,
            },
            "classifier_head": classifier_params,
            "total": embedding_params
            + encoder_params_attention * len(self.encoders)
            + encoder_params_ffn * len(self.encoders)
            + classifier_params,
        }


if __name__ == "__main__":
    model = Transformer(config)
    # print(json.dumps(model.get_dimensions(), indent=2))
    print(json.dumps(model.get_params_count(), indent=2))
