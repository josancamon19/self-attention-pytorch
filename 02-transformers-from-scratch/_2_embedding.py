import torch
import torch.nn as nn
from transformers import AutoTokenizer

from _0_tokenization import tokenize_input
from _1_config import config

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create an embedding layer, with ~32,000 possible embeddings, each having 128 dimensions
        # nn.Embedding is initialized with random weights, so we need to train it.
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_dimensions
        )

    def forward(self, tokenized_sentence):
        return self.token_embedding(tokenized_sentence)


def embed(input_sequence: torch.Tensor, add_logs: bool = False):
    token_embedding = TokenEmbedding(config).to(config.device)
    embedding_output = token_embedding(input_sequence)
    if add_logs:
        print("embed:Shape of output:", embedding_output.size())
    return embedding_output


if __name__ == "__main__":
    text = "Hi, this is a test"
    input_sequence = tokenize_input(text)
    embed(input_sequence, True)
