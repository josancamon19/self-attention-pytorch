import torch
import torch.nn as nn
from transformers import AutoTokenizer

from _1_tokenization import tokenize_input

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


# Setting parameters for our model
config = {  # We get the vocabulary size used by our tokenizer
    "vocab_size": tokenizer.vocab_size,
    "embedding_dimensions": 128,
    # We're only going to use a maximum of 100 tokens per input sequence
    "max_tokens": 100,
    # Number of attention heads to be used
    "num_attention_heads": 8,
    # Dropout on feed-forward network
    "hidden_dropout_prob": 0.3,
    # Number of neurons in the intermediate hidden layer (quadruple the number of emb dims)
    "intermediate_size": 128 * 4,
    # How many encoder blocks to use in our architecture
    "num_encoder_layers": 2,
    "device": "cpu",
}
config = Config(config)


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
