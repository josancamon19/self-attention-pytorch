import torch
from _0_tokenization import tokenizer


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


# Setting parameters for our model
config = {  # We get the vocabulary size used by our tokenizer
    "vocab_size": tokenizer.vocab_size,
    # arbitrary choice, usually works for small tasks
    "embedding_dimensions": 128,
    # max 100 tokens, hard cut, any longer is not learned or used during inference.
    "max_tokens": 100,
    # Number of attention heads to be used, 4/8/16/32
    "num_attention_heads": 8,
    # Dropout on feed-forward network, 0.1/0.2/0.3
    "hidden_dropout_prob": 0.3,
    # Number of neurons in the intermediate hidden layer (quadruple the number of emb dims)
    "intermediate_size": 128 * 4,
    # How many encoder blocks to use in our architecture
    "num_encoder_layers": 2,
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
}
config = Config(config)
