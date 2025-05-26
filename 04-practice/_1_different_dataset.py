import pandas as pd
import torch
from _0_shared import train
import sys
import os
from torch.utils.data import DataLoader

# Add the parent directory and the 03-tf-from-scratch-kaggle-guide directory to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../03-tf-from-scratch-kaggle-guide")
    )
)

from _0_tokenization import tokenize_input, tokenizer  # type: ignore
from _1_config import Config  # type: ignore
from _9_transformer import Transformer  # type: ignore
from _10_dataloader import TextDataset  # type: ignore


train_path = "04-practice/_1_data/train.csv"
test_path = "04-practice/_1_data/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


config = {
    "vocab_size": tokenizer.vocab_size,
    "embedding_dimensions": 128,
    "max_tokens": 100,  # should be okay as well, based on dataset, could even be less.
    "num_attention_heads": 8,
    "hidden_dropout_prob": 0.3,
    "intermediate_size": 128 * 4,
    "num_encoder_layers": 2,
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
}
config = Config(config)


def _tokenize(row, max_length):
    return tokenize_input(row["text"], max_length=max_length, return_tensors=None)


train_df["tokenized"] = train_df.apply(
    lambda row: _tokenize(row, config.max_tokens), axis=1
)
test_df["tokenized"] = test_df.apply(
    lambda row: _tokenize(row, config.max_tokens), axis=1
)

print("Train shape:", train_df.shape)
print(train_df.head())


train_dataset = TextDataset(train_df["tokenized"], train_df["Y"])
test_dataset = TextDataset(test_df["tokenized"], test_df["Y"])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


if __name__ == "__main__":
    model = Transformer(config).to(config.device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(
        config,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        val_dataloader,
        n_epochs=8,
        name_model="best_model_1",
    )
