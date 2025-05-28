import pandas as pd
import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../02-transformers-from-scratch")
    )
)

from _0_tokenization import tokenize_input, tokenizer  # type: ignore  # noqa: F401
from _1_config import Config  # type: ignore  # noqa: F401
from _9_transformer import Transformer  # type: ignore  # noqa: F401
from _10_dataloader import TextDataset  # type: ignore  # noqa: F401
from _11_inference import load_model, predict  # type: ignore  # noqa: F401


def get_model_config_and_data_loaders(
    train_path: str,
    test_path: str,
    max_tokens: int = 100,
    custom_tokenizer: AutoTokenizer | None = None,
    y_label: str = "label",
    embedding_dimensions: int = 128,
    num_attention_heads: int = 8,
    hidden_dropout_prob: float = 0.3,
    num_encoder_layers: int = 2,
    batch_size: int = 64,
    smaller_dataset: bool = False,
) -> tuple[Config, DataLoader, DataLoader]:
    #
    if smaller_dataset:
        train_df = pd.read_csv(train_path)[:1000]
        test_df = pd.read_csv(test_path)[:200]
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    config = {
        # instead of .vocab_size when using special tokens
        "vocab_size": len(custom_tokenizer or tokenizer),
        "embedding_dimensions": embedding_dimensions,
        "max_tokens": max_tokens,
        "num_attention_heads": num_attention_heads,
        "hidden_dropout_prob": hidden_dropout_prob,
        "intermediate_size": embedding_dimensions * 4,
        "num_encoder_layers": num_encoder_layers,
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
    }
    config = Config(config)

    def _tokenize(row, max_length):
        return tokenize_input(
            row["text"],
            max_length=max_length,
            return_tensors=None,
            custom_tokenizer=custom_tokenizer,
        )

    train_df["tokenized"] = train_df.apply(
        lambda row: _tokenize(row, config.max_tokens), axis=1
    )
    test_df["tokenized"] = test_df.apply(
        lambda row: _tokenize(row, config.max_tokens), axis=1
    )

    print("Train [rows, cols]:", train_df.shape)
    print("Val [rows, cols]:", test_df.shape)
    print()
    print(train_df.head())

    train_dataset = TextDataset(train_df["tokenized"], train_df[y_label])
    test_dataset = TextDataset(test_df["tokenized"], test_df[y_label])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return config, train_dataloader, val_dataloader


def train(
    config,
    model,
    loss_function,
    optimizer,
    train_dataloader,
    val_dataloader,
    n_epochs,
    name_model,
):
    best_val_accuracy = 0.0

    for epoch in range(n_epochs):
        for i, batch in enumerate(train_dataloader):
            inputs, targets = batch  # .to(config.device)
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()

            train_predictions = outputs.squeeze(-1).detach().cpu() > 0.5
            train_accuracy = (
                (train_predictions == targets.cpu()).type(torch.float).mean().item()
            )
            if i % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses, val_accuracies = [], []

                    for val_inputs, val_targets in val_dataloader:
                        val_inputs = val_inputs.to(config.device)
                        val_targets = val_targets.to(config.device)

                        val_outputs = model(val_inputs)
                        val_loss = loss_function(val_outputs.squeeze(-1), val_targets)
                        val_losses.append(val_loss.item())
                        # TODO: what this 2 lines do?
                        val_predictions = val_outputs.squeeze(-1).detach().cpu() > 0.5
                        val_accuracy = (
                            (val_predictions == val_targets.cpu())
                            .type(torch.float)
                            .mean()
                            .item()
                        )
                        val_accuracies.append(val_accuracy)

                val_loss = np.mean(val_losses)
                val_accuracy = np.mean(val_accuracies)
                print(
                    f"Epoch {epoch + 1}/{n_epochs} Step {i} \tTrain Loss: {loss.item():.2f} \tTrain Accuracy: {train_accuracy:.3f}\n\t\t\tVal Loss: {val_loss:.2f}   \tVal Accuracy: {val_accuracy:.3f}"
                )
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "val_accuracy": val_accuracy,
                            "epoch": epoch,
                            "step": i,
                        },
                        f"04-practice/{name_model}.pt",
                    )
                    print(
                        f"New best model saved with validation accuracy: {val_accuracy:.3f}"
                    )


def double_check_inference(
    config: Config,
    model_path: str,
    input_key: str,
    label_key: str,
    train_dataset_path: str,
):
    model = load_model(config, model_path)
    df = pd.read_csv(train_dataset_path)
    sample_df = df.sample(n=10, random_state=42)
    for _, row in sample_df.iterrows():
        text = row[input_key]
        prediction = predict(text, config, model)
        print(f"Text: {text}, Prediction: {prediction} | Expected: {row[label_key]}")
