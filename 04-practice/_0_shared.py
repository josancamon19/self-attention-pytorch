import torch
import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../03-tf-from-scratch-kaggle-guide")
    )
)

from _0_tokenization import tokenize_input, tokenizer  # type: ignore  # noqa: F401
from _1_config import Config  # type: ignore  # noqa: F401
from _9_transformer import Transformer  # type: ignore  # noqa: F401
from _10_dataloader import TextDataset  # type: ignore  # noqa: F401
from _10_inference import load_model, predict  # type: ignore  # noqa: F401


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
            loss = loss_function(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            # Exception has occurred: ValueError
            # Using a target size (torch.Size([1])) that is different to the input size (torch.Size([])) is deprecated. Please ensure they have the same size.
            #   File "/Users/joancabezas/Downloads/projects/ai-research/transformers/03-tf-from-scratch-kaggle-guide/_12_separate_exercise.py", line 69, in train
            #     loss = loss_function(outputs.squeeze(), targets)
            #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #   File "/Users/joancabezas/Downloads/projects/ai-research/transformers/03-tf-from-scratch-kaggle-guide/_12_separate_exercise.py", line 124, in <module>
            #     train()
            # ValueError: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([])) is deprecated. Please ensure they have the same size.

            train_predictions = outputs.squeeze().detach().cpu() > 0.5
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
                        val_loss = loss_function(val_outputs.squeeze(), val_targets)
                        val_losses.append(val_loss.item())
                        # TODO: what this 2 lines do?
                        val_predictions = val_outputs.squeeze().detach().cpu() > 0.5
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
