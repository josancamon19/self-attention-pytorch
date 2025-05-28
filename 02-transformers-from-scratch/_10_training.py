from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch

from _0_tokenization import tokenize_input
from _1_config import config
from _9_transformer import Transformer
from _10_dataloader import TextDataset


train_path, test_path = (
    "02-transformers-from-scratch/data/train.csv",
    "02-transformers-from-scratch/data/test.csv",
)
df = pd.read_csv(train_path, index_col=0, usecols=["id", "text", "target"])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Sample tweet:", train_df.iloc[42].text, "\n")


def _tokenize(row, max_length):
    return tokenize_input(row["text"], max_length=max_length, return_tensors=None)


max_length = config.max_tokens
train_df["tokenized"] = train_df.apply(lambda row: _tokenize(row, max_length), axis=1)
val_df["tokenized"] = val_df.apply(lambda row: _tokenize(row, max_length), axis=1)

print(train_df.head())

# Prepare the data for the model

X_train, y_train = train_df["tokenized"], train_df["target"]
X_val, y_val = val_df["tokenized"], val_df["target"]

# Create both datasets
train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)

# Create the DataLoaders
train_dataloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)  # Shuffle for random sampling without replacement
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


def train():
    # Instantiate a transformer model
    model = Transformer(config).to(config.device)

    # Define loss function and optimizer
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Move model to GPU if available
    model = model.to(config.device)

    # Number of training epochs
    n_epochs = 8

    # Metrics dictionary for plotting later
    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(n_epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            # Move inputs and targets to device
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            # Set model to training mode
            model.train()

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            # print(inputs.shape, targets.shape)
            outputs = model(inputs)

            # Compute loss
            train_loss = loss_function(outputs.squeeze(), targets)

            # Backward pass and optimize
            train_loss.backward()
            optimizer.step()

            # Calculate accuracy
            train_predictions = outputs.squeeze().detach().cpu() > 0.5
            train_accuracy = (
                (train_predictions == targets.cpu()).type(torch.float).mean().item()
            )

            # Validation loop
            if i % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    val_accuracies = []

                    for val_inputs, val_targets in val_dataloader:
                        val_inputs = val_inputs.to(config.device)
                        val_targets = val_targets.to(config.device)

                        val_outputs = model(val_inputs)
                        val_loss = loss_function(val_outputs.squeeze(), val_targets)
                        val_losses.append(val_loss.item())

                        val_predictions = val_outputs.squeeze().detach().cpu() > 0.5
                        val_accuracy = (
                            (val_predictions == val_targets.cpu())
                            .type(torch.float)
                            .mean()
                            .item()
                        )
                        val_accuracies.append(val_accuracy)

                # Get the mean loss and accuracy over the validation set
                val_loss = np.mean(val_losses)
                val_accuracy = np.mean(val_accuracies)

                # Print metrics here during training
                print(
                    f"Epoch {epoch + 1}/{n_epochs} Step {i} \tTrain Loss: {train_loss.item():.2f} \tTrain Accuracy: {train_accuracy:.3f}\n\t\t\tVal Loss: {val_loss:.2f}   \tVal Accuracy: {val_accuracy:.3f}"
                )

                # Store metrics
                metrics["train_loss"].append(train_loss.item())
                metrics["train_accuracy"].append(train_accuracy)
                metrics["val_loss"].append(val_loss)
                metrics["val_accuracy"].append(val_accuracy)

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
                        "02-transformers-from-scratch/best_model.pt",
                    )
                    print(
                        f"New best model saved with validation accuracy: {val_accuracy:.3f}"
                    )


if __name__ == "__main__":
    train()
