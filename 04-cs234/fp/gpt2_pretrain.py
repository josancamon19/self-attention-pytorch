import json
import os
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from models.gpt2 import GPT2Model
from config import GPT2Config
from optimizer import AdamW

subdir = "data"
os.makedirs(subdir, exist_ok=True)


def download_datasets():
    for ds in [
        # "webtext",
        "small-117M",
        "small-117M-k40",
        # "medium-345M",
        # "medium-345M-k40",
        # "large-762M",
        # "large-762M-k40",
        # "xl-1542M",
        # "xl-1542M-k40",
    ]:
        for split in ["train", "valid", "test"]:
            filename = ds + "." + split + ".jsonl"
            r = requests.get(
                "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
                + filename,
                stream=True,
            )

            with open(os.path.join(subdir, filename), "wb") as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
                ) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


def explore_dataset():
    with open("data/small-117M.test.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        print(f"Loaded {len(data)} examples")
        print("First example:", data[0])

    longest = max([i["length"] for i in data])
    print("longest sequence", longest)


class PreTrainDataset(Dataset):
    def __init__(self, file_path: str = "data/small-117M.train.jsonl"):
        with open(file_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", cache_dir="./.cache/huggingface"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # why??

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch, max_length=1024):
        texts = [item["text"] for item in batch]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_masks": attention_mask,
        }


def load_dataset(batch_size: int = 8):
    train_dataset = PreTrainDataset("data/small-117M.train.jsonl")
    valid_dataset = PreTrainDataset("data/small-117M.valid.jsonl")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
    )
    # for item in valid_dataloader:
    #     print(item)
    #     break
    return train_dataloader, valid_dataloader


def get_model(device):
    config = GPT2Config()
    model = GPT2Model(config)
    model.to(device)
    optimizer = AdamW(model.parameters())
    return model, optimizer


def train(model, optimizer, device, train_dataloader):
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"train-{epoch}"):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:]
            print("input,labels: ", input_ids.shape, labels.shape)
            labels = labels.contiguous().flatten()
            print("labels flattened:", labels.shape)
            attention_masks = batch["attention_masks"].to(device)
            pred = model(input_ids, attention_masks)
            pred = model.hidden_state_to_token(pred["last_hidden_state"])
            pred = pred[:, :-1, :]
            pred = pred.reshape(-1, pred.shape[2])
            print(pred.shape)
            # flattening is needed at both sides?
            loss = F.cross_entropy(pred, labels, reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("batch_loss", loss)
            train_loss += loss.item()

        # TODO: validation loss
        # TODO: saving the model

        print(f"epoch {epoch} train_loss: {train_loss / len(train_dataloader)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, valid_dataloader = load_dataset()
    model, optimizer = get_model(device)
    train(model, optimizer, device, train_dataloader)
