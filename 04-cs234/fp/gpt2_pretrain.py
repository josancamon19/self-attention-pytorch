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
from torch import autocast  # , nn_utils
import wandb

os.makedirs(".models/gpt2/", exist_ok=True)
os.makedirs("data/", exist_ok=True)


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

            with open(os.path.join("data/", filename), "wb") as f:
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
    print(
        "load_dataset train_dataset, valid_dataset",
        len(train_dataset),
        len(valid_dataset),
    )

    train_dataloader = DataLoader(
        train_dataset[:10000],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,  # [:1500],
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
    model.to(device, dtype=torch.bfloat16)
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    return model, optimizer


def comp_val_loss(model, device, valid_dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="valid-loss"):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = input_ids[:, 1:].contiguous().flatten()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                pred = model.hidden_state_to_token(
                    model(input_ids, attention_masks)["last_hidden_state"]
                )
                pred = pred[:, :-1, :].reshape(-1, pred.shape[2])
                pred = torch.clamp(pred, min=-100, max=100)
                loss = F.cross_entropy(pred, labels, reduction="mean")
                total_loss += loss.item()

    return total_loss / len(valid_dataloader)


def train(model, optimizer, device, train_dataloader, valid_dataloader):
    epochs = 10
    best_valid_loss = float("inf")
    run = wandb.init(
        entity="josancamon19-cifrato",
        project="pretrain-gpt2",
    )
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        step = 1
        for batch in tqdm(train_dataloader, desc=f"train-{epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = input_ids[:, 1:].contiguous().flatten()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                pred = model.hidden_state_to_token(
                    model(input_ids, attention_masks)["last_hidden_state"]
                )
                pred = pred[:, :-1, :].reshape(-1, pred.shape[2])
                loss = F.cross_entropy(pred, labels, reduction="mean")

                if torch.isnan(loss):  # issues with nan
                    print(f"NaN loss detected at step {step}")
                    print(f"Max pred: {pred.max()}, Min pred: {pred.min()}")
                    print(
                        f"Max input_ids: {input_ids.max()}, Min input_ids: {input_ids.min()}"
                    )
                    break

            loss.backward()
            # nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            step += 1
            optimizer.zero_grad()
            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        print(f"epoch {epoch} train_loss: {train_loss}")
        valid_loss = comp_val_loss(model, device, valid_dataloader)
        print(f"validation_loss: {valid_loss}")
        run.log({"train_loss": train_loss, "val_loss": valid_loss})
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            state = {
                "config": model.config,
                "model": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(state, ".models/gpt2/model.pt")


def inference(device, prompt: str, completion_tokens: int):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./.cache/huggingface")
    tokenizer.pad_token = tokenizer.eos_token

    saved = torch.load(".models/gpt2/model.pt", weights_only=False)
    model = GPT2Model(GPT2Config())
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()
    with torch.inference_mode():
        encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids, attention_mask = (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )
        for _ in range(completion_tokens):
            output = model(input_ids, attention_mask)
            next_token_pred = model.hidden_state_to_token(output["last_token"])
            next_token = torch.argmax(next_token_pred, dim=-1)
            # print("argmax:", next_token, "decoded:", tokenizer.decode(next_token.item()))
            next_token = next_token.unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token).to(device)], dim=1
            )

    print(tokenizer.decode(input_ids[0]))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, valid_dataloader = load_dataset(batch_size=14)
    model, optimizer = get_model(device)
    valid_loss = comp_val_loss(model, device, valid_dataloader)
    print(f"validation_loss: {valid_loss}")
    # train(model, optimizer, device, train_dataloader, valid_dataloader)
    # inference(device, "Hi, my name is", 50)
