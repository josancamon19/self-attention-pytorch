# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv
import os
import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


def get_balanced_dataset(dataset):
    true = sum([item[2] for item in dataset])
    false = len(dataset) - true
    print(f"ParaphraseDetectionDataset.__init__ len(true): {true} len(false): {false}")
    balanced = min(true, false)
    balanced_dataset = []
    count_true = count_false = 0
    for item in dataset:
        if item[2] == 1 and count_true == balanced:
            continue
        elif item[2] == 0 and count_false == balanced:
            continue
        balanced_dataset.append(item)

        if item[2] == 1:
            count_true += 1
        else:
            count_false += 1
    print(
        f"ParaphraseDetectionDataset.__init__ llen(dataset): {len(dataset)} len(balanced_dataset): {len(balanced_dataset)}"
    )
    return balanced_dataset


class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args):
        # dataset = get_balanced_dataset(dataset)

        self.dataset = dataset
        self.p = args

        # Use cached tokenizer path if available
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        labels = torch.LongTensor([x[2] for x in all_data])
        # labels = ["yes" if label == 1 else "no" for label in [x[2] for x in all_data]]
        # labels = self.tokenizer(
        #     labels, return_tensors="pt", padding=True, truncation=True
        # )["input_ids"]
        sent_ids = [x[3] for x in all_data]

        cloze_style_sents = [
            f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
            for (s1, s2) in zip(sent1, sent2)
        ]
        encoding = self.tokenizer(
            cloze_style_sents, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data


class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args

        # Use cached tokenizer path if available
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        cloze_style_sents = [
            f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
            for (s1, s2) in zip(sent1, sent2)
        ]

        encoding = self.tokenizer(
            cloze_style_sents, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sent_ids": sent_ids,
        }

        return batched_data


def load_paraphrase_data(paraphrase_filename, split="train"):
    paraphrase_data = []
    if split == "test":
        with open(paraphrase_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                paraphrase_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(paraphrase_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    paraphrase_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except:
                    pass

    # print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data


class SonnetsDataset(Dataset):
    def __init__(self, file_path, args=None):
        # Try to use local tokenizer first
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = self._load_sonnets(file_path)
        # print("SonnetsDataset, sonnet[0]:\n", self.sonnets[0], "\n")

    def _load_sonnets(self, file_path):
        """Reads the file and extracts individual sonnets."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
        sonnets = re.split(r"\n\s*\d+\s*\n", text)[1:]  # Remove header text

        # Strip leading/trailing spaces
        return [s.strip() for s in sonnets]

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        return (idx, self.sonnets[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        sonnets = [example[1] for example in all_data]

        encoding = self.tokenizer(
            sonnets, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sent_ids": idx,
        }

        return batched_data
