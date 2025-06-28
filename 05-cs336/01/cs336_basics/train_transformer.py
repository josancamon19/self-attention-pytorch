from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from cs336_basics.transformer import Transformer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


class PretrainDataset(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, dataset_path: str, max_sequence_length: int):
        self.samples = []
        with open(dataset_path, "rb") as f:
            self.samples = f.read().decode("utf-8", errors="ignore").split("<|endoftext|>")
            # TODO: missing a lot of tokens when truncating, many > max sequence length

        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch: list[str]):
        tokenized = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


def train():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_sequence_length = 1024
    batch_size = 16
    embedding_dim = 128

    dataset = PretrainDataset(tokenizer, "data/owt_valid.txt", max_sequence_length)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    model = Transformer(tokenizer.vocab_size, max_sequence_length, embedding_dim, 2, 8)
    model.train()

    epochs = 10
    loss_fn = CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    for _ in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"][:, 1:]
            labels = batch["input_ids"][:, :-1]  # better way to slice
            # print("input_ids.shape, labels.shape:", input_ids.shape, labels.shape)
            attention_mask = batch["attention_mask"][:, 1:]
            output = model(input_ids, attention_mask)
            output_flatten = output.view(-1, output.shape[-1])
            labels = labels.contiguous().view(-1)
            # print("output, output_flatten, labels:", output.shape, output_flatten.shape, labels.shape)
            loss = loss_fn(output_flatten, labels)
            loss.backward()
            optim.step()
            model.zero_grad()
            break


train()
