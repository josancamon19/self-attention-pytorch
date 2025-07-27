# issues with Stanford connection and stuff, running the trains on my own
from model import BasicsTransformerLM
from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss as cross_entropy_loss
from datasets import load_dataset
from tokenizer import train_tokenizer
import numpy as np
import wandb

vocab_size, seq_length = 32000, 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Consider the model.py architecture
# - abs pos instead of RoPE
# - LNorm instead of RMS (pre)
# - GeLU instead of SwiGLU, 2 Linear instead of 3, dff = 4d_model as usual
# - untied input/output embeddings
# - SlimPajama dataset
# - BPE 32k items on above dataset
# - seq_length 512
# - attn and residual dropout 0.1 ?
# - AdamW with WD 0.01 and gradient clipping 1.0
# - cos lr schedule to decay lr 10x, annealing steps = num of training steps,
# - no lr warmup used.


def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    indices = starting_indices.unsqueeze(1) + torch.arange(context_length + 1)
    batch = torch.from_numpy(x[indices]).to(device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def train(
    d_model,
    num_layers,
    num_heads,
    batch_size,
    lr,
    tokens,  # D
    # only for logging
    param_count,  # N
    flops,
):
    model = BasicsTransformerLM(
        vocab_size,
        seq_length,
        d_model,
        num_layers,
        num_heads,
        d_model * 4,
        0.1,
        0.1,
    ).to(device)
    model.train()

    train_data = np.load("", mmap_mode="r")
    steps = int(tokens / seq_length / batch_size)

    run_id = f"N{(param_count / 1e6):.1f}_D{(tokens / 1e6):.1f}_C{flops:.1e}_d{d_model}_l{num_layers}_h{num_heads}_b{batch_size}_lr{lr}"
    run = wandb.init(id=run_id, project="assignment-03-scaling-laws")

    optimizer = AdamW(model.parameters(), lr, weight_decay=0.01)
    lr_scheduler = CosineAnnealingLR(optimizer, steps, eta_min=lr / 10)
    train_loss = 0

    for i in range(steps):
        batch = data_loading(train_data, batch_size, seq_length, device)
        optimizer.zero_grad()

        output = model(batch)
        # [ ]preview doesn't says if it uses fp32 or bf16?
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            input_ids, labels = batch
            output = model(input_ids, None)
            output_flatten = output.view(-1, output.shape[-1])
            labels = labels.contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(output_flatten, labels)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        if i % 50 == 0:
            run.log({"train_loss": train_loss / i, "lr": lr}, step=i)


# ======= Data + Tokenizer =======

dataset_path = "data/slimpajama_sample_100M.txt"


def retrieve_dataset():
    # get dataset
    dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

    total_tokens = 0
    target_tokens = 100_000_000
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            # Each item is a dict with keys: 'text', 'meta'
            text = item.get("text", "")
            # Estimate tokens by whitespace split (rough, but ok for sampling)
            num_tokens = len(text.split())
            # Write only text, append <|endoftext|> divider
            f.write(text.strip() + "\n<|endoftext|>\n")
            total_tokens += num_tokens
            if total_tokens >= target_tokens:
                print(f"Saved {total_tokens} tokens to {dataset_path}")
                break


def validate_dataset_size():
    word_count = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "<|endoftext|>":
                continue
            words = line.split()
            word_count += len(words)
    print(f"Total words: {word_count}")


def _train_tokenizer():
    train_tokenizer(
        input_text_file=dataset_path,
        target_vocab_size=32000,
        save_results=True,
    )


if __name__ == "__main__":
    # retrieve_dataset()
    _train_tokenizer()
    # validate_dataset_size()
