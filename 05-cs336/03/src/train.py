# issues with Stanford connection and stuff, running the trains on my own
from model import BasicsTransformerLM
from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from tokenizer import train_tokenizer
import numpy as np
import wandb
import json
import ray
from ray import tune
from ray.tune import CLIReporter
import random
import os

vocab_size, seq_length = 32000, 512
torch.manual_seed(42)


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


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Extract config values
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    tokens = config["D"]
    param_count = config["N"]
    flops = config["C"]

    model = BasicsTransformerLM(
        vocab_size, seq_length, d_model, num_layers, num_heads, d_model * 4, 0.1, 0.1
    )
    model = model.to(device)
    model = torch.compile(model)
    model.train()

    dataset_path = config.get("dataset", "data/slimpajama_sample_100M.npy")
    train_data = np.load(dataset_path, mmap_mode="r")
    steps = int(tokens / seq_length / batch_size)
    rand_i = random.randint(0, 10000)
    run_id = f"N{(param_count / 1e6):.1f}_D{(tokens / 1e6):.1f}_C{flops}_d{d_model}_l{num_layers}_h{num_heads}_b{batch_size}_lr{lr}_r_ignore{rand_i}"
    run = wandb.init(id=run_id, project="assignment-03-scaling-laws", reinit=True)

    optimizer = AdamW(model.parameters(), lr, weight_decay=0.01)  # betas=[0.9, 0.95]
    lr_scheduler = CosineAnnealingLR(optimizer, steps, eta_min=lr / 10)

    train_loss = 0
    for i in range(steps):
        batch = data_loading(train_data, batch_size, seq_length, device)
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            input_ids, labels = batch
            output = model(input_ids)
            output_flatten = output.view(-1, output.shape[-1])
            labels = labels.contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(output_flatten, labels)

        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        if i % 50 == 0:
            avg_loss = train_loss / (i + 1)
            run.log({"train_loss": avg_loss}, step=i)
            tune.report({"train_loss": avg_loss})

    final_loss = train_loss / steps
    print(f"Training completed. Final loss: {final_loss:.6f}")

    tune.report({"train_loss": avg_loss, "final_loss": final_loss})

    try:
        run.log({"train_loss": final_loss, "final_loss": final_loss}, step=steps)
        wandb.finish()
    except Exception:
        pass  # wandb Broken Pipeline

    return final_loss


# ======= Data + Tokenizer =======

target_tokens = 1000_000_000
# dataset_path = "data/slimpajama_sample_100M.txt"
dataset_path = "data/slimpajama_sample_1B.txt"


def retrieve_dataset():
    # get dataset
    dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

    total_tokens = 0
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


# ======= Train distributed =======


def run_distributed_training(config_path=".configs/config_1e+13.json"):
    with open(config_path, "r") as f:
        configs = json.load(f)["configs"]
        print(f"Loaded {len(configs)} configurations")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"excludes": ["data/slimpajama_sample_100M.npy"]},
    )
    for config in configs:
        config["dataset"] = (
            "/workspace/transformers/05-cs336/03/data/slimpajama_sample_100M.npy"
        )

    reporter = CLIReporter(
        metric_columns=["final_loss", "train_loss", "lr"], max_report_frequency=10
    )
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected {num_gpus} GPUs")

    analysis = tune.run(
        train,
        config={"grid_search": configs},
        resources_per_trial={"gpu": 1} if num_gpus > 0 else {},
        num_samples=1,
        progress_reporter=reporter,
        verbose=1,
    )

    # Collect and display results
    results = []
    for trial in analysis.trials:
        result = {
            "config": trial.config,
            "final_loss": trial.last_result.get("final_loss", float("inf")),
        }
        results.append(result)

    print("\n=== Training Results ===")
    for i, result in enumerate(results):
        print(f"Trial {i}: Loss = {result['final_loss']:.6f}")
        print(f"  Config: {result['config']}")

    ray.shutdown()
    return results


if __name__ == "__main__":
    retrieve_dataset()
    train_tokenizer(dataset_path, 32000, ["<|endoftext|>"], True)
    # validate_dataset_size()
    # run_distributed_training(".configs/config_1e+13.json")
    # run_distributed_training(".configs/config_5e+13.json")
    # run_distributed_training(".configs/config_1e+14.json")
    # run_distributed_training(".configs/config_5e+14.json")
    # run_distributed_training(".configs/config_1e+15.json")
    # after exhaustive search expanded ratio > 40, up to 100 + set lr 1e-3 and batch_size 128 fixed.
    # need a new tokenizer, 1B tokens to run this experiments up to 1e17 FLOPs
    # run_distributed_training(".configs/config_5e+15.json")
    # run_distributed_training(".configs/config_1e+16.json")
    pass
