import argparse
import math
from collections.abc import Iterable
from collections.abc import Callable
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from src.transformer import Transformer, softmax
from src.tokenizer import Tokenizer
from torch.optim import AdamW
import torch.nn.functional as F
import os
import wandb
import numpy as np

os.makedirs("./.models", exist_ok=True)


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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    iteration: int | None = None,  # adapter
):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(data, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    saved = torch.load(path)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if "iteration" in saved:
        return saved["iteration"]


def cos_lr_schedule(lr_min, lr_max, warmup_steps, annealing_steps, step):
    # TODO: deeper understanding
    if step < warmup_steps:
        return (step / warmup_steps) * lr_max
    elif warmup_steps <= step <= annealing_steps:
        cos = math.cos(math.pi * (step - warmup_steps) / (annealing_steps - warmup_steps))
        return lr_min + ((1 + cos) / 2) * (lr_max - lr_min)
    else:  # step > annealing_steps
        return lr_min


# TODO: not optimal when compared to original ones, not at all, why?
def clip_gradients(params: Iterable, max_norm: float):
    grads = [p.grad for p in params if p.grad is not None]
    norms = [torch.linalg.vector_norm(g) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    if total_norm < max_norm:
        return
    scale_factor = max_norm / (total_norm + 1e-6)
    [grad.data.mul_(scale_factor) for grad in grads]


def cross_entropy_loss(result: torch.Tensor, labels: torch.Tensor):
    # TODO: understand deeper, how log/exponentials cancel, how that fixes numerical instability
    # -- cases like 0.00000 in logs include an infinite
    max_vals = torch.max(result, dim=1, keepdim=True)[0]
    shifted_logits = result - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=1, keepdim=True))
    log_probs = shifted_logits - log_sum_exp

    indices = torch.arange(result.shape[0])
    correct_log_probs = log_probs[indices, labels]
    # result = softmax(result, dim=1)
    return -torch.mean(correct_log_probs)


class SGD(torch.optim.Optimizer):
    # Problem (learning_rate_tuning):
    # 1e1, kept on 9.30
    # 1e2, .30, .28
    # TODO: something fun to test, train the whole model, and get losses curves for 2 epochs, on 5 different lr's
    def __init__(self, params, lr=1e-3):
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)  # step
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad  # update
                state["t"] = t + 1  # update step
        return loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["tinystories", "owt"], default="tinystories")

    args, _ = parser.parse_known_args()

    if args.dataset == "tinystories":
        default_train_dataset = ".tokenizer/TinyStoriesV2-GPT4-train-encoded.npy"
        default_valid_dataset = ".tokenizer/TinyStoriesV2-GPT4-valid-encoded.npy"
        default_tokenizer_vocab = ".tokenizer/TinyStoriesV2-GPT4-train-vocab.json"
        default_tokenizer_merges = ".tokenizer/TinyStoriesV2-GPT4-train-merges.json"
        default_epochs = 20
        default_lr_min = 1e-5
        default_lr_warmup = 200
        default_lr_max = 3e-4
        default_adam_weight_decay = 0.1
        default_batch_size = 32
        default_seq_length = 256
        default_embedding_dim = 512  # gpt suggests 256
        default_num_layers = 4
        default_num_attention_heads = 16
    else:  # owt
        # default_train_dataset = "data/owt_train.txt"
        # default_valid_dataset = "data/owt_valid.txt"
        default_epochs = 3
        default_lr_min = 1e-5
        default_lr_warmup = 2000
        default_lr_max = 1e-3
        default_adam_weight_decay = 0.01
        default_batch_size = 16
        default_seq_length = 1024
        default_embedding_dim = 512
        default_num_layers = 8
        default_num_attention_heads = 8

    parser.add_argument("--hf-tokenizer", action="store_true", default=False)
    parser.add_argument("--tokenizer-vocab-path", type=str, default=default_tokenizer_vocab)
    parser.add_argument("--tokenizer-merges-path", type=str, default=default_tokenizer_merges)

    parser.add_argument("--train-dataset-path", type=str, default=default_train_dataset)
    parser.add_argument("--valid-dataset-path", type=str, default=default_valid_dataset)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--lr-min", type=float, default=default_lr_min)
    parser.add_argument("--lr-warmup-steps", type=int, default=default_lr_warmup)
    parser.add_argument("--lr-max", type=float, default=default_lr_max)
    parser.add_argument("--adam-weight-decay", type=float, default=default_adam_weight_decay)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--seq-length", type=int, default=default_seq_length)
    parser.add_argument("--embedding-dim", type=int, default=default_embedding_dim)
    parser.add_argument("--num-layers", type=int, default=default_num_layers)
    parser.add_argument("--num-attention-heads", type=int, default=default_num_attention_heads)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def generate(
    args,
    model_path: str,
    prompt: list[str],
    seq_length: int,
):
    tokenizer = Tokenizer.from_files(
        args.tokenizer_vocab_path,
        args.tokenizer_merges_path,
        ["<|endoftext|>"],
    )
    encoded = tokenizer.encode_batched(prompt, True, args.seq_length, True)
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    data = torch.load(model_path)
    model = Transformer()
    model.load_state_dict(data["model"])
    for _ in range(seq_length):
        logits = model(input_ids, attention_mask)
        softmax(logits, dim=1)
    pass


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.hf_tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = Tokenizer.from_files(
            args.tokenizer_vocab_path,
            args.tokenizer_merges_path,
            ["<|endoftext|>"],
        )

    # TODO: when using owt, too big use memmap
    train_data = np.load(args.train_dataset_path)
    valid_data = np.load(args.valid_dataset_path)
    print(len(train_data), args.seq_length, args.batch_size)
    train_steps = len(train_data) // args.seq_length // args.batch_size
    valid_steps = len(valid_data) // args.seq_length // args.batch_size

    model = Transformer(
        tokenizer.vocab_size,
        args.seq_length,
        args.embedding_dim,
        args.num_layers,
        args.num_attention_heads,
    )
    model.to(device)

    epochs = args.epochs

    lr_min = args.lr_min
    lr_max = args.lr_max
    warmup_steps = args.lr_warmup_steps
    annealing_steps = train_steps * epochs

    run = wandb.init(project="cs336-assignment-01", config=vars(args))

    optim = AdamW(
        model.parameters(),
        lr=lr_min,
        weight_decay=args.adam_weight_decay,
    )
    # optim = SGD(model.parameters(), lr=1e3)

    use_checkpoint, load_at_epoch = False, 0
    # TODO: load wandb later. (continue it)
    # TODO: cursor linter is trash, why?
    if use_checkpoint:
        load_checkpoint(model, optim, f"./.models/gpt2-epoch-{load_at_epoch}.pt")

    def compute_inputs_loss(batch):
        input_ids, labels = batch
        output = model(input_ids, None)
        output_flatten = output.view(-1, output.shape[-1])
        labels = labels.contiguous().view(-1)
        # return cross_entropy_loss(output_flatten, labels)
        return F.cross_entropy(output_flatten, labels)

    best_valid_loss = float("inf")

    steps = 0
    for i in range(epochs):
        train_loss = 0
        model.train()
        pbar = tqdm(total=train_steps, desc=f"train-epoch {i + 1}")
        for j in range(train_steps):
            batch = data_loading(train_data, args.batch_size, args.seq_length, device)
            optim.zero_grad()
            loss = compute_inputs_loss(batch)
            train_loss += loss.item()
            loss.backward()
            # clip_gradients(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            lr = cos_lr_schedule(lr_min, lr_max, warmup_steps, annealing_steps, steps)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            optim.step()
            steps += 1

            if steps % 20 == 0:
                run.log({"lr": lr}, step=steps)
            pbar.update(1)

        train_loss = train_loss / train_steps
        print(f"epoch {i + 1} train_loss: {train_loss}")

        valid_loss = 0
        model.eval()
        with torch.inference_mode():
            pbar = tqdm(total=valid_steps, desc=f"valid-epoch {i + 1}")
            for _ in range(valid_steps):
                batch = data_loading(valid_data, args.batch_size, args.seq_length, device)
                valid_loss += compute_inputs_loss(batch).item()
                pbar.update()

        valid_loss = valid_loss / valid_steps
        print(f"epoch {i + 1} valid_loss: {valid_loss}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optim, f"./.models/gpt2-epoch-{i + 1}.pt")

        run.log({"train_loss": train_loss, "valid_loss": valid_loss, "steps": steps})


if __name__ == "__main__":
    train()
