import argparse
from collections import deque
import torch
from tqdm import tqdm
from types import SimpleNamespace

from src.models.tokenizer import Tokenizer
from torch.optim import AdamW
import torch.nn.functional as F
import os
import wandb
import numpy as np
from src.utils import (
    data_loading,
    save_checkpoint,
    load_checkpoint,
    cos_lr_schedule,
    AdamW as CustomAdamW,
    clip_gradients,
    # cross_entropy_loss,
)
from src.models.transformer import PosEmbeddingType, NormType, NormPosition, FFNType, Transformer

os.makedirs("./.models", exist_ok=True)

# gpu opt, torch.compile
# holy fuck, 20 it/s to 48, wtf
torch.set_float32_matmul_precision("high")
# torch.backends.cudnn.benchmark = True

# fuck, easier to experiment, a bit late
torch.manual_seed(42)
np.random.seed(42)


def get_tokenizer(args):
    return Tokenizer.from_files(args.tokenizer_vocab_path, args.tokenizer_merges_path, ["<|endoftext|>"])


def get_model_path(epoch, args):
    arch = f"{args.seq_length}-{args.embedding_dim}-{args.num_layers}-{args.num_attention_heads}"
    custom = f"{int(args.use_custom_adam)}-{int(args.use_custom_gradient_clipping)}"
    return f"./.models/{args.dataset}-epoch-{epoch}-lr-{args.lr_max}-batch-{args.batch_size}-arch-{arch}-custom-{custom}.pt"


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
        default_embedding_dim = 512
        default_num_layers = 4
        default_num_attention_heads = 16
    else:  # owt
        default_train_dataset = ".tokenizer/owt_train-encoded.npy"
        default_valid_dataset = ".tokenizer/owt_valid-encoded.npy"
        default_tokenizer_vocab = ".tokenizer/owt_train-vocab.json"
        default_tokenizer_merges = ".tokenizer/owt_train-merges.json"
        default_epochs = 5
        default_lr_min = 1e-5
        default_lr_warmup = 200  # TODO: consider more warm up steps for higher lr_max
        default_lr_max = 4e-3
        default_adam_weight_decay = 0.1  # 0.01
        default_batch_size = 64

        default_seq_length = 1024  # vs 512?
        default_embedding_dim = 768
        default_num_layers = 6
        default_num_attention_heads = 12

    # parser.add_argument("--hf-tokenizer", action="store_true", default=False)
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
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-ss", "--small-subset", action="store_true", default=False)
    parser.add_argument("-g", "--gpu-id", type=int, default=0)

    # compare slow down
    parser.add_argument("--use-custom-adam", action="store_true", default=False)
    parser.add_argument("--use-custom-gradient-clipping", action="store_true", default=False)

    # Oblations
    parser.add_argument("--pos-embedding", type=str, default="rope")  # rope, nope, sinusoidal
    parser.add_argument("--norm-type", type=str, default="rms")  # rms, layer, none
    parser.add_argument("--norm-position", type=str, default="pre")  # pre, post
    parser.add_argument("--ffn-type", type=str, default="swiglu")  # swiglu, silu

    # Further
    parser.add_argument("-mp", "--use-mixed-precision", action="store_true", default=False)
    parser.add_argument("--use-torch-compile", action="store_true", default=True)

    return parser.parse_args()


def train():
    args = get_args()
    print("[train]: ", vars(args))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args)

    train_data = np.load(args.train_dataset_path, mmap_mode="r")
    valid_data = np.load(args.valid_dataset_path, mmap_mode="r")
    train_steps = 327000000 // args.seq_length // args.batch_size
    valid_steps = len(valid_data) // args.seq_length // args.batch_size

    if args.small_subset:
        train_steps = int(train_steps * 0.005)
        valid_steps = int(valid_steps * 0.05)

    model = Transformer(
        tokenizer.vocab_size,
        args.seq_length,
        args.embedding_dim,
        args.num_layers,
        args.num_attention_heads,
        pos_embedding=PosEmbeddingType(args.pos_embedding.lower()),
        norm_type=NormType(args.norm_type.lower()),
        norm_position=NormPosition(args.norm_position.lower()),
        ffn_type=FFNType(args.ffn_type.lower()),
    )
    model.to(device)

    if args.use_torch_compile:
        model = torch.compile(model)
        print("[INFO] Pre-compiling model...")
        dummy_input = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_length), device=device)
        with torch.no_grad():
            _ = model(dummy_input, None)
        print("[INFO] Model compilation complete.")
        torch.cuda.empty_cache()  # Clean up compilation memory

    epochs, lr_min, lr_max, warmup_steps = args.epochs, args.lr_min, args.lr_max, args.lr_warmup_steps
    annealing_steps = train_steps * epochs

    run = wandb.init(
        id=args.wandb_id,
        project="assignment-01-owt",
        config=vars(args),
    )

    AdamCLS = CustomAdamW if args.use_custom_adam else AdamW
    grad_clipping_fn = clip_gradients if args.use_custom_gradient_clipping else torch.nn.utils.clip_grad_norm_

    optim = AdamCLS(
        model.parameters(),
        lr=lr_min,
        weight_decay=args.adam_weight_decay,
    )

    if args.checkpoint:
        load_checkpoint(model, optim, args.checkpoint)

    def compute_inputs_loss(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_mixed_precision):
            input_ids, labels = batch
            output = model(input_ids, None)
            output_flatten = output.view(-1, output.shape[-1])
            labels = labels.contiguous().view(-1)
            # return cross_entropy_loss(output_flatten, labels)
            loss = F.cross_entropy(output_flatten, labels)
        return loss

    best_valid_loss = float("inf")
    gradient_norms = []
    loss_history = deque(maxlen=100)

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

            grad_norm = grad_clipping_fn(model.parameters(), max_norm=1.0)

            gradient_norms.append(grad_norm.item())
            loss_history.append(loss.item())

            lr = cos_lr_schedule(lr_min, lr_max, warmup_steps, annealing_steps, steps)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            optim.step()
            steps += 1

            if steps % 20 == 0:
                recent_grad_norm = np.mean(gradient_norms[-20:])
                loss_moving_avg = np.mean(loss_history)
                loss_std = np.std(loss_history)

                run.log(
                    {
                        "lr": lr,
                        "grad_norm": recent_grad_norm,
                        "loss_moving_avg": loss_moving_avg,
                        "loss_std": loss_std,
                        "loss_variance": loss_std**2,
                    },
                    step=steps,
                )
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
            save_checkpoint(
                model,
                optim,
                get_model_path(i + 1, args),
                args=vars(args),
                iteration=i + 1,
            )

        run.log({"train_loss": train_loss, "valid_loss": valid_loss, "steps": steps})


def isolated_validation_check(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(model_path, map_location=device)
    args = SimpleNamespace(**data["args"]) if "args" in data else get_args()
    print("isolated_validation_check args:", args)
    tokenizer = get_tokenizer(args)

    model = Transformer(
        tokenizer.vocab_size,
        args.seq_length,
        args.embedding_dim,
        args.num_layers,
        args.num_attention_heads,
        pos_embedding=PosEmbeddingType(args.pos_embedding.lower()),
        norm_type=NormType(args.norm_type.lower()),
        norm_position=NormPosition(args.norm_position.lower()),
        ffn_type=FFNType(args.ffn_type.lower()),
    )
    valid_data = np.load(args.valid_dataset_path)

    if args.use_torch_compile:
        model = torch.compile(model)

    model.load_state_dict(data["model"])
    model.to(device)
    valid_loss = 0
    valid_steps = len(valid_data) // args.seq_length // args.batch_size
    with torch.inference_mode():
        pbar = tqdm(total=valid_steps, desc="valid-dataset")
        for _ in range(valid_steps):
            batch = data_loading(valid_data, args.batch_size, args.seq_length, device)
            input_ids, labels = batch
            output = model(input_ids, None)
            output_flatten = output.view(-1, output.shape[-1])
            labels = labels.contiguous().view(-1)
            valid_loss += F.cross_entropy(output_flatten, labels).item()
            pbar.update()
    print(valid_loss / valid_steps)


if __name__ == "__main__":
    train()
    # isolated_validation_check(".models/owt-epoch-8-lr-0.004-batch-64-arch-1024-768-6-12.pt")
