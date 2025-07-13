import argparse
from collections import deque
import torch
from tqdm import tqdm
import time

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
    inverse_sqrt_schedule,
    constant_with_warmup,
    linear_lr_schedule,
    cosine_with_restarts,
    AdamW as CustomAdamW,
    clip_gradients,
    # single_data_loading,
    # cross_entropy_loss,
)
from src.models.transformer import Transformer

os.makedirs("./.models", exist_ok=True)

# gpu opt, torch.compile
# holy fuck, 20 it/s to 48, wtf
torch.set_float32_matmul_precision("high")

# fuck, easier to experiment, a bit late
torch.manual_seed(42)
np.random.seed(42)


def get_model_path(step, args):
    arch = f"{args.seq_length}-{args.embedding_dim}-{args.num_layers}-{args.num_heads}"
    custom = f"{int(args.use_custom_adam)}-{int(args.use_custom_gradient_clipping)}"
    return (
        f"./.models/{args.dataset}-step-{step}-lr-{args.lr_max}-batch-{args.batch_size}-arch-{arch}-custom-{custom}.pt"
    )


schedule_map = {
    "cosine": cos_lr_schedule,
    "linear": linear_lr_schedule,
    "sqrt": inverse_sqrt_schedule,
    "constant": constant_with_warmup,
    "cosine_restarts": cosine_with_restarts,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["tinystories", "owt"], default="owt")

    args, _ = parser.parse_known_args()

    if args.dataset == "tinystories":
        default_train_dataset = ".tokenizer/TinyStoriesV2-GPT4-train-encoded.npy"
        default_valid_dataset = ".tokenizer/TinyStoriesV2-GPT4-valid-encoded.npy"
        default_tokenizer_vocab = ".tokenizer/TinyStoriesV2-GPT4-train-vocab.json"
        default_tokenizer_merges = ".tokenizer/TinyStoriesV2-GPT4-train-merges.json"
        training_tokens = 3.27e8  # 327000000

        default_lr_min = 1e-5
        default_lr_warmup = 200
        default_lr_max = 3e-4
        default_adam_weight_decay = 0.1
        default_batch_size = 32
        default_seq_length = 256
        default_embedding_dim = 512
        default_num_layers = 4
        default_num_heads = 16
    else:  # owt
        default_train_dataset = ".tokenizer/owt_train-encoded.npy"
        default_valid_dataset = ".tokenizer/owt_valid-encoded.npy"
        default_tokenizer_vocab = ".tokenizer/owt_train-vocab.json"
        default_tokenizer_merges = ".tokenizer/owt_train-merges.json"
        training_tokens = 2.6e9

        default_lr_min = 1e-5
        default_lr_warmup = 1200
        default_lr_max = 7e-4
        default_adam_weight_decay = 0.1  # 0.01
        default_batch_size = 64

        default_seq_length = 512
        default_embedding_dim = 1280
        default_num_layers = 6
        default_num_heads = 8

    # parser.add_argument("--hf-tokenizer", action="store_true", default=False)
    parser.add_argument("--tokenizer-vocab-path", type=str, default=default_tokenizer_vocab)
    parser.add_argument("--tokenizer-merges-path", type=str, default=default_tokenizer_merges)
    parser.add_argument("--train-dataset-path", type=str, default=default_train_dataset)
    parser.add_argument("--valid-dataset-path", type=str, default=default_valid_dataset)

    # architecture
    parser.add_argument("--seq-length", type=int, default=default_seq_length)
    parser.add_argument("--embedding-dim", type=int, default=default_embedding_dim)
    parser.add_argument("--num-layers", type=int, default=default_num_layers)
    parser.add_argument("--num-heads", type=int, default=default_num_heads)

    parser.add_argument("-tt", "--tokens", type=float, default=training_tokens)

    parser.add_argument("--batch-size", type=int, default=default_batch_size)

    parser.add_argument("--lr-min", type=float, default=default_lr_min)
    parser.add_argument("--lr-max", type=float, default=default_lr_max)
    parser.add_argument("--lr-warmup-steps", type=int, default=default_lr_warmup)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "sqrt", "constant"],  # "cosine_restarts"
    )
    # parser.add_argument("--lr-cosine-restarts", type=int, default=0)
    parser.add_argument("--adam-weight-decay", type=float, default=default_adam_weight_decay)

    # compare slow down
    parser.add_argument("-uca", "--use-custom-adam", action="store_true", default=False)
    parser.add_argument("-ucgc", "--use-custom-gradient-clipping", action="store_true", default=False)

    # Oblations
    parser.add_argument("--pos-embedding", type=str, default="rope")  # rope, nope, sinusoidal
    parser.add_argument("--norm-type", type=str, default="rms")  # rms, layer, none
    parser.add_argument("--norm-position", type=str, default="pre")  # pre, post
    parser.add_argument("--ffn-type", type=str, default="swiglu")  # swiglu, silu
    parser.add_argument("--weight-tying", action="store_true", default=False)
    parser.add_argument("--qk-norm", action="store_true", default=False)

    # Further
    parser.add_argument("-mp", "--use-mixed-precision", action="store_true", default=True)
    parser.add_argument("-tc", "--use-torch-compile", action="store_true", default=False)

    # ops
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("-g", "--gpu-id", type=int, default=0)
    parser.add_argument("--max-wall-time", type=int, default=None, help="Maximum wall clock time in minutes")

    return parser.parse_args()


# Add this function to your training script CLAUDE IMPLEMENTS
def get_simple_diagnostic_metrics(model, loss: torch.Tensor, step: int, initial_loss: float | None = None):
    """Just the essentials for architectural comparison."""
    # They make sense, but had never seen them, so gotta check, many of this are relative values, cause are upscaled
    metrics = {}

    # 1. Are layers learning different things? (Capacity)
    with torch.no_grad():
        # Just check if first and last layer weights are becoming similar
        first_layer_weight = model.blocks[0].attention.QKV.weights
        last_layer_weight = model.blocks[-1].attention.QKV.weights

        # Simple cosine similarity
        weight_similarity = F.cosine_similarity(
            first_layer_weight.flatten().unsqueeze(0), last_layer_weight.flatten().unsqueeze(0)
        ).item()

        metrics["capacity/layer_similarity"] = weight_similarity  # Want this LOW

    # 2. How efficiently are we learning? (Efficiency)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # millions

    metrics["efficiency/loss_per_million_params"] = loss.item() / total_params
    metrics["efficiency/params_millions"] = total_params

    # 3. Are we using the full model? (Dead neurons in FFN)
    # Sample one middle layer
    mid_layer = len(model.blocks) // 2
    ffn_weight = model.blocks[mid_layer].pos_wise.W2.weights

    # Neurons with very small outgoing weights are probably dead
    dead_neurons = (ffn_weight.abs().max(dim=0)[0] < 0.01).float().mean()
    metrics["capacity/dead_neurons_percent"] = dead_neurons.item() * 100

    if initial_loss is not None:
        loss_reduction = initial_loss - loss.item()
        # How much loss reduction per million parameters per step
        learning_efficiency = (loss_reduction / total_params) / step * 1000
        metrics["efficiency/learning_efficiency"] = learning_efficiency

    return metrics


def compute_batch_loss(model, args, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_mixed_precision):
        input_ids, labels = batch
        output = model(input_ids, None)

        # if torch.isnan(output).any():
        #     print("[NaN DETECTED] Model output contains NaN values!")
        #     print(f"[NaN DETECTED] Output shape: {output.shape}")
        #     print(f"[NaN DETECTED] NaN count: {torch.isnan(output).sum().item()}")
        #     # Check which parameters have NaN
        #     for name, param in model.named_parameters():
        #         if torch.isnan(param).any():
        #             print(f"[NaN DETECTED] Parameter {name} has NaN values")

        # if torch.isinf(output).any():
        #     print("[INF DETECTED] Model output contains Inf values!")
        #     print(f"[INF DETECTED] Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        output_flatten = output.view(-1, output.shape[-1])
        labels = labels.contiguous().view(-1)
        loss = F.cross_entropy(output_flatten, labels)

        if torch.isnan(loss):
            print("NaN loss detected!")
            raise Exception("loss is nan")  # quick exit, don't waste compute
        # return cross_entropy_loss(output_flatten, labels)
        return loss


def execute_validation_loss(
    model,
    optim,
    args,
    device,
    valid_data,
    wandb_run,
    best_valid_loss: float,
    valid_steps: int,
    step: int,
    save_best: bool = True,
    show_progress: bool = True,
):
    valid_loss = 0
    with torch.inference_mode():
        for _ in tqdm(range(valid_steps), total=valid_steps, desc="validation-check", disable=not show_progress):
            batch = data_loading(valid_data, args.batch_size, args.seq_length, device)
            valid_loss += compute_batch_loss(model, args, batch).item()

    valid_loss = valid_loss / valid_steps
    wandb_run.log({"valid_loss": valid_loss}, step=step)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        if save_best:
            save_checkpoint(model, optim, get_model_path(step, args), args=vars(args), iteration=step)
            print(f"saved model at step {step} with valid_loss: {valid_loss}")

    return best_valid_loss


def train():
    # TODO: pass ray here to avoid bad iterations fast
    args = get_args()
    print("[train]: ", vars(args))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    lr_schedule_fn = schedule_map[args.lr_schedule]

    train_data = np.load(args.train_dataset_path, mmap_mode="r")
    valid_data = np.load(args.valid_dataset_path, mmap_mode="r")

    train_steps = int(args.tokens / args.seq_length / args.batch_size)
    valid_steps = int(len(valid_data) / args.seq_length / args.batch_size)

    model: Transformer = Transformer.from_args(args)
    model.to(device)

    if args.use_torch_compile:
        model = torch.compile(model)

    run = wandb.init(id=args.wandb_id, project="assignment-01-owt", config=vars(args))

    AdamCLS = CustomAdamW if args.use_custom_adam else AdamW
    grad_clipping_fn = clip_gradients if args.use_custom_gradient_clipping else torch.nn.utils.clip_grad_norm_

    optim = AdamCLS(
        model.parameters(),
        lr=args.lr_min,
        weight_decay=args.adam_weight_decay,
        betas=(0.9, 0.95),
    )

    if args.checkpoint:
        load_checkpoint(model, optim, args.checkpoint)

    # gradient_norms = deque(maxlen=20)
    # loss_history = deque(maxlen=100)
    # initial_loss = None

    # inputs, labels = single_data_loading(train_data, args.batch_size, train_steps, args.seq_length, device)

    best_valid_loss = float("inf")
    train_loss = 0
    model.train()

    # Wall clock timer setup
    start_time = time.time()
    max_wall_seconds = args.max_wall_time * 60 if args.max_wall_time else None

    for step in tqdm(range(1, train_steps + 1), total=train_steps, desc="training-steps"):
        batch = data_loading(train_data, args.batch_size, args.seq_length, device)
        optim.zero_grad()
        # loss = compute_inputs_loss(model, args, (inputs[step - 1, ...], labels[step - 1, ...]))
        loss = compute_batch_loss(model, args, batch)
        train_loss += loss.item()
        loss.backward()

        # if step == 1:
        #     initial_loss = loss.item()

        grad_norm = grad_clipping_fn(model.parameters(), max_norm=1.0)

        # if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        #     print(f"[GRAD NaN/INF] Step {step}: Gradient norm is {grad_norm.item()}")
        #     # Check individual parameter gradients
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             param_grad_norm = torch.linalg.vector_norm(param.grad)
        #             if torch.isnan(param_grad_norm) or torch.isinf(param_grad_norm):
        #                 print(f"[GRAD NaN/INF] Parameter {name} grad norm: {param_grad_norm.item()}")

        #     raise Exception("grad_norm has nan's")

        # gradient_norms.append(grad_norm.item())
        # loss_history.append(loss.item())

        lr = lr_schedule_fn(args.lr_min, args.lr_max, args.lr_warmup_steps, train_steps, step)
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        optim.step()

        if step % 50 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # recent_grad_norm = np.mean(gradient_norms[-20:])
            # loss_moving_avg = np.mean(loss_history)
            # loss_std = np.std(loss_history)
            # param_norm = torch.linalg.vector_norm(
            #     torch.stack([torch.linalg.vector_norm(p) for p in model.parameters()])
            # )

            run.log(
                {
                    "train_loss": train_loss / step,
                    "lr": lr,
                    "speed": step / elapsed_time,
                    # "stability/grad_norm": recent_grad_norm,
                    # "stability/loss_moving_avg": loss_moving_avg,
                    # "stability/loss_std": loss_std,
                    # "stability/loss_variance": loss_std**2,
                    # "stability/param_norm": param_norm.item(),
                },
                step=step,
            )

            # Check wall clock limit
            if max_wall_seconds and elapsed_time >= max_wall_seconds:
                print(f"\n[WALL CLOCK] Reached time limit of {args.max_wall_time} minutes ({elapsed_time:.1f}s)")
                print(f"[WALL CLOCK] Completed {step}/{train_steps} steps ({step / train_steps * 100:.1f}%)")
                train_steps = step  # train_loss / train_steps work
                break

            if step % 1000 == 0:
                best_valid_loss = execute_validation_loss(
                    model, optim, args, device, valid_data, run, best_valid_loss, 20, step, False, False
                )
                # diagnostic_metrics = get_simple_diagnostic_metrics(model, loss, step, initial_loss)
                # run.log(diagnostic_metrics, step=step)

    print(f"Final training loss: {train_loss / train_steps:.6f}")
    execute_validation_loss(model, optim, args, device, valid_data, run, best_valid_loss, valid_steps, step)


if __name__ == "__main__":
    train()
    # isolated_validation_check(".models/owt-epoch-8-lr-0.004-batch-64-arch-1024-768-6-12.pt")
