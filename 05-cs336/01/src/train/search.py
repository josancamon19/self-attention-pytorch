from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import sys
import os
import subprocess
import torch
import wandb
import uuid

# Architecture search space based on your configs
# max_time_minutes = 15
# config = {
#     "embedding_dim": tune.grid_search([768, 1024, 1280]),
#     "num_layers": tune.grid_search([6, 8, 12, 16, 20, 24]),
#     "num_heads": tune.grid_search([8, 12, 16, 20]),
#     "tokens": 2e8,
# }

max_time_minutes = 25
config = {
    "embedding_dim": tune.grid_search([1280]),
    "num_layers": tune.grid_search([6]),
    "num_heads": tune.grid_search([8]),
    # "batch_size": tune.grid_search([16, 32, 48, 64, 96, 128, 192]),
    "batch_size": tune.grid_search([64]),  # maybe 48
    "lr": tune.grid_search([5e-4, 6e-4, 7e-4, 9e-4, 1e-3, 3e-3, 4e-3]),  # 3e-4 is the default used before
    "tokens": 5e8,  # (closer to the max amount it should be able to execute)
}
# lr, 1e-4, 6e-4 9e-4 1e-3 3e-3 4e-3 6e-3

# Best architectures (batch size 64, 300 warmup, tokens 5e8)
# final 1280, 6, 8
# maybe 1280, 6, 10

# Best batch sizes (default was 64)
# 64, 48

# Best lr's
#

# Maybe try higher ones, with qk norm

# Run multiple full trains 90 minutes on 2.6B tokens. (almost max 13x model size instead of 20x scaling laws)
# different warmup steps


def train_transformer_architecture(config):
    embedding_dim = config["embedding_dim"]
    num_heads = config["num_heads"]
    head_dim = embedding_dim // num_heads

    if embedding_dim % num_heads != 0:
        tune.report({"valid_loss": float("inf"), "status": "embedding_dim % num_heads != 0"})
        return

    if head_dim < 64:
        tune.report({"valid_loss": float("inf"), "status": "head_dim < 64"})
        return

    # if (embedding_dim, config["num_layers"], num_heads) not in valid_only:
    #     tune.report({"valid_loss": float("inf"), "status": "invalid_arch"})
    #     return

    # wandb_id = f"arch_search_{embedding_dim}_{config['num_layers']}_{num_heads}_{uuid.uuid4().hex[:8]}"
    wandb_id = f"lr_search_{config['lr']}_{uuid.uuid4().hex[:8]}"
    # Get assigned GPU ID from Ray
    # gpu_id = train.get_context().get_trial_resources().get("gpu", [0])[0] if torch.cuda.is_available() else 0
    # apparently ray always makes each gpu visible for each process as 0
    gpu_id = config.get("gpu_id", 0)

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src/train/
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Gets project root
    train_script = os.path.join(project_root, "src", "train", "transformer.py")

    # clear torch.compile cache on every iteration, cause when changing architectures .compile causes nan's wtf.
    trial_id = f"{embedding_dim}_{config['num_layers']}_{num_heads}"
    cache_dir = f"/tmp/torch_compile_cache_{trial_id}"
    env = os.environ.copy()
    env["TORCH_COMPILE_CACHE_DIR"] = cache_dir

    cmd = [
        sys.executable,
        train_script,
        "--tokens",
        str(int(config["tokens"])),
        "--embedding-dim",
        str(embedding_dim),
        "--num-layers",
        str(config["num_layers"]),
        "--num-heads",
        str(num_heads),
        "--wandb-id",
        wandb_id,
        "--dataset",
        "owt",
        "--gpu-id",
        str(gpu_id),
        "--max-wall-time",
        str(max_time_minutes),
        # relative path issues
        "--train-dataset-path",
        f"{project_root}/.tokenizer/owt_train-encoded.npy",
        "--valid-dataset-path",
        f"{project_root}/.tokenizer/owt_valid-encoded.npy",
        "--tokenizer-vocab-path",
        f"{project_root}/.tokenizer/owt_train-vocab.json",
        "--tokenizer-merges-path",
        f"{project_root}/.tokenizer/owt_train-merges.json",
        "--lr-warmup-steps",
        "300",  # 5% of abotu 6000 steps
        "--lr-max",
        str(config["lr"]),  # started with 3e-4
        "--lr-schedule",
        "cosine",
        "--adam-weight-decay",
        "0.1",
        "--batch-size",
        str(config["batch_size"]),
        "--use-mixed-precision",
        "--use-torch-compile",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=(max_time_minutes + 5) * 60,
            env=env,
        )
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            tune.report({"valid_loss": float("inf"), "status": "training_failed"})
            return

        run = wandb.Api().run(f"assignment-01-owt/{wandb_id}")
        final_valid_loss = run.history(keys=["valid_loss"])["valid_loss"].iloc[-1]
        # final_train_loss = run.history(keys=["train_loss"])["train_loss"].iloc[-1]
        tune.report({"valid_loss": final_valid_loss, "status": "completed"})

    except subprocess.TimeoutExpired:
        tune.report({"valid_loss": float("inf"), "status": "timeout"})
        # lots like this, they should be still alive, check wandb instead of this
    except Exception as e:
        print(f"Error in training: {e}")
        tune.report({"valid_loss": float("inf"), "status": "error"})
    finally:
        import shutil

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected {num_gpus} GPUs")
    max_concurrent_trials = num_gpus if num_gpus > 0 else 1

    tuner = tune.Tuner(
        tune.with_resources(train_transformer_architecture, resources={"gpu": 1} if num_gpus > 0 else {}),
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            max_concurrent_trials=max_concurrent_trials,
        ),
        param_space=config,
    )

    tuner.fit()
    print("\n" + "=" * 50)
    print("ARCHITECTURE SEARCH FINISHED")
    print("=" * 50)


if __name__ == "__main__":
    main()
