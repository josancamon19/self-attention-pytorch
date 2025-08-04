from ray import tune
import ray
import sys
import os
import subprocess
import torch
import wandb
import uuid

max_time_minutes = 10  # let it finish the run til the end, even if it's a few more FLOPs, standardize better later
config = {
    "batch_size": tune.grid_search([64]),
    "lr": tune.grid_search([4e-3]),  # 1e-2
    # "lr": tune.grid_search([1e-2]), # 1e-2
    "embedding_dim": tune.grid_search([768, 1024, 1280]),
    "num_layers": tune.grid_search([6, 8, 12, 16]),
    "num_heads": tune.grid_search([8, 12, 16]),
    "ffn_type": tune.grid_search(["relu2", "swiglu"]),  # "swiglu",
    "qk_norm": tune.grid_search([1]),
    "qk_norm_type": tune.grid_search(["rms"]),  # l2 (default), rms
    # "tokens": tune.grid_search([2.2e9]),  # 2.2e9
    "warmup_steps": tune.grid_search([300]),  # 300, 1200,
    # "lr_annealing_multiplier": tune.grid_search([1.0, 1.1, 1.2, 1.3])
}

# 8 minutes, D?
# 6, 768 ≈ 95M, 2e8
# 8, 768 ≈ 105M, ≈ 1.9e8
# 12, 768 ≈ 134M, ≈ 1.7e8
# 16, 768 ≈ 162M, ≈ 1.35e8

# 6, 1024 ≈ 141M, ≈ 1.92e8
# 8, 1024 ≈ 166M, ≈ 1.7e8
# 12, 1024 ≈ 217M, ≈ 1.25e8
# 16, 1024 ≈ 267M, ≈ 9.5e7

# 6, 1280 ≈ 199M, ≈ 1.6e8
# 8, 1280 ≈ 238M, ≈ 1.25e8
# 12, 1280 ≈ 316M, ≈ 9e7
# 16, 1280 ≈ 395M, ≈ 7.2e7

ray.init(
    runtime_env={
        "excludes": [
            ".tokenizer/TinyStoriesV2-GPT4-train-encoded.npy",
            ".tokenizer/TinyStoriesV2-GPT4-valid-encoded.npy",
            ".tokenizer/owt_valid-encoded.npy",
            ".tokenizer/owt_train-encoded.npy",  # Add this too if it exists
        ]
    }
)

# Experiments
# - High learning rates = 3e-3 worst than 1e-3 (best), 6e-2 exploded (TODO: check where is exploding)
# - RMSNorm using elementwise instead of scalar = ok, better
# - For QK Norm, can try rms vs l2 normalization = holy shit clear win.
# - adding a bunch of logs
# - - we'll see if 6e-3 is exploding, yea it is, check where/why
# - rope no complex + .compile error
# - full run explodes with warmup 300, 1200
# - - clearly get's unstable, lots' of spikes, trying 3000 warmup steps.
# - - still explodes
# TODO: if any of this explode, log, linear output softmax, try z-loss, logit cap


# TODO: flash attn any improvement?

# TODO: Mup initializations trick
# TODO: Scheduler variations, Hexagon kinda, cosine lr schedule longer than
# TODO: Adam with different lr's per layer (heads vs embeddings vs else) https://arxiv.org/pdf/2502.19002 https://arxiv.org/pdf/2406.16793

# TODO: Muon and WSD optimizers

# python src/train/transformer.py --num-layers 6 --num-heads 12 --embedding-dim 768 --batch-size 64 --lr-max 4e-3 --lr-warmup-steps 4000 --tokens 2.3e9 --ffn-type relu2 -tc


def train_transformer_architecture(config):
    embedding_dim = config["embedding_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    head_dim = embedding_dim // num_heads

    if embedding_dim % num_heads != 0:
        tune.report({"valid_loss": float("inf"), "status": "embedding_dim % num_heads != 0"})
        return

    if head_dim not in [64, 128]:
        tune.report({"valid_loss": float("inf"), "status": "head_dim < 64"})
        return

    # Set tokens manually based on (layers, d_model) tuples
    tokens_map = {  # each sweep 8 minutes
        (6, 768): 2e8,
        (8, 768): 1.9e8,
        (12, 768): 1.7e8,
        (16, 768): 1.35e8,
        (6, 1024): 1.92e8,
        (8, 1024): 1.7e8,
        (12, 1024): 1.25e8,
        (16, 1024): 9.5e7,
        (6, 1280): 1.6e8,
        (8, 1280): 1.25e8,
        (12, 1280): 9e7,
        (16, 1280): 7.2e7,
    }

    tokens = tokens_map.get((num_layers, embedding_dim), 2e8)  # default fallback

    ffn_type = config.get("ffn_type", "swiglu")
    architecture = f"arch_{embedding_dim}_{config['num_layers']}_{num_heads}"
    wandb_id = f"search_{architecture}_{config['lr']}_{config['warmup_steps']}_{config['qk_norm']}_{config['qk_norm_type']}_{ffn_type}_{uuid.uuid4().hex[:8]}"
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
        str(int(tokens)),
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
        # "300",  # 2% of about 15000 steps
        str(config["warmup_steps"]),
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
        "--ffn-type",
        ffn_type,
        "--lr-annealing-multiplier",
        str(1.0),
    ]

    if config.get("qk_norm"):
        cmd.append("--qk-norm")
        cmd.append("--qk-norm-type")
        cmd.append(str(config["qk_norm_type"]))

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
