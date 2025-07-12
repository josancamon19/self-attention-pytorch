from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import sys
import os
import subprocess
import torch
import wandb

SMOKE_TEST = True

# Architecture search space based on your configs
config = {
    "embedding_dim": tune.choice([768, 1024, 1280]),
    "num_layers": tune.choice([6, 8, 12, 16, 20, 24]),
    "num_heads": tune.choice([8, 12, 16, 20]),
    "tokens": 1e9,
    "smoke_test": SMOKE_TEST,
}


def train_transformer_architecture(config):
    """Train a transformer with the given architecture config."""

    # Validate architecture constraints
    embedding_dim = config["embedding_dim"]
    num_heads = config["num_heads"]

    # Skip invalid head configurations
    if embedding_dim % num_heads != 0:
        # Report failure for invalid configs
        train.report({"valid_loss": float("inf"), "status": "invalid_config"})
        return

    # Skip configurations where head dimension is too small
    head_dim = embedding_dim // num_heads
    if head_dim < 64:
        train.report({"valid_loss": float("inf"), "status": "head_dim_too_small"})
        return

    # Create wandb run ID for this trial
    wandb_id = f"arch_search_{embedding_dim}_{config['num_layers']}_{num_heads}"

    # Get assigned GPU ID from Ray
    gpu_id = train.get_context().get_trial_resources().get("gpu", [0])[0] if torch.cuda.is_available() else 0

    # Build command
    cmd = [
        sys.executable,
        "src/train/transformer.py",
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
        str(10),
    ]

    try:
        # Run training subprocess
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=900 if SMOKE_TEST else 1800,  # 1-2 hours timeout
        )

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            train.report({"valid_loss": float("inf"), "status": "training_failed"})
            return

        run = wandb.Api().run(f"your-entity/assignment-01-owt/{wandb_id}")
        history = run.history(keys=["valid_loss"])
        # if len(history) > 0:
        final_valid_loss = history["valid_loss"].iloc[-1]
        train.report({"valid_loss": final_valid_loss, "status": "completed"})

    except subprocess.TimeoutExpired:
        train.report({"valid_loss": float("inf"), "status": "timeout"})
    except Exception as e:
        print(f"Error in training: {e}")
        train.report({"valid_loss": float("inf"), "status": "error"})


def main():
    # Multi-GPU configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected {num_gpus} GPUs")

    # Always use 1 GPU per trial - compute bound, not memory bound

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=10,  # Max iterations
        grace_period=1,
        reduction_factor=2,
    )

    # Run as many concurrent trials as we have GPUs
    max_concurrent_trials = num_gpus if num_gpus > 0 else 1

    tuner = tune.Tuner(
        tune.with_resources(train_transformer_architecture, resources={"gpu": 1} if num_gpus > 0 else {}),
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=max_concurrent_trials * 10,  # Scale with GPU count for more experiments
            max_concurrent_trials=max_concurrent_trials,
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("valid_loss", "min")

    print("\n" + "=" * 50)
    print("ARCHITECTURE SEARCH RESULTS")
    print("=" * 50)
    print(f"Best config: {best_result.config}")
    print(f"Best validation loss: {best_result.metrics['valid_loss']:.4f}")
    print(f"Total parameters: {best_result.metrics.get('total_params', 'Unknown'):,}")
    print(f"Efficiency (loss/M params): {best_result.metrics.get('efficiency', 'Unknown'):.4f}")

    # Print top 5 results
    print("\nTop 5 configurations:")
    for i, result in enumerate(results.get_dataframe().nsmallest(5, "valid_loss").iterrows()):
        config_data = result[1]
        print(
            f"{i + 1}. Loss: {config_data['valid_loss']:.4f} | "
            f"dim: {config_data['config/embedding_dim']} | "
            f"layers: {config_data['config/num_layers']} | "
            f"heads: {config_data['config/num_heads']}"
        )


if __name__ == "__main__":
    main()
