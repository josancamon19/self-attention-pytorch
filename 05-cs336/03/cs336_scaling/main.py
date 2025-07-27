# Instructions
# - FLOPs budget of 1e19
# - operate in smaller scale first
# - Be sure to carefully plan your runs before you get started
# - once scaling laws budget of 2e18 is consumed, no further api requests
# - Batch size {128, 256}

# Q
# - Given 2e18 budget, how did you decide which runs to query?
# - Runs result, scaling laws fitting
# - - [ ] Fit for another method besides IsoFLOPs
# - Given the budget, what params/loss your scaling laws predict.
# - what hyper-parameters would you use given your predicted optimal params?

# ----


import argparse
import json
import requests
import os
from datetime import datetime

# Ensure the configs directory exists
os.makedirs(".configs", exist_ok=True)
BASE_URL = "http://hyperturing.stanford.edu:8000"
api_key = os.getenv("API_KEY", "123")


def get_total_flops():
    response = requests.get(f"{BASE_URL}/total_flops_used", {"api_key": api_key})
    return response.json()["total_flops_used"]


def get_previous_runs():
    response = requests.get(f"{BASE_URL}/previous_runs", {"api_key": api_key})
    return response.json()


def estimate_params(d_model, num_layers):
    # - Query, Key, Value projections: 3 * d_model * d_model = 3 * d_model^2
    # - Output projection: d_model * d_model = d_model^2
    # - Total: 4 * d_model^2
    # 2. Feed-Forward Network:
    # - First linear: d_model * d_ff (where d_ff = 4 * d_model)
    # - Second linear: d_ff * d_model
    # - Total: d_model * 4*d_model + 4*d_model * d_model = 8 * d_model^2
    # 3. Layer Norms: Negligible compared to linear layers
    # Per layer total: 4 * d_model^2 + 8 * d_model^2 = 12 * d_model^2
    # For all layers: num_layers * 12 * d_model^2
    return 12 * num_layers * (d_model**2)


def query_loss(d_model, num_layers, num_heads, batch_size, lr, train_flops):
    """Query the API for loss given hyperparameters"""
    # querying this endpoint with a previously queried doesn't discount FLOPs, tho save them and assert when calling
    # endpoint returns (loss, total_flops_used (new total))

    payload = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "lr": lr,
        "train_flops": train_flops,
        "api_key": api_key,
    }

    response = requests.post(f"{BASE_URL}/loss", json=payload)
    result = response.json()

    # Add estimated params and timestamp
    result["estimated_params"] = estimate_params(d_model, num_layers)
    result["timestamp"] = datetime.now().isoformat()
    result["config"] = payload

    return result


def save_run(result, filename="runs.jsonl"):
    os.makedirs("results", exist_ok=True)
    filepath = f"results/{filename}"
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved run to {filepath}")


def load_runs(filename="runs.jsonl"):
    filepath = f"results/{filename}"
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f]


def get_parser_args():
    parser = argparse.ArgumentParser(description="Query scaling law API")
    parser.add_argument(
        "--action",
        choices=["status", "previous", "run"],
        default="status",
        help="Action to perform",
    )

    # Query parameters
    parser.add_argument("--d_model", type=int, help="Model dimension [64-1024]")
    parser.add_argument("--num_layers", type=int, help="Number of layers [2-24]")
    parser.add_argument("--num_heads", type=int, help="Number of heads [2-16]")
    parser.add_argument("--batch_size", type=int, choices=[128, 256], help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate [1e-4, 1e-3]")
    parser.add_argument("--train_flops", type=float, help="Training FLOPs")

    args = parser.parse_args()
    return args


def main():
    args = get_parser_args()

    if args.action == "status":
        flops = get_total_flops()
        print(f"Total FLOPs used: {flops:.2e}")
        print(f"Remaining: {2e18 - flops:.2e}")

    elif args.action == "previous":
        runs = get_previous_runs()
        print(f"Found {len(runs)} previous runs")
        for i, run in enumerate(runs[:5]):  # Show first 5
            print(
                f"  {i + 1}: loss={run.get('loss', 'N/A'):.4f}, params={run.get('estimated_params', 'N/A')}"
            )

    elif args.action == "run":
        if not all(
            [
                args.d_model,
                args.num_layers,
                args.num_heads,
                args.batch_size,
                args.lr,
                args.train_flops,
            ]
        ):
            print("Missing required parameters for query")
            return

        print(
            f"Querying: d_model={args.d_model}, layers={args.num_layers}, "
            f"heads={args.num_heads}, bs={args.batch_size}, lr={args.lr}, "
            f"flops={args.train_flops:.2e}"
        )

        result = query_loss(
            args.d_model,
            args.num_layers,
            args.num_heads,
            args.batch_size,
            args.lr,
            args.train_flops,
        )

        print(f"Loss: {result['loss']:.4f}")
        print(f"Total FLOPs: {result['total_flops_used']:.2e}")
        print(f"Estimated params: {result['estimated_params']:.2e}")

        save_run(result)


def sanity_check():
    args = get_parser_args()
    seq_length, vocab_size = 512, 32000
    params = estimate_params(args.d_model, args.num_layers)
    other_params = (
        vocab_size * args.d_model
        + seq_length * args.d_model
        + args.d_model * vocab_size  # embedding + pos + output head
    )

    tokens = args.train_flops / (6 * params)
    bytes_per_param = 2  # bf16
    peak_memory = (
        # activations
        (
            args.batch_size
            * seq_length
            * args.d_model
            * args.num_layers
            * bytes_per_param
            * 2
        )
        + (params + other_params * bytes_per_param)  # params
        + (params + other_params * bytes_per_param * 2)  # optimizer
    ) / (1024**3)  # GB

    steps = int(tokens / seq_length / args.batch_size)

    print(
        f"Config: d={args.d_model} L={args.num_layers} H={args.num_heads} bs={args.batch_size} lr={args.lr} | "
        f"Compute: {args.train_flops:.1e} FLOPs | "
        f"Model: N={params / 1e6:.1f}M D={tokens / 1e6:.1f}M ({tokens / params:.1f}:1) | "
        f"Resources: {other_params / 1e6:.1f}M extra, {peak_memory:.1f}GB, {steps:,} steps"
    )


def generate_configs_given_C():
    args = get_parser_args()
    valid_configs = []
    valid_d_model = [128, 256, 512, 768, 1024]
    for d_model in valid_d_model:  # 6 values
        for num_layers in range(2, 16 + 1, 2):
            N = estimate_params(d_model, num_layers)
            D = int(args.train_flops / (6 * N))
            ratio = D / N  # 75% to 97.5%
            if ratio < 3 or ratio > 40:
                continue
            print(
                f"  config d={d_model}, L={num_layers}: N={N / 1e6:.2f}M, D={D / 1e6:.2f}M tokens, ratio (N:D)=1:{ratio:.1f}"
            )
            for head_size in [64, 128]:  # no model uses smth different
                num_heads = d_model // head_size
                if num_heads == 1:
                    continue

                for lr in [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]:  # range 1e-4, 1e-3
                    for batch_size in [128, 256]:  # pre determined
                        valid_configs.append(
                            {
                                "N": N,
                                "D": D,
                                "ratio(N:D)": f"1:{ratio:.1f}",
                                "d_model": d_model,
                                "num_layers": num_layers,
                                "num_heads": num_heads,
                                "head_size": head_size,
                                "lr": lr,
                                "batch_size": batch_size,
                            }
                        )

    # if you account for full exploration of all sizes you are fucked
    # do you really need to test multiple widths, random ones, not really, can do for really small
    # I can test 1e13 5e13 1e14 5e14 1e15, with full grid search
    # then reduce the num of configs given the learnings / scaling laws pre-trace from this models
    # another strategy is using e.g. 2 lr's, then tune hard on best at each flop ratio
    # after initial run < 10%, limit search space to go 5e15, 1e16, and maybe 5e16.

    total_budget = 2e18
    total_flops = args.train_flops * len(valid_configs)
    used_flops_percentage = total_flops * 100 / total_budget
    print(
        f"Found {len(valid_configs)} configs, "
        f"total FLOPs: {total_flops:.1e} ~ "
        f"{used_flops_percentage:.3f}%"
    )

    flops_str = f"{int(args.train_flops):.0e}".replace("+00", "")
    filename = f".configs/config_{flops_str}.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "configs": valid_configs,
                "total_configs": len(valid_configs),
                "total_flops": f"{(total_flops):.1e}",
                "total_flops_percentage": used_flops_percentage,
                "unique_N:D_ratios": len(set(cfg["ratio(N:D)"] for cfg in valid_configs))
            },
            f,
            indent=2,
        )
    print(f"Saved {len(valid_configs)} configs to {filename}")

    # some scaling laws extrapolate from a 1:4k ratio if my max model is somewhere around 500M params,
    # means I could check 1M, 100k param token, check this, cause why not


if __name__ == "__main__":
    # main()
    # sanity_check()
    generate_configs_given_C()
