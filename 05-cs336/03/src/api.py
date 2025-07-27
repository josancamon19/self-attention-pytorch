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


import json
import requests
import os
from datetime import datetime
from configs import get_parser_args, estimate_params

# Ensure the configs directory exists
BASE_URL = "http://hyperturing.stanford.edu:8000"
api_key = os.getenv("API_KEY", "123")


def get_total_flops():
    response = requests.get(f"{BASE_URL}/total_flops_used", {"api_key": api_key})
    return response.json()["total_flops_used"]


def get_previous_runs():
    response = requests.get(f"{BASE_URL}/previous_runs", {"api_key": api_key})
    return response.json()


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


if __name__ == "__main__":
    main()
