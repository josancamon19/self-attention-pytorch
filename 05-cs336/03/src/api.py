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
from dotenv import load_dotenv

load_dotenv()
# Ensure the configs directory exists
BASE_URL = "http://hyperturing.stanford.edu:8000"
api_key = os.getenv("API_KEY", "123")


def get_total_flops():
    response = requests.get(f"{BASE_URL}/total_flops_used", {"api_key": api_key})
    return response.json()


def get_previous_runs():
    response = requests.get(f"{BASE_URL}/previous_runs", {"api_key": api_key})
    return response.json()["previous_runs"]


def query_loss(d_model, num_layers, num_heads, batch_size, lr, train_flops):
    """Query the API for loss given hyperparameters"""
    # querying this endpoint with a previously queried doesn't discount FLOPs, tho save them and assert when calling
    # endpoint returns (loss, total_flops_used (new total))

    payload = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_flops": train_flops,
        "api_key": api_key,
    }

    response = requests.get(f"{BASE_URL}/loss", payload)
    result = response.json()

    # Add estimated params and timestamp
    result["estimated_params"] = estimate_params(d_model, num_layers)
    result["timestamp"] = datetime.now().isoformat()
    del payload["api_key"]
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
        print(runs[0])

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


def process_config_file(config_file_path):
    """Process a config file and call query_loss for each configuration"""
    # Load the config file
    with open(config_file_path, "r") as f:
        data = json.load(f)

    configs = data.get("configs", [])
    results = []

    print(f"Processing {len(configs)} configurations from {config_file_path}")

    for i, config in enumerate(configs):
        print(f"\nProcessing config {i + 1}/{len(configs)}...")
        result = query_loss(
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            train_flops=float(config["C"]),
        )

        result["config_index"] = i
        result["N"] = config["N"]
        result["D"] = config["D"]
        result["ratio_N_D"] = config["ratio(N:D)"]

        results.append(result)

        # Save each run
        save_run(result)
        # print(result)

        # Print results
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Total FLOPs used: {result['total_flops_used']:.2e}")
        print(f"  Estimated params: {result['estimated_params']:.2e}")

    print(f"\nProcessed all {len(configs)} configurations")
    print(f"Final total FLOPs used: {results[-1]['total_flops_used']:.2e}")

    return results


if __name__ == "__main__":
    # print(get_total_flops())
    # print(get_previous_runs())
    main()
    # process_config_file(".configs/config_1e+13.json")
    # process_config_file(".configs/config_3e+13.json")
    # process_config_file(".configs/config_6e+13.json")
    # process_config_file(".configs/config_1e+14.json")
    # process_config_file(".configs/config_3e+14.json")
    # process_config_file(".configs/config_6e+14.json")
    # process_config_file(".configs/config_1e+15.json")
    # I fucked up, huhhhh, forgot should've not ran this
    # process_config_file(".configs/config_3e+15.json") 
