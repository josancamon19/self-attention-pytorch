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


def query_loss(
    d_model,
    num_layers,
    num_heads,
    batch_size,
    lr,
    train_flops,
):
    """Query the API for loss given hyperparameters"""
    # querying this endpoint with a previously queried doesn't discount FLOPs, tho save them and assert when calling
    # endpoint returns (loss, total_flops_used (new total))

    payload = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_flops": int(train_flops),
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

    print(f"Processing {len(configs)} configurations from {config_file_path} ")

    for i, config in enumerate(configs):
        print(f"\nProcessing config {i + 1}/{len(configs)}...")
        print(float(config["C"]), type(float(config["C"])))
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
        print(result)

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
    # main()
    # process_config_file(".configs/config_1e+13.json")
    # process_config_file(".configs/config_3e+13.json")
    # process_config_file(".configs/config_6e+13.json")
    # process_config_file(".configs/config_1e+14.json")
    # process_config_file(".configs/config_3e+14.json")
    # process_config_file(".configs/config_6e+14.json")
    # process_config_file(".configs/config_1e+15.json")
    # I fucked up, huhhhh, forgot should've not ran this
    # process_config_file(".configs/config_3e+15.json")
    # from here, reduced batch size to 128, lr 1e-3, running 6e15
    # process_config_file(".configs/config_6e+15.json")
    # ratio width depth, seem 256 and 10+ layers is overperforming.
    # will check for 1e16 next, how many flops by filtering layers >= 4, have used 32.5%
    # running 1e16 with 4,16 layers, would reduce 8% flops, 60% left.
    # 3e16 at this same search space is 30% gotta reduce to 15, would have to reduce ()
    # process_config_file(".configs/config_1e+16.json")
    # gonna do from 512 now for 3e16, + >= 6 layers
    # ratio min 4, nothing better than 1:4 has won
    # wish I could reduce search space for hed size, but don't feel ready yet. is not clear.
    # process_config_file(".configs/config_3e+16.json") # 9% flops, totaling 50%
    # Fuck, coefficients have changed so drastically, initial search 0.3/0.7, then after 3e15 0.35, now 0.46 wtf
    # also fuck, missed <3 in ratios, kept the 3, instead of 4, 3% flops wasted
    # some small guesstimate from now on, 128 head size is better, doesn't seem fully clear, tho is a more general assumption
    # 6e16, going for 18% of flops
    # process_config_file(".configs/config_6e+16.json") # 9% flops, totaling 50%
    # adjusting ratios to min 1:8 maybe 1:10, and less than 1:50, a very small model doesn't seem to be able to take
    # this part is scary, is it, limiting ratios to 1:10, the only at 1e17 is with 512 d_
    # If i remove 512 as options, and limit rations to 1:10, 1e17 would have nothing to test
    # so should I reduce the ratio to < 10, I mean at bigger scaler clear will not be 512 > 768, no way
    # let's double check ratios ...
    # yea no ratio < 10 has won in it's flop count.
    # skipping 1e17 then, for 3e17, got:
    #   config d=768, L=6: N=42.47M, D=1177.38M tokens, ratio (N:D)=1:27.7
    #   config d=768, L=8: N=56.62M, D=883.03M tokens, ratio (N:D)=1:15.6
    # process_config_file(".configs/config_3e+17.json")
    # oh fuck, bad accounting, couldn't run 2nd opt .-.
    # 2.95e+17 left, needed 3e17, fuck, well gotta reduce the ratio a bit so it fits hahah
    # "D": 883031774,
    #   "C": "3.0e+17",
    #   "ratio(N:D)": "1:15.6",
    # adjusted to # "D": 873031774, # "ratio(N:D)": "1:15",
    # oh this is annoying, it will not run it, cause I don't specify tokens, so F it, agh, anw let's plot what we have
    # I guess this is left, so let's see what to do
    # Optimal config for 1.00e+19 FLOPs:
    #   Params: 2.27e+08
    #   Tokens: 7.35e+09
    #   Loss:   2.6107
    #   Ratio:  32.4 tokens/param

    # Scaling laws:
    #   N_opt = 1.17e-01 × C^0.489
    #   D_opt = 1.43e+00 × C^0.511
    #   L_opt = 2.99e+02 × C^-0.108
    #   α + β = 1.000 (should ≈ 1.0)

    # let's check what d_model, num_layers work
    print(2.27e08)  # target
    print(estimate_params(1024, 18))
    print(estimate_params(768, 24))
    # ohhhh didn't test layers > 16, and assignment says up to 24!! ofc, cause at 227M target, 1024,16, is still at 200M
    # but actually when generating configs now, the only d_model that works is 1024, and the 18 layers, hit perfectly on ratio and params expected
    # answer
    # lr 1e-3
    # batch size 128
    # d_model 1024
    # num_layers 18
    # num_heads 8
