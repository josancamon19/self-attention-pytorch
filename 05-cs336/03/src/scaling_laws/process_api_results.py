#!/usr/bin/env python3
"""
Script to process runs.jsonl and create exercise_isoflops_curves.json
Groups results by estimated_params and config.train_flops, finding the best loss for each group.
"""

import json
from collections import defaultdict
from pathlib import Path
from process_wandb_results import ScientificEncoder


def process_runs_to_isoflops():
    """Process runs.jsonl and create isoflops curves data."""

    # Read the JSONL file
    runs_file = Path("results/runs.jsonl")

    if not runs_file.exists():
        raise FileNotFoundError(f"{runs_file} not found")

    # Group by (estimated_params, train_flops) and collect losses and configs
    groups = defaultdict(list)

    with open(runs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data = json.loads(line)
                key = (data["estimated_params"], data["config"]["train_flops"])

                # Add compute optimal ratio C=6ND to config
                config = data["config"].copy()
                config["N"] = data["N"]
                config["D"] = data["D"]
                config["ratio_N_D"] = data["ratio_N_D"]
                del config['train_flops']

                groups[key].append((data["loss"], config))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {e}")
                continue

    # Create the output structure
    results = []

    for (params, compute_budget), loss_config_pairs in groups.items():
        # Find the minimum loss for this group (best performance) and its config
        best_loss, best_config = min(loss_config_pairs, key=lambda x: x[0])

        results.append(
            {
                "parameters": params,
                "compute_budget": compute_budget,
                "final_loss": best_loss,
                "config": best_config,
            }
        )

    # Sort by compute_budget, then by parameters for consistent ordering
    results.sort(key=lambda x: (x["compute_budget"], x["parameters"]))

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Write to output file
    output_file = data_dir / "api_isoflops_curves.json"
    with open(output_file, "w") as f:
        f.write(ScientificEncoder().encode(results))

    print(f"Created {output_file} with {len(results)} data points")
    print(f"Compute budgets: {sorted(set(r['compute_budget'] for r in results))}")
    print(
        f"Parameter counts per budget: {len(set(r['parameters'] for r in results if r['compute_budget'] == min(r['compute_budget'] for r in results)))}"
    )


if __name__ == "__main__":
    process_runs_to_isoflops()
