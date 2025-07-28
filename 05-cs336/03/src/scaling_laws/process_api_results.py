#!/usr/bin/env python3
"""
Script to process runs.jsonl and create exercise_isoflops_curves.json
Groups results by estimated_params and config.train_flops, finding the best loss for each group.
"""

import json
from collections import defaultdict
from pathlib import Path


def process_runs_to_isoflops():
    """Process runs.jsonl and create isoflops curves data."""

    # Read the JSONL file
    runs_file = Path("results/runs.jsonl")

    if not runs_file.exists():
        raise FileNotFoundError(f"{runs_file} not found")

    # Group by (estimated_params, train_flops) and collect losses
    groups = defaultdict(list)

    with open(runs_file, "r") as f:
        for line in f:
            data = json.loads(line)
            key = (data["estimated_params"], data["config"]["train_flops"])
            groups[key].append(data["loss"])

    # Create the output structure
    results = []

    for (params, compute_budget), losses in groups.items():
        # Find the minimum loss for this group (best performance)
        best_loss = min(losses)

        results.append(
            {
                "parameters": params,
                "compute_budget": compute_budget,
                "final_loss": best_loss,
            }
        )

    # Sort by compute_budget, then by parameters for consistent ordering
    results.sort(key=lambda x: (x["compute_budget"], x["parameters"]))

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Write to output file
    output_file = data_dir / "exercise_isoflops_curves.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Created {output_file} with {len(results)} data points")
    print(f"Compute budgets: {sorted(set(r['compute_budget'] for r in results))}")
    print(
        f"Parameter counts per budget: {len(set(r['parameters'] for r in results if r['compute_budget'] == min(r['compute_budget'] for r in results)))}"
    )


if __name__ == "__main__":
    process_runs_to_isoflops()
