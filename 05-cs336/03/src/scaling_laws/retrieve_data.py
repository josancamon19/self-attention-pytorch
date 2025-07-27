import wandb
import re
import pandas as pd
from typing import Dict, List, Optional
import json


def parse_run_name(run_name: str) -> Optional[Dict]:
    """Parse run name to extract parameters.

    Format: N{param_count}_D{tokens}_C{flops}_d{d_model}_l{num_layers}_h{num_heads}_b{batch_size}_lr{lr}_r_ignore{rand_i}
    """
    pattern = r"N([\d.]+)_D([\d.]+)_C([\d.e+]+)_d(\d+)_l(\d+)_h(\d+)_b(\d+)_lr([\d.e-]+)_r_ignore(\d+)"
    match = re.match(pattern, run_name)

    if not match:
        print(f"Warning: Could not parse run name: {run_name}")
        return None

    return {
        "N": float(match.group(1)) * 1e6,  # Convert back from millions
        "D": float(match.group(2)) * 1e6,  # Convert back from millions
        "C": float(match.group(3)),
        "d_model": int(match.group(4)),
        "num_layers": int(match.group(5)),
        "num_heads": int(match.group(6)),
        "batch_size": int(match.group(7)),
        "lr": float(match.group(8)),
        "rand_i": int(match.group(9)),
    }


def retrieve_wandb_data(
    project_name: str = "assignment-03-scaling-laws",
) -> pd.DataFrame:
    """Retrieve all runs from wandb project and extract final_loss with parsed parameters."""
    api = wandb.Api()
    runs = api.runs(f"{project_name}")

    data = []
    for run in runs:
        # Get run name and parse parameters
        params = parse_run_name(run.name)
        if params is None:
            continue

        # Get final_loss from summary
        final_loss = run.summary.get("final_loss", None)

        # Get training loss history if final_loss not in summary
        if final_loss is None:
            history = run.history(keys=["train_loss"])
            if not history.empty:
                final_loss = history["train_loss"].dropna().iloc[-1]

        if final_loss is not None:
            # Combine parameters with metrics
            row = params.copy()
            row["final_loss"] = final_loss
            row["run_name"] = run.name
            row["run_id"] = run.id

            # Add any other useful metrics
            row["runtime"] = run.summary.get("_runtime", None)
            row["state"] = run.state

            data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Sort by parameter count for easier analysis
    if not df.empty:
        df = df.sort_values("N")

    return df


def save_data(df: pd.DataFrame, output_path: str = "data/wandb_scaling_data.json"):
    """Save the retrieved data in the same format as exercise_isoflops_curves.json."""
    # Create list of dicts with only the required fields
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "parameters": int(row["N"]),
            "compute_budget": float(row["C"]),  # Ensure it's a float
            "final_loss": row["final_loss"]
        })
    
    # Custom JSON encoder to handle scientific notation
    class ScientificEncoder(json.JSONEncoder):
        def encode(self, obj):
            if isinstance(obj, dict):
                # Handle dict specially to format compute_budget
                items = []
                for k, v in obj.items():
                    if k == "compute_budget" and isinstance(v, (int, float)):
                        # Format as scientific notation
                        formatted = f"{v:.0e}"
                        # Clean up the format to match exercise file (e.g., 6e+18)
                        formatted = formatted.replace('e+0', 'e+').replace('e-0', 'e-')
                        items.append(f'    "{k}": {formatted}')
                    else:
                        items.append(f'    "{k}": {json.dumps(v)}')
                return "{\n" + ",\n".join(items) + "\n  }"
            elif isinstance(obj, list):
                # Handle list of dicts
                return "[\n" + ",\n".join(self.encode(item) for item in obj) + "\n]"
            return super().encode(obj)
    
    # Save as JSON with custom encoder
    with open(output_path, 'w') as f:
        f.write(ScientificEncoder().encode(data_list))
    
    print(f"Saved {len(data_list)} runs to {output_path}")


def main():
    """Main function to retrieve and save wandb data."""
    print("Retrieving data from wandb...")

    # Retrieve data
    df = retrieve_wandb_data()

    if df.empty:
        print("No valid runs found!")
        return

    # Display summary
    print(f"\nRetrieved {len(df)} runs")
    print(f"Parameter range: {df['N'].min():.0f} - {df['N'].max():.0f}")
    print(f"Token range: {df['D'].min():.0f} - {df['D'].max():.0f}")
    print(f"Loss range: {df['final_loss'].min():.4f} - {df['final_loss'].max():.4f}")

    # Save to JSON in data directory
    save_data(df, "data/wandb_scaling_data.json")

    # Print sample
    print("\nSample data:")
    print(df[["N", "D", "C", "d_model", "num_layers", "final_loss"]].head())


if __name__ == "__main__":
    main()
