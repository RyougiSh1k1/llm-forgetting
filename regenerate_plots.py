"""Regenerate plots from existing results.json file."""
import json
import os
import sys
import numpy as np

from loss_landscape import plot_forgetting_vs_sharpness, plot_eigenvalue_comparison

def regenerate_plots(results_file: str):
    """
    Regenerate plots from results.json file.

    Args:
        results_file: Path to results.json file
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    output_dir = os.path.dirname(results_file)

    print(f"Regenerating plots from: {results_file}")
    print(f"Output directory: {output_dir}")

    # Extract task names from config
    task1 = results["config"]["task1"]
    task2 = results["config"]["task2"]

    # Prepare forgetting data
    # Only include the forgetting point after Task 2 (not the baseline with 0 forgetting)
    forgetting_data = {
        f"Task2_forgetting_{task1}": {
            'lambda_max': results["landscape_task1_after_task2"].get("lambda_max_lora", 0.0),
            'displacement': results["landscape_task1_after_task2"].get("lora_displacement", 0.0),
            'forgetting': results["forgetting_metrics"].get("absolute_forgetting", 0.0),
            'task_name': f"After Task2 on {task1}"
        }
    }

    print("\nForgetting data:")
    for key, data in forgetting_data.items():
        print(f"  {key}:")
        print(f"    λ_max: {data['lambda_max']:.2f}")
        print(f"    displacement: {data['displacement']:.2f}")
        print(f"    forgetting: {data['forgetting']:.4f}")

    # Create forgetting vs sharpness plot
    print("\nGenerating forgetting vs sharpness plot...")
    plot_forgetting_vs_sharpness(forgetting_data, output_dir, "forgetting_vs_sharpness_lora.png")

    # Prepare eigenvalue data
    eigenvalue_data = {
        f"Task1 - {task1}": np.array(results["landscape_task1_after_task1"].get("eigenvalues_lora", [])),
        f"Task2 - {task2}": np.array(results["landscape_task2_after_task2"].get("eigenvalues_lora", [])),
        f"After Task2 on {task1}": np.array(results["landscape_task1_after_task2"].get("eigenvalues_lora", []))
    }

    print("\nGenerating eigenvalue comparison plot...")
    plot_eigenvalue_comparison(eigenvalue_data, output_dir, "eigenvalue_comparison_lora.png")

    print("\n✓ Plots regenerated successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Default to latest experiment
        results_file = "/home/exouser/llm-forgetting/outputs/experiment_20251007_062505/results.json"

    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    regenerate_plots(results_file)
