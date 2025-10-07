"""Run multiple experiments with different configurations to generate data points."""
import os
import subprocess
import json
import numpy as np
from datetime import datetime

# Experiment configurations - vary random seeds to change training sample selection
# Each seed will result in different data shuffling and sample selection
EXPERIMENT_CONFIGS = [
    {"name": "seed_42", "seed": 42, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_123", "seed": 123, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_456", "seed": 456, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_789", "seed": 789, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_999", "seed": 999, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_111", "seed": 111, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_222", "seed": 222, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_333", "seed": 333, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_555", "seed": 555, "lora_r": 2, "lr": 3e-4, "epochs": 10},
    {"name": "seed_777", "seed": 777, "lora_r": 2, "lr": 3e-4, "epochs": 10},
]

def run_experiment(config):
    """
    Run a single experiment with given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Path to results.json file
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: {config['name']}")
    print(f"  Seed: {config['seed']}, LoRA rank: {config['lora_r']}, LR: {config['lr']}, Epochs: {config['epochs']}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        "python", "main.py",
        "--seed", str(config['seed']),
        "--lora-r", str(config['lora_r']),
        "--learning-rate", str(config['lr']),
        "--num-epochs", str(config['epochs']),
        "--batch-size", "1",
        "--task1", "scienceqa",
        "--task2", "fomc"
    ]

    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Experiment {config['name']} failed!")
        print(result.stderr)
        return None

    # Find the output directory (latest experiment_* folder)
    output_base = "./outputs"
    exp_dirs = [d for d in os.listdir(output_base) if d.startswith("experiment_")]
    exp_dirs.sort(reverse=True)  # Latest first
    latest_exp = os.path.join(output_base, exp_dirs[0])
    results_file = os.path.join(latest_exp, "results.json")

    if os.path.exists(results_file):
        # Rename the experiment folder to include config name
        new_exp_dir = os.path.join(output_base, f"experiment_{config['name']}")
        if os.path.exists(new_exp_dir):
            # Add timestamp if directory exists
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_exp_dir = os.path.join(output_base, f"experiment_{config['name']}_{timestamp}")
        os.rename(latest_exp, new_exp_dir)
        results_file = os.path.join(new_exp_dir, "results.json")
        print(f"✓ Experiment completed: {results_file}\n")
        return results_file
    else:
        print(f"ERROR: Results file not found for {config['name']}\n")
        return None

def collect_all_results(results_files):
    """
    Collect all results into a single summary file.

    Args:
        results_files: List of paths to results.json files
    """
    all_data = []

    for results_file in results_files:
        if results_file is None:
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract key metrics
        data_point = {
            'experiment_name': os.path.basename(os.path.dirname(results_file)),
            'config': results['config'],
            'forgetting': results['forgetting_metrics']['absolute_forgetting'],
            'relative_forgetting_pct': results['forgetting_metrics']['relative_forgetting_pct'],
            'lambda_max': results['landscape_task1_after_task2']['lambda_max_lora'],
            'displacement': results['landscape_task1_after_task2']['lora_displacement'],
            'sharpness': 0.5 * results['landscape_task1_after_task2']['lambda_max_lora'] *
                        (results['landscape_task1_after_task2']['lora_displacement'] ** 2),
            'task1_acc_before': results['forgetting_metrics']['task1_acc_before'],
            'task1_acc_after': results['forgetting_metrics']['task1_acc_after'],
            'task2_acc_after': results['forgetting_metrics']['task2_acc_after']
        }
        all_data.append(data_point)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./outputs/all_experiments_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*70}\n")

    # Print summary statistics
    print("Summary Statistics:")
    print(f"  Total experiments: {len(all_data)}")
    forgetting_vals = [d['forgetting'] for d in all_data]
    sharpness_vals = [d['sharpness'] for d in all_data]
    print(f"  Forgetting - Mean: {np.mean(forgetting_vals):.4f}, Std: {np.std(forgetting_vals):.4f}")
    print(f"  Sharpness - Mean: {np.mean(sharpness_vals):.2f}, Std: {np.std(sharpness_vals):.2f}")

    return summary_file, all_data

def main():
    """Run all experiments and collect results."""
    print("="*70)
    print("RUNNING MULTIPLE EXPERIMENTS FOR FORGETTING ANALYSIS")
    print("="*70)
    print(f"\nTotal experiments to run: {len(EXPERIMENT_CONFIGS)}\n")

    results_files = []

    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n[{i}/{len(EXPERIMENT_CONFIGS)}] Starting experiment: {config['name']}")
        results_file = run_experiment(config)
        results_files.append(results_file)

    # Collect all results
    summary_file, all_data = collect_all_results(results_files)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review summary: {summary_file}")
    print(f"  2. Generate combined plots with all data points")
    print(f"  3. Analyze correlation between sharpness and forgetting")

if __name__ == "__main__":
    main()
