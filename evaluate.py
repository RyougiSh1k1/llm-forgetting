"""Evaluation utilities for catastrophic forgetting experiments."""

import torch
import numpy as np
from typing import Dict, Tuple
import json
import os
from data_loader import evaluate_accuracy


def evaluate_both_tasks(
    model,
    task1_loader,
    task2_loader,
    device,
    task1_name: str = "Task 1",
    task2_name: str = "Task 2"
) -> Dict[str, float]:
    """
    Evaluate model on both tasks.

    Args:
        model: Model to evaluate
        task1_loader: DataLoader for task 1
        task2_loader: DataLoader for task 2
        device: Device
        task1_name: Name of task 1
        task2_name: Name of task 2

    Returns:
        Dictionary with accuracies
    """
    print(f"\nEvaluating on {task1_name}...")
    task1_acc = evaluate_accuracy(model, task1_loader, device)

    print(f"Evaluating on {task2_name}...")
    task2_acc = evaluate_accuracy(model, task2_loader, device)

    results = {
        f"{task1_name}_accuracy": task1_acc,
        f"{task2_name}_accuracy": task2_acc
    }

    print(f"\nResults:")
    print(f"  {task1_name} Accuracy: {task1_acc:.4f}")
    print(f"  {task2_name} Accuracy: {task2_acc:.4f}")

    return results


def compute_forgetting_metrics(
    acc_task1_before: float,
    acc_task1_after: float,
    acc_task2_after: float
) -> Dict[str, float]:
    """
    Compute catastrophic forgetting metrics.

    Args:
        acc_task1_before: Task 1 accuracy before training on Task 2
        acc_task1_after: Task 1 accuracy after training on Task 2
        acc_task2_after: Task 2 accuracy after training

    Returns:
        Dictionary with forgetting metrics
    """
    # Absolute forgetting
    absolute_forgetting = acc_task1_before - acc_task1_after

    # Relative forgetting (percentage)
    if acc_task1_before > 0:
        relative_forgetting = (absolute_forgetting / acc_task1_before) * 100
    else:
        relative_forgetting = 0.0

    # Backward transfer (negative = forgetting)
    backward_transfer = acc_task1_after - acc_task1_before

    metrics = {
        "task1_acc_before": acc_task1_before,
        "task1_acc_after": acc_task1_after,
        "task2_acc_after": acc_task2_after,
        "absolute_forgetting": absolute_forgetting,
        "relative_forgetting_pct": relative_forgetting,
        "backward_transfer": backward_transfer,
        "avg_accuracy": (acc_task1_after + acc_task2_after) / 2
    }

    return metrics


def save_results(results: Dict, save_path: str):
    """
    Save evaluation results to JSON.

    Args:
        results: Results dictionary
        save_path: Path to save file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results = convert_types(results)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_path}")


def print_forgetting_report(metrics: Dict):
    """
    Print a formatted catastrophic forgetting report.

    Args:
        metrics: Forgetting metrics dictionary
    """
    print("\n" + "=" * 60)
    print("CATASTROPHIC FORGETTING ANALYSIS")
    print("=" * 60)

    print(f"\nTask 1 Performance:")
    print(f"  Before Task 2 training: {metrics['task1_acc_before']:.4f}")
    print(f"  After Task 2 training:  {metrics['task1_acc_after']:.4f}")
    print(f"  Change: {metrics['backward_transfer']:+.4f}")

    print(f"\nTask 2 Performance:")
    print(f"  After training: {metrics['task2_acc_after']:.4f}")

    print(f"\nForgetting Metrics:")
    print(f"  Absolute Forgetting: {metrics['absolute_forgetting']:.4f}")
    print(f"  Relative Forgetting: {metrics['relative_forgetting_pct']:.2f}%")
    print(f"  Backward Transfer:   {metrics['backward_transfer']:+.4f}")

    print(f"\nOverall:")
    print(f"  Average Accuracy: {metrics['avg_accuracy']:.4f}")

    print("=" * 60 + "\n")


def compare_landscapes(
    landscape1_metrics: Dict,
    landscape2_metrics: Dict,
    name1: str = "After Task 1",
    name2: str = "After Task 2"
):
    """
    Compare two loss landscapes and print analysis.

    Args:
        landscape1_metrics: Metrics from first landscape
        landscape2_metrics: Metrics from second landscape
        name1: Name of first landscape
        name2: Name of second landscape
    """
    print("\n" + "=" * 60)
    print("LOSS LANDSCAPE COMPARISON (Efficient Metrics)")
    print("=" * 60)

    # Filter out list values (eigenvalues) for comparison
    scalar_metrics1 = {k: v for k, v in landscape1_metrics.items() if not isinstance(v, list) and v is not None}
    scalar_metrics2 = {k: v for k, v in landscape2_metrics.items() if not isinstance(v, list) and v is not None}

    print(f"\n{name1}:")
    for key, value in scalar_metrics1.items():
        print(f"  {key}: {value:.6f}")

    print(f"\n{name2}:")
    for key, value in scalar_metrics2.items():
        print(f"  {key}: {value:.6f}")

    print("\nChanges (Task 2 - Task 1):")
    for key in scalar_metrics1:
        if key in scalar_metrics2:
            change = scalar_metrics2[key] - scalar_metrics1[key]
            pct_change = (change / scalar_metrics1[key] * 100) if scalar_metrics1[key] != 0 else 0
            print(f"  Δ{key}: {change:+.6f} ({pct_change:+.2f}%)")

    # Interpret flatness based on lambda_max
    print("\nSharpness Analysis (based on Mirzadeh et al. 2020):")
    if "lambda_max" in scalar_metrics1 and "lambda_max" in scalar_metrics2:
        if scalar_metrics2["lambda_max"] < scalar_metrics1["lambda_max"]:
            print("  → Loss landscape became FLATTER after Task 2 training (λ_max decreased)")
            print("     This suggests REDUCED forgetting (wider minima)")
        else:
            print("  → Loss landscape became SHARPER after Task 2 training (λ_max increased)")
            print("     This suggests INCREASED forgetting (sharper minima)")

    if "displacement" in scalar_metrics2 and scalar_metrics2["displacement"] is not None:
        print(f"\nParameter Displacement:")
        print(f"  ||Δw|| = {scalar_metrics2['displacement']:.2f}")
        print(f"  Bound on forgetting: F₁ ≤ (1/2) λ_max ||Δw||² = {0.5 * scalar_metrics2.get('lambda_max', 0) * scalar_metrics2['displacement']**2:.4f}")

    print("=" * 60 + "\n")
