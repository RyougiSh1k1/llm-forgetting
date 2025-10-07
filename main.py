"""Main experiment script for examining loss landscape flatness and catastrophic forgetting."""
import numpy as np
import torch
import os
import argparse
from datetime import datetime

from config import ModelConfig, LoRAConfig, TrainingConfig, TRACEConfig, LossLandscapeConfig
from data_loader import TRACEDataset
from train import setup_model_and_tokenizer, train_on_task, save_checkpoint, load_checkpoint
from loss_landscape import (
    analyze_loss_landscape,
    analyze_loss_landscape_efficient,
    analyze_loss_landscape_lora,
    get_model_parameters,
    get_lora_parameters,
    plot_forgetting_vs_sharpness,
    plot_eigenvalue_comparison
)
from evaluate import (
    evaluate_both_tasks,
    compute_forgetting_metrics,
    save_results,
    print_forgetting_report,
    compare_landscapes
)


def main(args):
    """Run the complete experiment."""

    # Setup configurations
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    lora_config.r = args.lora_r
    lora_config.lora_alpha = args.lora_r * 2  # Typically 2*r
    training_config = TrainingConfig()
    training_config.batch_size = args.batch_size
    training_config.num_epochs = args.num_epochs
    training_config.learning_rate = args.learning_rate
    trace_config = TRACEConfig()
    trace_config.task1 = args.task1
    trace_config.task2 = args.task2
    landscape_config = LossLandscapeConfig()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(training_config.output_dir, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TRACE BENCHMARK: Loss Landscape Flatness & Catastrophic Forgetting")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_config.model_name}")
    print(f"  Task 1: {trace_config.task1}")
    print(f"  Task 2: {trace_config.task2}")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Epochs per task: {training_config.num_epochs}")
    print(f"  LoRA r: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ========== PHASE 1: Setup Model ==========
    print("\n" + "="*70)
    print("PHASE 1: Setting up model and data")
    print("="*70 + "\n")

    model, tokenizer = setup_model_and_tokenizer(
        model_config.model_name,
        lora_config.__dict__
    )

    # Create TRACE datasets with configured sample sizes
    task1_train_dataset = TRACEDataset(
        trace_config.task1, split="train", tokenizer=tokenizer, max_length=model_config.max_length,
        train_samples=trace_config.train_samples, test_samples=trace_config.test_samples
    )
    task1_test_dataset = TRACEDataset(
        trace_config.task1, split="test", tokenizer=tokenizer, max_length=model_config.max_length,
        train_samples=trace_config.train_samples, test_samples=trace_config.test_samples
    )
    task2_train_dataset = TRACEDataset(
        trace_config.task2, split="train", tokenizer=tokenizer, max_length=model_config.max_length,
        train_samples=trace_config.train_samples, test_samples=trace_config.test_samples
    )
    task2_test_dataset = TRACEDataset(
        trace_config.task2, split="test", tokenizer=tokenizer, max_length=model_config.max_length,
        train_samples=trace_config.train_samples, test_samples=trace_config.test_samples
    )

    from torch.utils.data import DataLoader
    task1_train_loader = DataLoader(task1_train_dataset, batch_size=training_config.batch_size, shuffle=True)
    task1_test_loader = DataLoader(task1_test_dataset, batch_size=training_config.batch_size, shuffle=False)
    task2_train_loader = DataLoader(task2_train_dataset, batch_size=training_config.batch_size, shuffle=True)
    task2_test_loader = DataLoader(task2_test_dataset, batch_size=training_config.batch_size, shuffle=False)

    print(f"Dataset sizes:")
    print(f"  Task 1 ({trace_config.task1}): {len(task1_train_dataset)} train, {len(task1_test_dataset)} test")
    print(f"  Task 2 ({trace_config.task2}): {len(task2_train_dataset)} train, {len(task2_test_dataset)} test")

    # ========== PHASE 2: Train on Task 1 ==========
    print("\n" + "="*70)
    print("PHASE 2: Training on Task 1")
    print("="*70 + "\n")

    task1_output_dir = os.path.join(output_dir, "task1")
    model = train_on_task(
        model,
        tokenizer,
        task1_train_dataset,
        task1_test_dataset,
        task1_output_dir,
        training_config.__dict__,
        trace_config.task1
    )

    # Save initial parameters (for displacement tracking)
    initial_params = get_model_parameters(model, device)
    initial_lora_params = get_lora_parameters(model, device)

    # Save checkpoint after Task 1
    checkpoint1_path = os.path.join(output_dir, "checkpoint_after_task1")
    save_checkpoint(model, tokenizer, checkpoint1_path)

    # Get parameters after Task 1
    params_after_task1 = get_model_parameters(model, device)
    lora_params_after_task1 = get_lora_parameters(model, device)

    # ========== PHASE 3: Evaluate after Task 1 ==========
    print("\n" + "="*70)
    print("PHASE 3: Evaluation after Task 1")
    print("="*70 + "\n")

    results_after_task1 = evaluate_both_tasks(
        model, task1_test_loader, task2_test_loader, device,
        trace_config.task1, trace_config.task2
    )

    # ========== PHASE 4: Analyze Loss Landscape after Task 1 (LoRA Subspace) ==========
    print("\n" + "="*70)
    print("PHASE 4: Loss Landscape Analysis after Task 1 (LoRA Subspace)")
    print("="*70 + "\n")

    landscape1_task1 = analyze_loss_landscape_lora(
        model, task1_test_loader, device,
        output_dir, f"task1_{trace_config.task1}",
        initial_lora_params=initial_lora_params,
        num_eigenvalues=20,
        compute_hessian=True
    )

    landscape1_task2 = analyze_loss_landscape_lora(
        model, task2_test_loader, device,
        output_dir, f"task1_on_{trace_config.task2}",
        initial_lora_params=initial_lora_params,
        num_eigenvalues=20,
        compute_hessian=True
    )

    # ========== PHASE 5: Train on Task 2 ==========
    print("\n" + "="*70)
    print("PHASE 5: Training on Task 2")
    print("="*70 + "\n")

    task2_output_dir = os.path.join(output_dir, "task2")
    model = train_on_task(
        model,
        tokenizer,
        task2_train_dataset,
        task2_test_dataset,
        task2_output_dir,
        training_config.__dict__,
        trace_config.task2
    )

    # Save checkpoint after Task 2
    checkpoint2_path = os.path.join(output_dir, "checkpoint_after_task2")
    save_checkpoint(model, tokenizer, checkpoint2_path)

    # Get parameters after Task 2
    params_after_task2 = get_model_parameters(model, device)
    lora_params_after_task2 = get_lora_parameters(model, device)

    # ========== PHASE 6: Evaluate after Task 2 ==========
    print("\n" + "="*70)
    print("PHASE 6: Evaluation after Task 2")
    print("="*70 + "\n")

    results_after_task2 = evaluate_both_tasks(
        model, task1_test_loader, task2_test_loader, device,
        trace_config.task1, trace_config.task2
    )

    # ========== PHASE 7: Analyze Loss Landscape after Task 2 (LoRA Subspace) ==========
    print("\n" + "="*70)
    print("PHASE 7: Loss Landscape Analysis after Task 2 (LoRA Subspace)")
    print("="*70 + "\n")

    landscape2_task1 = analyze_loss_landscape_lora(
        model, task1_test_loader, device,
        output_dir, f"task2_on_{trace_config.task1}",
        initial_lora_params=lora_params_after_task1,
        num_eigenvalues=20,
        compute_hessian=True
    )

    landscape2_task2 = analyze_loss_landscape_lora(
        model, task2_test_loader, device,
        output_dir, f"task2_{trace_config.task2}",
        initial_lora_params=lora_params_after_task1,
        num_eigenvalues=20,
        compute_hessian=True
    )

    # ========== PHASE 8: Compute Forgetting Metrics ==========
    print("\n" + "="*70)
    print("PHASE 8: Catastrophic Forgetting Analysis")
    print("="*70 + "\n")

    forgetting_metrics = compute_forgetting_metrics(
        results_after_task1[f"{trace_config.task1}_accuracy"],
        results_after_task2[f"{trace_config.task1}_accuracy"],
        results_after_task2[f"{trace_config.task2}_accuracy"]
    )

    print_forgetting_report(forgetting_metrics)

    # ========== PHASE 9: Compare Loss Landscapes ==========
    print("\n" + "="*70)
    print("PHASE 9: Loss Landscape Comparison")
    print("="*70 + "\n")

    print(f"\nFor Task 1 ({trace_config.task1}) data:")
    compare_landscapes(
        landscape1_task1["metrics"],
        landscape2_task1["metrics"],
        "After Task 1 Training",
        "After Task 2 Training"
    )

    # ========== PHASE 10: Generate Figure 2(c)/(d) style plots ==========
    print("\n" + "="*70)
    print("PHASE 10: Generating Forgetting Analysis Plots")
    print("="*70 + "\n")

    # Prepare data for forgetting vs sharpness plot (Figure 2c style)
    # Only include the forgetting point after Task 2 (not the baseline with 0 forgetting)
    forgetting_data = {
        f"Task2_forgetting_{trace_config.task1}": {
            'lambda_max': landscape2_task1["metrics"].get("lambda_max_lora", 0.0),
            'displacement': landscape2_task1["metrics"].get("lora_displacement", 0.0),
            'forgetting': forgetting_metrics.get("absolute_forgetting", 0.0),
            'task_name': f"After Task2 on {trace_config.task1}"
        }
    }

    # Create forgetting vs sharpness plot
    plot_forgetting_vs_sharpness(forgetting_data, output_dir, "forgetting_vs_sharpness_lora.png")

    # Prepare eigenvalue comparison data (Figure 2d style)
    eigenvalue_data = {
        f"Task1 - {trace_config.task1}": np.array(landscape1_task1["metrics"].get("eigenvalues_lora", [])),
        f"Task2 - {trace_config.task2}": np.array(landscape2_task2["metrics"].get("eigenvalues_lora", [])),
        f"After Task2 on {trace_config.task1}": np.array(landscape2_task1["metrics"].get("eigenvalues_lora", []))
    }

    # Create eigenvalue comparison plot
    plot_eigenvalue_comparison(eigenvalue_data, output_dir, "eigenvalue_comparison_lora.png")

    # ========== PHASE 11: Save All Results ==========
    print("\n" + "="*70)
    print("PHASE 11: Saving Results")
    print("="*70 + "\n")

    final_results = {
        "config": {
            "model": model_config.model_name,
            "task1": trace_config.task1,
            "task2": trace_config.task2,
            "batch_size": training_config.batch_size,
            "num_epochs": training_config.num_epochs,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha
        },
        "accuracy_after_task1": results_after_task1,
        "accuracy_after_task2": results_after_task2,
        "forgetting_metrics": forgetting_metrics,
        "landscape_task1_after_task1": landscape1_task1["metrics"],
        "landscape_task1_after_task2": landscape2_task1["metrics"],
        "landscape_task2_after_task1": landscape1_task2["metrics"],
        "landscape_task2_after_task2": landscape2_task2["metrics"],
    }

    results_path = os.path.join(output_dir, "results.json")
    save_results(final_results, results_path)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - Checkpoints: checkpoint_after_task1/, checkpoint_after_task2/")
    print(f"  - LoRA Hessian Eigenvalues: eigenvalues_*_lora.png")
    print(f"  - Forgetting Analysis (Fig 2c): forgetting_vs_sharpness_lora.png")
    print(f"  - Eigenvalue Comparison (Fig 2d): eigenvalue_comparison_lora.png")
    print(f"  - Metrics: results.json")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE Benchmark - Catastrophic Forgetting Experiment")
    parser.add_argument("--task1", type=str, default="scienceqa", help="First TRACE task")
    parser.add_argument("--task2", type=str, default="fomc", help="Second TRACE task")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-epochs", "--num_epochs", type=int, default=10, help="Number of epochs per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora-r", type=int, default=2, help="LoRA rank")
    parser.add_argument("--learning-rate", "--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()
    main(args)
