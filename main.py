"""Main experiment script for examining loss landscape flatness and catastrophic forgetting."""

import torch
import os
import argparse
from datetime import datetime

from config import ModelConfig, LoRAConfig, TrainingConfig, MMLUConfig, LossLandscapeConfig
from data_loader import create_dataloaders, MMLUDataset
from train import setup_model_and_tokenizer, train_on_task, save_checkpoint, load_checkpoint
from loss_landscape import analyze_loss_landscape
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
    training_config = TrainingConfig()
    training_config.batch_size = args.batch_size
    training_config.num_epochs = args.num_epochs
    mmlu_config = MMLUConfig()
    mmlu_config.task1 = args.task1
    mmlu_config.task2 = args.task2
    landscape_config = LossLandscapeConfig()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(training_config.output_dir, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: Loss Landscape Flatness & Catastrophic Forgetting")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_config.model_name}")
    print(f"  Task 1: {mmlu_config.task1}")
    print(f"  Task 2: {mmlu_config.task2}")
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

    # Create dataloaders
    task1_train_dataset = MMLUDataset(
        mmlu_config.task1, split="dev", tokenizer=tokenizer, max_length=model_config.max_length
    )
    task1_test_dataset = MMLUDataset(
        mmlu_config.task1, split="test", tokenizer=tokenizer, max_length=model_config.max_length
    )
    task2_train_dataset = MMLUDataset(
        mmlu_config.task2, split="dev", tokenizer=tokenizer, max_length=model_config.max_length
    )
    task2_test_dataset = MMLUDataset(
        mmlu_config.task2, split="test", tokenizer=tokenizer, max_length=model_config.max_length
    )

    from torch.utils.data import DataLoader
    task1_train_loader = DataLoader(task1_train_dataset, batch_size=training_config.batch_size, shuffle=True)
    task1_test_loader = DataLoader(task1_test_dataset, batch_size=training_config.batch_size, shuffle=False)
    task2_train_loader = DataLoader(task2_train_dataset, batch_size=training_config.batch_size, shuffle=True)
    task2_test_loader = DataLoader(task2_test_dataset, batch_size=training_config.batch_size, shuffle=False)

    print(f"Dataset sizes:")
    print(f"  Task 1 ({mmlu_config.task1}): {len(task1_train_dataset)} train, {len(task1_test_dataset)} test")
    print(f"  Task 2 ({mmlu_config.task2}): {len(task2_train_dataset)} train, {len(task2_test_dataset)} test")

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
        mmlu_config.task1
    )

    # Save checkpoint after Task 1
    checkpoint1_path = os.path.join(output_dir, "checkpoint_after_task1")
    save_checkpoint(model, tokenizer, checkpoint1_path)

    # ========== PHASE 3: Evaluate after Task 1 ==========
    print("\n" + "="*70)
    print("PHASE 3: Evaluation after Task 1")
    print("="*70 + "\n")

    results_after_task1 = evaluate_both_tasks(
        model, task1_test_loader, task2_test_loader, device,
        mmlu_config.task1, mmlu_config.task2
    )

    # ========== PHASE 4: Analyze Loss Landscape after Task 1 ==========
    print("\n" + "="*70)
    print("PHASE 4: Loss Landscape Analysis after Task 1")
    print("="*70 + "\n")

    landscape1_task1 = analyze_loss_landscape(
        model, task1_test_loader, device,
        output_dir, f"task1_{mmlu_config.task1}",
        distance=landscape_config.distance,
        num_points=landscape_config.num_points,
        seed=landscape_config.random_seed
    )

    landscape1_task2 = analyze_loss_landscape(
        model, task2_test_loader, device,
        output_dir, f"task1_on_{mmlu_config.task2}",
        distance=landscape_config.distance,
        num_points=landscape_config.num_points,
        seed=landscape_config.random_seed
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
        mmlu_config.task2
    )

    # Save checkpoint after Task 2
    checkpoint2_path = os.path.join(output_dir, "checkpoint_after_task2")
    save_checkpoint(model, tokenizer, checkpoint2_path)

    # ========== PHASE 6: Evaluate after Task 2 ==========
    print("\n" + "="*70)
    print("PHASE 6: Evaluation after Task 2")
    print("="*70 + "\n")

    results_after_task2 = evaluate_both_tasks(
        model, task1_test_loader, task2_test_loader, device,
        mmlu_config.task1, mmlu_config.task2
    )

    # ========== PHASE 7: Analyze Loss Landscape after Task 2 ==========
    print("\n" + "="*70)
    print("PHASE 7: Loss Landscape Analysis after Task 2")
    print("="*70 + "\n")

    landscape2_task1 = analyze_loss_landscape(
        model, task1_test_loader, device,
        output_dir, f"task2_on_{mmlu_config.task1}",
        distance=landscape_config.distance,
        num_points=landscape_config.num_points,
        seed=landscape_config.random_seed
    )

    landscape2_task2 = analyze_loss_landscape(
        model, task2_test_loader, device,
        output_dir, f"task2_{mmlu_config.task2}",
        distance=landscape_config.distance,
        num_points=landscape_config.num_points,
        seed=landscape_config.random_seed
    )

    # ========== PHASE 8: Compute Forgetting Metrics ==========
    print("\n" + "="*70)
    print("PHASE 8: Catastrophic Forgetting Analysis")
    print("="*70 + "\n")

    forgetting_metrics = compute_forgetting_metrics(
        results_after_task1[f"{mmlu_config.task1}_accuracy"],
        results_after_task2[f"{mmlu_config.task1}_accuracy"],
        results_after_task2[f"{mmlu_config.task2}_accuracy"]
    )

    print_forgetting_report(forgetting_metrics)

    # ========== PHASE 9: Compare Loss Landscapes ==========
    print("\n" + "="*70)
    print("PHASE 9: Loss Landscape Comparison")
    print("="*70 + "\n")

    print(f"\nFor Task 1 ({mmlu_config.task1}) data:")
    compare_landscapes(
        landscape1_task1["metrics"],
        landscape2_task1["metrics"],
        "After Task 1 Training",
        "After Task 2 Training"
    )

    # ========== PHASE 10: Save All Results ==========
    print("\n" + "="*70)
    print("PHASE 10: Saving Results")
    print("="*70 + "\n")

    final_results = {
        "config": {
            "model": model_config.model_name,
            "task1": mmlu_config.task1,
            "task2": mmlu_config.task2,
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
    print(f"  - Landscapes: landscape_*.png")
    print(f"  - Metrics: results.json")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Catastrophic Forgetting Experiment")
    parser.add_argument("--task1", type=str, default="abstract_algebra", help="First MMLU task")
    parser.add_argument("--task2", type=str, default="anatomy", help="Second MMLU task")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs per task")

    args = parser.parse_args()
    main(args)
