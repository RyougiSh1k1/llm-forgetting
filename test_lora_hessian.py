"""Test script for LoRA Hessian computation."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset

from loss_landscape import (
    compute_hessian_eigenvalues_lora_subspace,
    get_lora_parameters,
    analyze_loss_landscape_lora
)


def create_dummy_model_and_data():
    """Create a small LoRA model and dummy data for testing."""
    # Use a tiny model for testing
    model_name = "gpt2"  # Small model for testing

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )

    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],  # GPT-2 attention module
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dummy data
    batch_size = 2
    seq_length = 32

    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Custom collate function
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    return model, tokenizer, dataloader


def test_lora_parameter_extraction():
    """Test LoRA parameter extraction."""
    print("="*70)
    print("TEST 1: LoRA Parameter Extraction")
    print("="*70 + "\n")

    model, tokenizer, dataloader = create_dummy_model_and_data()
    device = "cpu"

    # Get LoRA parameters
    lora_params = get_lora_parameters(model, device)

    print(f"✓ LoRA parameters extracted successfully")
    print(f"  - LoRA parameter dimension: {lora_params.numel()}")
    print(f"  - LoRA parameter norm: {torch.norm(lora_params).item():.4f}")

    # Count LoRA params manually
    lora_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            lora_count += param.numel()

    assert lora_params.numel() == lora_count, "LoRA parameter count mismatch!"
    print(f"  - Manual count verification: {lora_count} (matches!)")

    return True


def test_hessian_computation():
    """Test Hessian eigenvalue computation in LoRA subspace."""
    print("\n" + "="*70)
    print("TEST 2: Hessian Eigenvalue Computation (LoRA Subspace)")
    print("="*70 + "\n")

    model, tokenizer, dataloader = create_dummy_model_and_data()
    device = "cpu"

    # Train for one step to ensure gradients exist
    print("Training model for 1 step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    batch = next(iter(dataloader))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"  Loss after 1 step: {loss.item():.4f}\n")

    # Compute Hessian eigenvalues
    num_eigenvalues = 5
    eigenvalues = compute_hessian_eigenvalues_lora_subspace(
        model, dataloader, device, num_eigenvalues=num_eigenvalues
    )

    print(f"\n✓ Hessian eigenvalues computed successfully")
    print(f"  - Number of eigenvalues: {len(eigenvalues)}")
    print(f"  - Eigenvalues: {eigenvalues}")
    print(f"  - λ_max (largest eigenvalue): {eigenvalues[0]:.6f}")

    # Verify properties
    assert len(eigenvalues) == num_eigenvalues, "Incorrect number of eigenvalues!"
    assert all(ev >= 0 for ev in eigenvalues), "Eigenvalues should be non-negative!"
    assert eigenvalues[0] >= eigenvalues[-1], "Eigenvalues should be sorted descending!"

    print("  - All eigenvalues are non-negative ✓")
    print("  - Eigenvalues are sorted in descending order ✓")

    return True


def test_landscape_analysis():
    """Test full landscape analysis in LoRA subspace."""
    print("\n" + "="*70)
    print("TEST 3: Full LoRA Landscape Analysis")
    print("="*70 + "\n")

    model, tokenizer, dataloader = create_dummy_model_and_data()
    device = "cpu"

    # Get initial LoRA params
    initial_lora_params = get_lora_parameters(model, device)

    # Train for a few steps
    print("Training model for 3 steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for step in range(3):
        batch = next(iter(dataloader))
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")

    # Analyze landscape
    print("\nAnalyzing loss landscape...")
    results = analyze_loss_landscape_lora(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir="./test_output",
        name="test_analysis",
        initial_lora_params=initial_lora_params,
        num_eigenvalues=5,
        compute_hessian=True
    )

    metrics = results["metrics"]

    print(f"\n✓ Landscape analysis completed successfully")
    print(f"\nResults:")
    print(f"  - λ_max (LoRA): {metrics['lambda_max_lora']:.6f}")
    print(f"  - LoRA displacement ||Δw||: {metrics['lora_displacement']:.4f}")
    print(f"  - LoRA dimension: {metrics['lora_dimension']}")
    print(f"  - Forgetting bound (½λ_max||Δw||²): {metrics['forgetting_bound']:.6f}")

    # Verify results
    assert metrics['lambda_max_lora'] >= 0, "λ_max should be non-negative!"
    assert metrics['lora_displacement'] > 0, "Displacement should be positive after training!"
    assert metrics['forgetting_bound'] >= 0, "Forgetting bound should be non-negative!"

    print("\n  All metrics are valid ✓")

    return True


def main():
    """Run all tests."""
    print("\n" + "🔬 TESTING LORA HESSIAN COMPUTATION 🔬")
    print("="*70 + "\n")

    try:
        # Run tests
        test1_passed = test_lora_parameter_extraction()
        test2_passed = test_hessian_computation()
        test3_passed = test_landscape_analysis()

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"\n✓ All tests passed successfully!\n")
        print("The LoRA Hessian computation implementation is working correctly.")
        print("\nYou can now run the main experiment with:")
        print("  python main.py --task1 scienceqa --task2 fomc")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
