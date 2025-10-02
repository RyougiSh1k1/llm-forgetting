"""MMLU dataset loader for catastrophic forgetting experiments."""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict, List


class MMLUDataset(Dataset):
    """MMLU dataset wrapper for specific tasks."""

    def __init__(self, task_name: str, split: str = "test", tokenizer=None, max_length: int = 512):
        """
        Initialize MMLU dataset for a specific task.

        Args:
            task_name: MMLU task name (e.g., 'abstract_algebra', 'anatomy')
            split: Dataset split ('test', 'dev', 'validation')
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load MMLU dataset
        self.dataset = load_dataset("cais/mmlu", task_name, split=split)

        # Answer choices
        self.choices = ["A", "B", "C", "D"]

    def __len__(self) -> int:
        return len(self.dataset)

    def format_question(self, item: Dict) -> str:
        """Format MMLU question as prompt."""
        question = item["question"]
        choices = "\n".join([
            f"{self.choices[i]}: {item['choices'][i]}"
            for i in range(len(item['choices']))
        ])
        prompt = f"Question: {question}\n{choices}\nAnswer:"
        return prompt

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item."""
        item = self.dataset[idx]

        # Format prompt
        prompt = self.format_question(item)
        answer = self.choices[item["answer"]]

        # Full text for training
        full_text = f"{prompt} {answer}"

        if self.tokenizer:
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # For causal LM, labels are the same as input_ids
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()

            # Create labels (mask padding tokens)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt": prompt,
                "answer": answer
            }
        else:
            return {
                "prompt": prompt,
                "answer": answer,
                "full_text": full_text
            }


def create_dataloaders(task1: str, task2: str, tokenizer, batch_size: int = 16,
                       max_length: int = 512, split: str = "test"):
    """
    Create dataloaders for two MMLU tasks.

    Args:
        task1: First MMLU task name
        task2: Second MMLU task name
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        max_length: Maximum sequence length
        split: Dataset split

    Returns:
        Tuple of (task1_loader, task2_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    dataset1 = MMLUDataset(task1, split=split, tokenizer=tokenizer, max_length=max_length)
    dataset2 = MMLUDataset(task2, split=split, tokenizer=tokenizer, max_length=max_length)

    # Create dataloaders
    loader1 = DataLoader(
        dataset1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    loader2 = DataLoader(
        dataset2,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return loader1, loader2


def evaluate_accuracy(model, dataloader, device) -> float:
    """
    Evaluate model accuracy on MMLU task.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on

    Returns:
        Accuracy score
    """
    model.eval()
    correct = 0
    total = 0

    choices = ["A", "B", "C", "D"]

    with torch.no_grad():
        for batch in dataloader:
            prompts = batch["prompt"]
            answers = batch["answer"]

            # Get model predictions
            for prompt, answer in zip(prompts, answers):
                # Tokenize prompt only
                inputs = dataloader.dataset.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                # Generate next token
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=dataloader.dataset.tokenizer.pad_token_id
                )

                # Decode prediction
                generated = dataloader.dataset.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # Check if prediction matches
                if generated and generated[0] == answer:
                    correct += 1
                total += 1

    model.train()
    return correct / total if total > 0 else 0.0
