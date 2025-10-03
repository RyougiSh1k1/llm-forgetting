"""TRACE dataset loader for catastrophic forgetting experiments."""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict, List


class TRACEDataset(Dataset):
    """TRACE dataset wrapper supporting multiple task types."""

    def __init__(self, task_name: str, split: str = "train", tokenizer=None, max_length: int = 512,
                 train_samples: int = 500, test_samples: int = 200):
        """
        Initialize TRACE dataset for a specific task.

        Args:
            task_name: Task name ('scienceqa', 'fomc', etc.)
            split: Dataset split ('train', 'test', 'validation')
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            train_samples: Number of training samples to use
            test_samples: Number of test samples to use
        """
        self.task_name = task_name.lower()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.train_samples = train_samples
        self.test_samples = test_samples

        # Load dataset based on task name
        if self.task_name == "scienceqa":
            # ScienceQA dataset - only text-based questions (no images)
            full_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
            # Filter out image-based questions
            filtered_dataset = full_dataset.filter(lambda x: x['image'] is None)

            # Split into train/test using configured sample sizes
            total_available = len(filtered_dataset)
            train_size = min(train_samples, int(total_available * 0.8))

            if split == "train":
                self.dataset = filtered_dataset.select(range(train_size))
            else:  # test
                test_start = int(total_available * 0.8)
                test_end = min(total_available, test_start + test_samples)
                self.dataset = filtered_dataset.select(range(test_start, test_end))

        elif self.task_name == "fomc":
            # FOMC dataset for hawkish/dovish classification
            full_dataset = load_dataset("gtfintechlab/fomc_communication", split="train")

            # Split into train/test using configured sample sizes
            total_available = len(full_dataset)
            train_size = min(train_samples, int(total_available * 0.8))

            if split == "train":
                self.dataset = full_dataset.select(range(train_size))
            else:  # test
                test_start = int(total_available * 0.8)
                test_end = min(total_available, test_start + test_samples)
                self.dataset = full_dataset.select(range(test_start, test_end))
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def format_scienceqa(self, item: Dict) -> tuple:
        """Format ScienceQA question as prompt."""
        question = item["question"]
        choices = item["choices"]

        # Format choices as A, B, C, D...
        choice_labels = ["A", "B", "C", "D", "E", "F"][:len(choices)]
        formatted_choices = "\n".join([
            f"{label}: {choice}" for label, choice in zip(choice_labels, choices)
        ])

        # TRACE prompt: "Choose an answer for the following question and give your reasons."
        prompt = f"Choose an answer for the following question and give your reasons.\nQuestion: {question}\n{formatted_choices}\nAnswer:"

        # Get answer
        answer_idx = item["answer"]
        answer = choice_labels[answer_idx]

        return prompt, answer

    def format_fomc(self, item: Dict) -> tuple:
        """Format FOMC text as prompt."""
        text = item["sentence"]

        # TRACE prompt from Table 32
        prompt = "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n"
        prompt += f"Text: {text}\nAnswer:"

        # Map label to A/B/C (0=dovish, 1=hawkish, 2=neutral)
        label = item["label"]
        label_map = {0: "A", 1: "B", 2: "C"}
        answer = label_map[label]

        return prompt, answer

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item."""
        item = self.dataset[idx]

        # Format based on task type
        if self.task_name == "scienceqa":
            prompt, answer = self.format_scienceqa(item)
        elif self.task_name == "fomc":
            prompt, answer = self.format_fomc(item)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

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


def evaluate_accuracy(model, dataloader, device) -> float:
    """
    Evaluate model accuracy on TRACE task.

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

                # Check if prediction matches (first character should be the answer)
                if generated and generated[0].upper() == answer.upper():
                    correct += 1
                total += 1

    model.train()
    return correct / total if total > 0 else 0.0
