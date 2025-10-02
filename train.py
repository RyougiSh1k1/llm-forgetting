"""LoRA fine-tuning script for LLaMA-2 7B on MMLU tasks."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import os
from typing import Optional


def setup_model_and_tokenizer(model_name: str, lora_config: dict):
    """
    Setup LLaMA-2 model with LoRA.

    Args:
        model_name: Model identifier
        lora_config: LoRA configuration dict

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_on_task(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    training_config: dict,
    task_name: str
):
    """
    Train model on a specific task.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory
        training_config: Training configuration dict
        task_name: Name of task for logging

    Returns:
        Trained model
    """
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=training_config["fp16"],
        report_to="none",
        load_best_model_at_end=False,
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\n{'='*50}")
    print(f"Training on {task_name}")
    print(f"{'='*50}\n")

    trainer.train()

    # Save final model
    final_dir = os.path.join(output_dir, f"final_{task_name}")
    trainer.save_model(final_dir)

    print(f"\nModel saved to {final_dir}")

    return model


def load_checkpoint(base_model_name: str, checkpoint_path: str):
    """
    Load model from checkpoint.

    Args:
        base_model_name: Base model identifier
        checkpoint_path: Path to checkpoint

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return model, tokenizer


def save_checkpoint(model, tokenizer, save_path: str):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        tokenizer: Tokenizer
        save_path: Path to save checkpoint
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Checkpoint saved to {save_path}")


def get_model_parameters(model):
    """
    Get trainable parameters as a flat vector.

    Args:
        model: Model

    Returns:
        1D tensor of parameters
    """
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.data.view(-1))
    return torch.cat(params)


def set_model_parameters(model, param_vector):
    """
    Set trainable parameters from a flat vector.

    Args:
        model: Model
        param_vector: 1D tensor of parameters
    """
    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.data = param_vector[offset:offset + numel].view(param.shape)
            offset += numel
