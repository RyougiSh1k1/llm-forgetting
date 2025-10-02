"""Configuration for LLaMA-2 LoRA fine-tuning on MMLU tasks."""

from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_length: int = 512

@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./outputs"
    fp16: bool = True

@dataclass
class MMLUConfig:
    """MMLU dataset configuration."""
    task1: str = "abstract_algebra"
    task2: str = "anatomy"
    num_shots: int = 0  # 0 for fine-tuning

@dataclass
class LossLandscapeConfig:
    """Loss landscape analysis configuration."""
    num_directions: int = 2
    distance: float = 1.0
    num_points: int = 21  # Grid resolution
    random_seed: int = 42
