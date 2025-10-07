"""Configuration for LLaMA-2 LoRA fine-tuning on MMLU tasks."""

from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 512

@dataclass
class LoRAConfig:
    """LoRA configuration for Sequential Fine-Tuning (LoRA-SEFT)."""
    r: int = 2  # Increased from 16 for more capacity
    lora_alpha: int = 4  # Increased from 32 (typically 2*r)
    target_modules: List[str] = None
    lora_dropout: float = 0.1  # Increased from 0.05 for regularization
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # Include all attention projections + MLP layers for better learning
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"  # MLP (compatible with LLaMA-2/3)
            ]

@dataclass
class TrainingConfig:
    """Training configuration for continual learning."""
    batch_size: int = 1  # Reduced to avoid OOM
    gradient_accumulation_steps: int = 1  # Effective batch = 16
    num_epochs: int = 10  # Increased from 3 for better learning
    learning_rate: float = 3e-4  # Slightly higher for faster adaptation
    weight_decay: float = 0.01
    warmup_steps: int = 50  # Reduced from 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./outputs"
    fp16: bool = True
    max_grad_norm: float = 1.0  # Gradient clipping
    gradient_checkpointing: bool = True  # Enable to save memory

@dataclass
class TRACEConfig:
    """TRACE benchmark dataset configuration.

    Following TRACE benchmark: use maximally different domains
    to induce catastrophic forgetting.

    TRACE uses 8 tasks total, we implement the first 2:
    - Task 1: ScienceQA (Domain-specific - Science)
    - Task 2: FOMC (Domain-specific - Finance)

    Each task uses:
    - 5000 training samples
    - 2000 test samples
    """
    # TRACE benchmark tasks (first two)
    task1: str = "scienceqa"  # Science domain - multi-hop QA
    task2: str = "fomc"  # Finance domain - sentiment classification

    num_shots: int = 0  # 0 for fine-tuning

    # TRACE dataset sizes per task (reduced for faster training/testing)
    train_samples: int = 500  # Reduced from 5000
    test_samples: int = 200   # Reduced from 2000

@dataclass
class LossLandscapeConfig:
    """Loss landscape analysis configuration."""
    num_directions: int = 2
    distance: float = 1.0
    num_points: int = 21  # Grid resolution
    random_seed: int = 42
    compute_hessian: bool = False  # Set to False to skip Hessian/Fisher computation and avoid OOM
