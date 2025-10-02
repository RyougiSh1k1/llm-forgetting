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
    """LoRA configuration for Sequential Fine-Tuning (LoRA-SEFT)."""
    r: int = 64  # Increased from 16 for more capacity
    lora_alpha: int = 128  # Increased from 32 (typically 2*r)
    target_modules: List[str] = None
    lora_dropout: float = 0.1  # Increased from 0.05 for regularization
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # Include all attention projections + MLP layers for better learning
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"  # MLP (for LLaMA-2)
            ]

@dataclass
class TrainingConfig:
    """Training configuration for continual learning."""
    batch_size: int = 8  # Smaller batch for more frequent updates
    gradient_accumulation_steps: int = 2  # Effective batch = 16
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

@dataclass
class MMLUConfig:
    """MMLU dataset configuration.

    Following TRACE principles: use maximally different domains
    to induce catastrophic forgetting.

    TRACE-style task selection:
    - Different knowledge domains (STEM vs Humanities)
    - Different cognitive processes (reasoning vs memorization)
    - Minimal semantic overlap
    """
    # OPTION 1: Math/Logic → Social Science/Memorization (RECOMMENDED)
    task1: str = "college_mathematics"  # Quantitative reasoning
    task2: str = "high_school_us_history"  # Historical facts/memorization (FIXED: was "us_history")

    # OPTION 2: Hard Science → Soft Humanities
    # task1: str = "high_school_physics"  # Physics concepts
    # task2: str = "moral_scenarios"  # Ethics/judgment

    # OPTION 3: Technical → Cultural
    # task1: str = "high_school_computer_science"  # Technical
    # task2: str = "world_religions"  # Cultural knowledge

    # OPTION 4: Formal Logic → Subjective Interpretation
    # task1: str = "formal_logic"  # Deductive reasoning
    # task2: str = "philosophy"  # Subjective reasoning

    num_shots: int = 0  # 0 for fine-tuning

@dataclass
class LossLandscapeConfig:
    """Loss landscape analysis configuration."""
    num_directions: int = 2
    distance: float = 1.0
    num_points: int = 21  # Grid resolution
    random_seed: int = 42
