# LLM Catastrophic Forgetting & Loss Landscape Analysis

A framework for examining the relationship between loss landscape flatness and catastrophic forgetting in Large Language Models, using LLaMA-2 7B fine-tuned with LoRA on MMLU tasks.

## Overview

This project investigates how the flatness of the loss landscape relates to catastrophic forgetting when sequentially fine-tuning LLMs on multiple tasks. The framework:

1. Fine-tunes LLaMA-2 7B with LoRA on Task 1
2. Analyzes the loss landscape around the checkpoint
3. Fine-tunes on Task 2
4. Analyzes the loss landscape again
5. Measures catastrophic forgetting and compares landscape properties

## Project Structure

```
llm-forgetting/
├── config.py              # Configuration classes
├── data_loader.py         # MMLU dataset loading and evaluation
├── train.py              # LoRA fine-tuning utilities
├── loss_landscape.py     # Loss landscape analysis and visualization
├── evaluate.py           # Catastrophic forgetting metrics
├── main.py              # Main experiment orchestration
└── requirements.txt     # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

**Note:** You'll need access to LLaMA-2 7B. Set up your HuggingFace token:
```bash
huggingface-cli login
```

## Usage

### Basic Experiment

Run with default tasks (abstract_algebra → anatomy):
```bash
python main.py
```

### Custom Tasks

Specify your own MMLU tasks:
```bash
python main.py --task1 computer_science --task2 philosophy --batch_size 16 --num_epochs 3
```

### Available Arguments

- `--task1`: First MMLU task name (default: abstract_algebra)
- `--task2`: Second MMLU task name (default: anatomy)
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Epochs per task (default: 3)

## MMLU Tasks

Available MMLU tasks include:
- abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge
- college_biology, college_chemistry, college_computer_science, college_mathematics
- college_medicine, college_physics, computer_security, conceptual_physics
- econometrics, electrical_engineering, elementary_mathematics, formal_logic
- global_facts, high_school_biology, high_school_chemistry, and more...

See the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu) for the complete list.

## Experiment Workflow

The framework runs through 10 phases:

1. **Setup**: Load model, tokenizer, and datasets
2. **Train Task 1**: Fine-tune with LoRA on first task
3. **Evaluate**: Measure accuracy on both tasks
4. **Landscape 1**: Analyze loss landscape after Task 1
5. **Train Task 2**: Fine-tune on second task
6. **Evaluate**: Measure accuracy again on both tasks
7. **Landscape 2**: Analyze loss landscape after Task 2
8. **Forgetting**: Compute catastrophic forgetting metrics
9. **Compare**: Compare loss landscape properties
10. **Save**: Export all results and visualizations

## Output

Results are saved in `outputs/experiment_TIMESTAMP/`:

- `checkpoint_after_task1/`: Model checkpoint after Task 1
- `checkpoint_after_task2/`: Model checkpoint after Task 2
- `landscape_*.png`: 3D and contour plots of loss landscapes
- `results.json`: All metrics and configurations

### Metrics Computed

**Catastrophic Forgetting:**
- Absolute forgetting (accuracy drop on Task 1)
- Relative forgetting (percentage drop)
- Backward transfer
- Average accuracy across tasks

**Loss Landscape Properties:**
- Center loss
- Maximum loss in neighborhood
- Standard deviation (flatness indicator)
- Maximum curvature (sharpness indicator)
- Loss range

## Key Insights

The framework helps answer questions like:
- Does a flatter loss landscape correlate with less catastrophic forgetting?
- How does the landscape change when learning a new task?
- What's the relationship between sharpness and task retention?

## Configuration

Edit `config.py` to customize:

- **Model**: Base model name, max sequence length
- **LoRA**: Rank, alpha, target modules, dropout
- **Training**: Batch size, learning rate, epochs, optimizer settings
- **Landscape**: Number of directions, distance, grid resolution

## Requirements

- CUDA-capable GPU (recommended)
- ~30GB GPU memory for LLaMA-2 7B with LoRA
- Python 3.8+
- PyTorch 2.0+
- Transformers, PEFT, Datasets libraries

## Citation

If you use this framework, please cite the MMLU dataset:
```
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

## License

MIT License - see LICENSE file for details
