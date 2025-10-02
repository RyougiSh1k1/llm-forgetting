# LoRA-SEFT Configuration for Catastrophic Forgetting

This document explains the updated configuration designed to induce and measure catastrophic forgetting in sequential fine-tuning.

## Changes Made

### 1. **LoRA Configuration** (config.py)

**Before:**
```python
r: int = 16
lora_alpha: int = 32
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
lora_dropout: float = 0.05
```

**After (LoRA-SEFT):**
```python
r: int = 64  # 4x increase → more trainable parameters
lora_alpha: int = 128  # 4x increase (maintains scaling)
target_modules: [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"  # MLP layers
]
lora_dropout: float = 0.1  # Increased regularization
```

**Why:**
- Higher rank (r=64) gives model more capacity to learn each task deeply
- More trainable modules means more parameter changes → more forgetting
- This matches typical continual learning setups where forgetting is observed

### 2. **Training Configuration**

**Before:**
```python
batch_size: int = 16
num_epochs: int = 3
learning_rate: float = 2e-4
```

**After:**
```python
batch_size: int = 8
gradient_accumulation_steps: int = 2  # Effective batch = 16
num_epochs: int = 10  # 3x increase
learning_rate: float = 3e-4  # 1.5x increase
max_grad_norm: float = 1.0  # Added gradient clipping
```

**Why:**
- More epochs → better task 1 performance → more to forget
- Higher learning rate → faster adaptation to task 2 → stronger interference
- Gradient clipping prevents training instability

### 3. **Task Selection**

**Before:**
```python
task1: str = "abstract_algebra"  # Math
task2: str = "anatomy"  # Biology
```

**After:**
```python
task1: str = "abstract_algebra"  # Math reasoning
task2: str = "world_religions"  # Memorization/facts
```

**Why:**
- Tasks with less semantic overlap → more forgetting
- Different cognitive demands (reasoning vs. memorization)
- Follows TRACE benchmark principles for continual learning

**Alternative pairs to try:**
- `college_physics` → `philosophy`
- `college_mathematics` → `world_religions`
- `high_school_chemistry` → `professional_law`

### 4. **Fixed Hessian Eigenvalue Computation**

**Problem:** Original implementation returned all zeros

**Solution:**
1. Improved finite difference approximation
2. Added support for `pytorch-hessian-eigenthings` library
3. Better error handling and diagnostics
4. Computes eigenvalues at trained checkpoints (not initialization)

## Expected Results

With these changes, you should see:

### Accuracy Pattern
```
After Task 1:
  Task 1 accuracy: 40-60% (improved from ~16%)
  Task 2 accuracy: ~25% (random baseline)

After Task 2:
  Task 1 accuracy: 30-45% (FORGETTING: 10-20% drop)
  Task 2 accuracy: 40-60% (learned)
```

### Eigenvalues
```
λ_max > 0 (e.g., 0.01 - 0.1)
Top 20 eigenvalues showing decay pattern
```

### Parameter Displacement
```
||Δw|| > 10 (meaningful parameter change)
Correlates with forgetting via: F₁ ≤ (1/2) λ_max ||Δw||²
```

## Running the Updated Experiment

```bash
# Kill any existing processes
pkill -f python

# Run with default settings (now uses LoRA-SEFT)
python main.py

# Or with custom tasks
python main.py --task1 college_physics --task2 philosophy --batch_size 8 --num_epochs 10
```

## Monitoring Progress

During training, you should see:
1. **Task 1 training:** Loss decreasing over 10 epochs
2. **Task 1 evaluation:** Accuracy significantly above random (>30%)
3. **Hessian computation:** Non-zero eigenvalues reported
4. **Task 2 training:** Loss decreasing again
5. **Task 1 re-evaluation:** Accuracy drop (forgetting!)

## Troubleshooting

### If accuracy still doesn't improve:
- Check that dataset is loading correctly (not empty)
- Verify LoRA modules are being trained (`model.print_trainable_parameters()`)
- Try even longer training (15-20 epochs)

### If eigenvalues still zero:
- Install: `pip install pytorch-hessian-eigenthings`
- Check that model has requires_grad=True parameters
- Verify gradients are flowing (add debug prints)

### If no forgetting occurs:
- Use more dissimilar task pairs
- Increase learning rate further (5e-4)
- Reduce dropout (allows more overfitting)

## References

- TRACE Benchmark: Wang et al. 2023
- Loss Landscape & Forgetting: Mirzadeh et al. 2020
- LoRA: Hu et al. 2021
