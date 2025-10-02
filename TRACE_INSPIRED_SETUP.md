# TRACE-Inspired Task Configuration

## Overview

Following the TRACE benchmark (Wang et al. 2023), we configure tasks to **maximize catastrophic forgetting** by using:
1. **Different knowledge domains** (STEM vs Humanities)
2. **Different cognitive processes** (reasoning vs memorization)
3. **Minimal semantic overlap** between tasks

## Why TRACE Principles Matter

### Problem with Same-Dataset Tasks
```
abstract_algebra → anatomy
```
- Both are MMLU multiple choice
- Both require factual recall
- Similar question format
- **Low interference → Low forgetting**

### TRACE Solution: Diverse Domains
```
college_mathematics → us_history
```
- Math reasoning vs Historical facts
- Formulas/proofs vs Names/dates
- Quantitative vs Qualitative
- **High interference → High forgetting** ✓

## Current Configuration

### Default Setup (config.py)
```python
task1: "college_mathematics"  # Quantitative reasoning
task2: "us_history"           # Historical facts/memorization
```

**Why this works:**
- Mathematics requires: formula application, logical reasoning, abstract thinking
- History requires: memorizing dates, understanding narratives, contextual knowledge
- **Minimal overlap** in neural representations → **Maximum forgetting**

## Alternative TRACE-Style Task Pairs

All pairs below follow TRACE principles of maximal domain separation:

### Option 2: Hard Science → Ethics
```python
task1: "high_school_physics"  # Physics concepts/formulas
task2: "moral_scenarios"      # Ethical judgment/values
```
- Physics: Objective, quantitative, rule-based
- Ethics: Subjective, qualitative, context-dependent

### Option 3: Technical → Cultural
```python
task1: "high_school_computer_science"  # Programming/algorithms
task2: "world_religions"               # Cultural/spiritual knowledge
```
- CS: Logic, algorithms, problem-solving
- Religion: Beliefs, traditions, narratives

### Option 4: Formal Logic → Philosophy
```python
task1: "formal_logic"  # Deductive reasoning
task2: "philosophy"    # Abstract/subjective reasoning
```
- Logic: Strict rules, binary truth
- Philosophy: Interpretation, multiple perspectives

## Expected Forgetting Patterns

With TRACE-inspired configuration:

### Task 1: College Mathematics
```
Before Task 2: 45-65% accuracy
After Task 2:  30-50% accuracy
Forgetting:    15-20% absolute drop ✓
```

### Task 2: US History
```
Before training: ~25% (random)
After training:  45-65% accuracy
```

## Why This Induces Forgetting

### 1. **Representational Interference**
- Math optimizes for quantitative features
- History optimizes for qualitative features
- Shared parameters must choose → forgetting

### 2. **Different Activation Patterns**
- Math: Calculation-heavy layers
- History: Language/context layers
- Sequential training disrupts previous patterns

### 3. **Knowledge Type Mismatch**
- Math: Procedural knowledge (how to solve)
- History: Declarative knowledge (what happened)
- Hard to maintain both simultaneously

## Full TRACE Benchmark (Future Work)

For complete TRACE compatibility, use these 8 diverse datasets:

```python
# Natural Language Understanding
1. SQuAD 2.0 (Question Answering)
2. MNLI (Natural Language Inference)

# Text Generation
3. XSum (Summarization)
4. WMT14 (Translation)

# Knowledge & Reasoning
5. MMLU (Multiple domains)
6. CommonsenseQA (Reasoning)

# Code & Math
7. APPS (Programming)
8. MATH (Mathematical reasoning)
```

## Running with TRACE Configuration

```bash
# Default TRACE-inspired setup (Math → History)
python main.py --num_epochs 10

# Try alternative pair
# Edit config.py to uncomment other options, then:
python main.py --num_epochs 10
```

## Validation Criteria

Your setup works if you see:

✅ **Task 1 accuracy improves** (>40%) during training
✅ **Task 1 accuracy drops** (>10% absolute) after Task 2
✅ **Parameter displacement** ||Δw|| > 20
✅ **Different loss curves** between tasks

## References

- **TRACE**: Wang et al. (2023) - "TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models"
- **Catastrophic Forgetting**: Mirzadeh et al. (2020) - "Understanding the Role of Training Regimes in Continual Learning"

## Quick Switch Between Options

Edit `config.py` line 59-60:
```python
# Current:
task1: str = "college_mathematics"
task2: str = "us_history"

# Switch to physics → ethics:
task1: str = "high_school_physics"
task2: str = "moral_scenarios"
```

Then run: `python main.py`
