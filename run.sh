#!/bin/bash
# Fix Intel MKL threading issue with PyTorch
export MKL_THREADING_LAYER=GNU

# Run experiment (assumes conda environment is already activated)
python main.py "$@"
