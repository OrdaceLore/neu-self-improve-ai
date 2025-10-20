# PAG with A*-PO for Mathematical Reasoning

This project implements the PAG (Policy as Generative Verifier) framework using A*-PO (A*-Policy Optimization) instead of PPO for training the Qwen2.5-1.5B-Instruct model on mathematical reasoning tasks.

## Overview

- **Model**: Qwen2.5-1.5B-Instruct
- **Dataset**: MATH dataset (training) and MATH 500 (evaluation)
- **Algorithm**: A*-PO (A*-Policy Optimization)
- **Framework**: PAG (Policy as Generative Verifier)

## Project Structure

```
├── requirements.txt          # Dependencies
├── README.md                # This file
├── config.py               # Configuration settings
├── data/
│   ├── __init__.py
│   ├── math_dataset.py     # MATH dataset loading and preprocessing
│   └── reward_model.py     # Reward model for mathematical reasoning
├── models/
│   ├── __init__.py
│   ├── qwen_model.py      # Qwen2.5-1.5B-Instruct integration
│   └── pag_model.py       # PAG framework implementation
├── algorithms/
│   ├── __init__.py
│   ├── astar_po.py        # A*-PO algorithm implementation
│   └── monte_carlo.py     # Monte Carlo gradient estimation
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Training loop
│   └── evaluator.py       # Evaluation on MATH 500
└── main.py                # Main entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config.py
```

## Key Components

1. **A*-PO Algorithm**: Two-stage policy optimization with offline value estimation and online policy updates
2. **PAG Framework**: Multi-turn self-correction with policy as generative verifier
3. **MATH Dataset**: Mathematical reasoning benchmark with 12,500 problems
4. **Reward Model**: Custom reward function for mathematical reasoning quality
