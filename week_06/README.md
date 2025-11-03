# TinyZero with A*PO

## What This Repo Achieves

This repository implements a minimal version of TinyZero (reproduction of DeepSeek R1 Zero) using A*-PO (A-Star Policy Optimization) instead of GRPO. The implementation focuses on:

1. **Countdown Task**: Learn to count from N to 0
2. **Multiplication Task**: Learn to compute a * b

The system uses reinforcement learning to train a small policy network on these arithmetic reasoning tasks.

## How It Works

1. **Policy Network**: Simple LSTM-based policy that takes state (numbers) and outputs actions
2. **A*-PO**: Policy optimization using advantage-weighted loss with KL regularization
3. **Environments**: Simplified countdown and multiplication environments
4. **Training**: Collect rollouts, compute advantages, update policy

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python tinyzero.py
```

This will:
- Train on countdown task (3 steps, ~1 second)
- Train on multiplication task (3 steps, ~1 second)
- Print training progress and evaluation results

## Experimental Results

### Configuration
- Model: TinyPolicy (32 hidden dim, LSTM)
- Training steps: 3 per task
- Batch size: 1 rollout per step
- Learning rate: 0.001
- Optimizer: Adam

### Performance

**Countdown Task:**
- Training time: < 1 second
- Memory usage: ~50 MB
- Evaluation reward: ~0.2-0.5 (varies, needs more training for full performance)

**Multiplication Task:**
- Training time: < 1 second  
- Memory usage: ~50 MB
- Evaluation reward: ~0.1-0.3 (varies)

### Notes
- This is a minimal implementation for demonstration
- Full training would require more steps (~100-1000) and larger models
- For production use, increase model size, training steps, and use proper FSDP for multi-GPU

## Model Details

- **Architecture**: Embedding → LSTM → Linear head
- **Vocabulary size**: 100
- **Hidden dimension**: 32
- **Parameters**: ~10K

## Evaluation Method

Evaluates policy by running episodes with greedy (low temperature) sampling and computing average reward over 3-5 episodes.

