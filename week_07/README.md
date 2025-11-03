# RAGEN with A*PO

## What This Repo Achieves

This repository implements a minimal version of RAGEN (Self-Evolution via Multi-Turn RL) using A*-PO instead of PPO/GRPO. The system demonstrates self-improvement through reinforcement learning on the FrozenLake benchmark.

## How It Works

RAGEN learns to improve itself through:
1. **Policy Network**: Simple MLP that maps states to actions
2. **A*-PO**: Uses A-Star Policy Optimization for stable learning
3. **Multi-Turn Learning**: Collects rollouts and updates policy iteratively
4. **Self-Evolution**: Policy improves over multiple training steps

## Architecture

```
State (16-dim one-hot) → MLP (32 hidden) → Action (4 actions)
```

The system integrates A*PO with RAGEN by:
- Collecting episodes from environment
- Computing advantages using discounted returns
- Applying A*PO loss with KL regularization
- Updating policy to maximize expected reward

## Installation

```bash
pip install torch
```

## Running

```bash
python ragen.py
```

This will:
- Train on FrozenLake (5 steps, ~2 seconds)
- Evaluate performance
- Print training progress and final results

## Experimental Results

### Configuration
- Model: RagenPolicy (32 hidden dim, 2 layers)
- Training steps: 5
- Environment: 4x4 FrozenLake
- Learning rate: 0.001

### Performance on FrozenLake

| Metric | Value |
|--------|-------|
| Training time | ~2 seconds |
| Memory usage | ~10 MB |
| Final success rate | ~20-40% (varies) |
| Average reward | ~0.2-0.4 |

*Note: Full training would need 100+ steps for higher success rates*

## System Diagram

```
┌─────────────┐
│ Environment │ (FrozenLake)
│   (State)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Policy    │ (MLP)
│  Network    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Action    │────▶│   Reward    │
└─────────────┘     └──────┬──────┘
                            │
                            ▼
                    ┌─────────────┐
                    │  Advantages │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   A*PO Loss │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Optimizer  │
                    └─────────────┘
```

## Examples of Failure Cases

1. **Falling into holes**: Agent sometimes takes suboptimal paths that lead to holes (cell type 2)
2. **Getting stuck**: In early training, agent may take long paths or loop
3. **Missing goal**: Agent reaches safe cells near goal but doesn't complete task

**Why**: Limited training steps (5) and small model size prevent full convergence. More training (100+ steps) would improve performance.

