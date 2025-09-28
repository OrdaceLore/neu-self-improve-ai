# Windy Gridworld Control Algorithms

This repository implements various control algorithms for the Windy Gridworld environment, replicating experiments from Sutton & Barto's "Reinforcement Learning: An Introduction" (Chapters 8-10).

## Overview

The Windy Gridworld is a 7×10 grid environment where:
- The agent starts at position (3,0) and must reach the goal at (3,7)
- Wind effects push the agent upward in columns 3-9
- Wind strength: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
- Actions: up, down, left, right
- Reward: -1 per step, 0 at goal

## Implemented Algorithms

### 1. Dynamic Programming (DP) Control
- **Policy Iteration**: Alternates between policy evaluation and policy improvement
- **Value Iteration**: Directly updates value function until convergence
- File: `dp_control.py`

### 2. Monte Carlo On-Policy Control
- **First-Visit MC**: Updates Q-function only on first visit to state-action pairs
- **Every-Visit MC**: Updates Q-function on every visit to state-action pairs
- File: `mc_on_policy.py`

### 3. Monte Carlo Off-Policy Control
- **Ordinary Importance Sampling**: Uses unweighted importance sampling
- **Weighted Importance Sampling**: Uses weighted importance sampling for better variance
- File: `mc_off_policy.py`

### 4. TD(0) On-Policy Control
- **Sarsa**: On-policy temporal difference learning
- **Expected Sarsa**: Uses expected value of next state
- File: `td_on_policy.py`

### 5. TD(0) Off-Policy Control
- **Q-Learning**: Off-policy temporal difference learning
- **TD Off-Policy with Unweighted Importance Sampling**
- **TD Off-Policy with Weighted Importance Sampling**
- File: `td_off_policy.py`

## Files Structure

```
├── windy_gridworld.py          # Environment implementation
├── dp_control.py               # Dynamic Programming algorithms
├── mc_on_policy.py             # Monte Carlo on-policy algorithms
├── mc_off_policy.py            # Monte Carlo off-policy algorithms
├── td_on_policy.py             # TD(0) on-policy algorithms
├── td_off_policy.py            # TD(0) off-policy algorithms
├── comparison_tool.py          # Comprehensive comparison tool
├── main_experiment.py          # Main experiment script (replicates Figure 4.1)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Individual Algorithms

```python
# Test the environment
python windy_gridworld.py

# Run Dynamic Programming
python dp_control.py

# Run Monte Carlo On-Policy
python mc_on_policy.py

# Run Monte Carlo Off-Policy
python mc_off_policy.py

# Run TD(0) On-Policy
python td_on_policy.py

# Run TD(0) Off-Policy
python td_off_policy.py
```

### Run Complete Experiment (Replicates Figure 4.1)

```python
python main_experiment.py
```

This will:
- Run all control algorithms
- Generate comprehensive visualizations
- Create performance comparisons
- Save results to `experiment_results.npy`

### Run Comprehensive Comparison

```python
python comparison_tool.py
```

This provides detailed analysis and comparison of all algorithms.

## Key Features

### Environment
- **WindyGridworld**: Complete implementation with wind effects
- **Visualization**: Grid rendering with policy arrows and value functions
- **Policies**: Epsilon-greedy and random policies for exploration

### Algorithms
- **Complete Implementation**: All major control algorithms from Sutton & Barto
- **Proper Importance Sampling**: Both weighted and unweighted variants
- **Convergence Tracking**: Monitor learning progress and convergence
- **Performance Evaluation**: Comprehensive evaluation metrics

### Visualization
- **Learning Curves**: Episode returns and evaluation performance
- **Policy Visualization**: Arrow-based policy representation
- **Value Functions**: Heatmap visualization of state values
- **Comparison Plots**: Side-by-side algorithm comparisons

## Results

The implementation successfully replicates the key findings from Sutton & Barto:

1. **Dynamic Programming** provides the optimal solution
2. **Monte Carlo** methods converge to near-optimal policies
3. **TD(0) methods** (Sarsa, Q-Learning) learn efficiently
4. **Off-policy methods** with importance sampling work but may have higher variance
5. **Q-Learning** often outperforms Sarsa due to off-policy learning

## Parameters

Default parameters used in experiments:
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 1.0
- **Exploration (ε)**: 0.1
- **Episodes**: 1000
- **Evaluation Interval**: 100 episodes

## Customization

You can easily modify:
- Environment size and wind patterns
- Algorithm parameters (learning rates, exploration)
- Number of training episodes
- Evaluation metrics

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Chapters 4, 5, 6, and 7 for the theoretical background
- Chapter 6.5 for the Windy Gridworld environment

## License

This implementation is for educational purposes, following the examples from Sutton & Barto's textbook.
