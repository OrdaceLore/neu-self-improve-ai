# Multi-Armed Bandit Algorithms (Week 01)

This project demonstrates and compares two classic approaches to the k-armed bandit problem:

- **ε-greedy Agent**
- **Gradient Bandit Agent**

The code simulates multiple runs of each agent on a stationary 10-armed bandit environment and plots their performance.

## Features
- Implements both ε-greedy and gradient bandit algorithms
- Compares average reward and percentage of optimal action over time
- Plots results and saves them as `bandit_results.png`

## Requirements
- Python 3.x
- numpy
- matplotlib

Install dependencies with:
```bash
pip install numpy matplotlib
```

## Usage
Run the experiment with:
```bash
python week_01/main.py
```

This will generate a plot (`bandit_results.png`) comparing the algorithms.

## File Structure
- `week_01/main.py` — Main script containing environment, agents, experiment runner, and plotting code
- `README.md` — This file

## Output
- `bandit_results.png` — Plot comparing average reward and % optimal action for each agent

## Remind
- It takes time to generate the photo, you could decrease the runs and steps to faster it in cost of efficacy

## Reference
Based on concepts from Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 2).