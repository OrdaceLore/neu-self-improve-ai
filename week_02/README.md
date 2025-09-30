# Reinforcement Learning Algorithm Implementation
## Based on Sutton & Barto "Reinforcement Learning: An Introduction" (2nd Edition)

### 🎯 Project Overview
This project implements and compares the core reinforcement learning control algorithms from Sutton & Barto's seminal textbook, including:

1. **Figure 4.1 Replication**: Policy iteration in a 4×4 gridworld
2. **Windy Gridworld Environment**: Classic RL environment with wind effects
3. **Six Control Algorithms**:
   - Dynamic Programming Control
   - Monte Carlo On-Policy Control
   - Monte Carlo Off-Policy Control
   - TD(0) On-Policy Control (SARSA)
   - TD(0) Off-Policy Control (Unweighted Importance Sampling)
   - TD(0) Off-Policy Control (Weighted Importance Sampling)

### 📁 Project Structure
```
├── figure_4_1_replication.py    # Replicates Figure 4.1 from the book
├── windy_gridworld.py           # Windy Gridworld environment implementation
├── rl_algorithms.py             # All RL control algorithms
├── analysis_and_visualization.py # Comprehensive analysis and plotting
├── main.py                      # Main script to run all experiments
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run All Experiments**:
   ```bash
   python main.py
   ```

3. **Run Individual Components**:
   ```bash
   # Replicate Figure 4.1
   python figure_4_1_replication.py
   
   # Test Windy Gridworld
   python windy_gridworld.py
   
   # Compare algorithms
   python rl_algorithms.py
   
   # Full analysis
   python analysis_and_visualization.py
   ```

### 📊 Generated Outputs
The experiments generate several visualization files:

- `figure_4_1_replication.png` - Policy iteration visualization
- `windy_gridworld.png` - Environment layout
- `algorithm_comparison.png` - Performance comparison
- `learning_curves.png` - Learning progress over episodes
- `optimal_policies.png` - Optimal policies found by each algorithm
- `convergence_analysis.png` - Convergence speed analysis
- `performance_summary.png` - Summary table of results

### 🔬 Algorithm Details

#### 1. Dynamic Programming Control
- **Method**: Policy Iteration
- **Convergence**: Guaranteed, typically 2-3 iterations
- **Advantage**: Fast convergence, optimal solution
- **Disadvantage**: Requires complete model of environment

#### 2. Monte Carlo On-Policy Control
- **Method**: First-visit MC with epsilon-greedy policy
- **Convergence**: Slower, requires many episodes
- **Advantage**: Model-free, learns from experience
- **Disadvantage**: High variance, slow convergence

#### 3. Monte Carlo Off-Policy Control
- **Method**: Importance sampling with behavior policy
- **Convergence**: Slower than on-policy
- **Advantage**: Can learn from any behavior policy
- **Disadvantage**: High variance, importance sampling issues

#### 4. TD(0) On-Policy Control (SARSA)
- **Method**: Temporal difference learning with on-policy updates
- **Convergence**: Faster than MC, online learning
- **Advantage**: Model-free, online, lower variance than MC
- **Disadvantage**: May converge to suboptimal policy

#### 5. TD(0) Off-Policy Control (Unweighted IS)
- **Method**: Q-learning with unweighted importance sampling
- **Convergence**: Can be unstable
- **Advantage**: Can learn optimal policy off-policy
- **Disadvantage**: High variance, potential instability

#### 6. TD(0) Off-Policy Control (Weighted IS)
- **Method**: Q-learning with weighted importance sampling
- **Convergence**: More stable than unweighted
- **Advantage**: Better variance control, more stable
- **Disadvantage**: Still more complex than on-policy methods

### 📈 Expected Results
Based on the implementation, you should observe:

1. **DP Control**: Fastest convergence, optimal performance
2. **TD(0) On-Policy**: Good balance of speed and performance
3. **MC Methods**: Slower convergence, higher variance
4. **Off-Policy Methods**: More complex, potentially unstable

### 🎓 Educational Value
This implementation helps understand:

- **Policy Iteration**: How optimal policies emerge from random initialization
- **Model-Free Learning**: How algorithms learn without environment models
- **On-Policy vs Off-Policy**: Trade-offs between different learning paradigms
- **Importance Sampling**: How to learn from different behavior policies
- **Convergence Properties**: How different algorithms converge

### 📚 References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- [Official Book Website](https://reinforcementlearning.pubpub.org/)
- [Sutton's Website](http://incompleteideas.net/)

### 🔧 Technical Notes
- **Environment**: 7×10 Windy Gridworld with wind strengths [0,0,0,1,1,1,2,2,1,0]
- **Reward Structure**: -1 per step, 0 at goal
- **Hyperparameters**: γ=1.0, ε=0.1, α=0.1 (where applicable)
- **Evaluation**: 100 episodes for final performance assessment

---
*Happy Learning! 🚀*
