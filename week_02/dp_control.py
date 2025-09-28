"""
Dynamic Programming Control for Windy Gridworld
Based on Sutton & Barto Chapter 4
"""

import numpy as np
from windy_gridworld import WindyGridworld
from typing import Tuple, List
import matplotlib.pyplot as plt


class DPControl:
    """
    Dynamic Programming Control using Policy Iteration.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, theta: float = 1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize value function and policy
        self.V = np.zeros(self.num_states)
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        
        # Transition probabilities and rewards
        self.P = self._compute_transition_probabilities()
        self.R = self._compute_rewards()
        
    def _compute_transition_probabilities(self) -> np.ndarray:
        """Compute transition probabilities P(s'|s,a)."""
        P = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for state_idx in range(self.num_states):
            state = self.env.get_state_from_index(state_idx)
            
            # Terminal states have no transitions
            if self.env.is_terminal(state):
                continue
                
            for action_idx in range(self.num_actions):
                # Simulate the action to get next state
                original_state = self.env.current_state
                original_steps = self.env.episode_steps
                
                self.env.current_state = state
                self.env.episode_steps = 0
                next_state, _, _, _ = self.env.step(action_idx)
                next_state_idx = self.env.get_state_index(next_state)
                
                P[state_idx, action_idx, next_state_idx] = 1.0
                
                # Restore original state
                self.env.current_state = original_state
                self.env.episode_steps = original_steps
                
        return P
    
    def _compute_rewards(self) -> np.ndarray:
        """Compute expected rewards R(s,a)."""
        R = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            state = self.env.get_state_from_index(state_idx)
            
            # Terminal states have no rewards
            if self.env.is_terminal(state):
                continue
                
            for action_idx in range(self.num_actions):
                # Simulate the action to get reward
                original_state = self.env.current_state
                original_steps = self.env.episode_steps
                
                self.env.current_state = state
                self.env.episode_steps = 0
                _, reward, _, _ = self.env.step(action_idx)
                R[state_idx, action_idx] = reward
                
                # Restore original state
                self.env.current_state = original_state
                self.env.episode_steps = original_steps
                
        return R
    
    def policy_evaluation(self) -> None:
        """Evaluate the current policy."""
        max_iterations = 100  # Limit iterations for efficiency
        for _ in range(max_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for state_idx in range(self.num_states):
                if self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                    continue
                    
                # Compute new value
                new_value = 0
                for action_idx in range(self.num_actions):
                    action_prob = self.policy[state_idx, action_idx]
                    
                    # Expected reward
                    expected_reward = self.R[state_idx, action_idx]
                    
                    # Expected next state value
                    expected_next_value = 0
                    for next_state_idx in range(self.num_states):
                        expected_next_value += (self.P[state_idx, action_idx, next_state_idx] * 
                                              V_old[next_state_idx])
                    
                    new_value += action_prob * (expected_reward + self.gamma * expected_next_value)
                
                self.V[state_idx] = new_value
                delta = max(delta, abs(V_old[state_idx] - new_value))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self) -> bool:
        """Improve the policy greedily. Returns True if policy changed."""
        policy_stable = True
        
        for state_idx in range(self.num_states):
            if self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                continue
                
            old_action = np.argmax(self.policy[state_idx])
            
            # Compute action values
            action_values = np.zeros(self.num_actions)
            for action_idx in range(self.num_actions):
                expected_reward = self.R[state_idx, action_idx]
                expected_next_value = 0
                for next_state_idx in range(self.num_states):
                    expected_next_value += (self.P[state_idx, action_idx, next_state_idx] * 
                                          self.V[next_state_idx])
                action_values[action_idx] = expected_reward + self.gamma * expected_next_value
            
            # Greedy action
            best_action = np.argmax(action_values)
            
            # Update policy
            new_policy = np.zeros(self.num_actions)
            new_policy[best_action] = 1.0
            self.policy[state_idx] = new_policy
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self, max_iterations: int = 100) -> Tuple[List[float], List[int]]:
        """
        Run policy iteration algorithm.
        
        Returns:
            (value_history, iteration_history)
        """
        value_history = []
        iteration_history = []
        
        for iteration in range(max_iterations):
            # Policy evaluation
            self.policy_evaluation()
            value_history.append(self.V.copy())
            iteration_history.append(iteration)
            
            # Policy improvement
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                print(f"Policy iteration converged after {iteration + 1} iterations")
                break
        
        return value_history, iteration_history
    
    def value_iteration(self, max_iterations: int = 1000) -> Tuple[List[float], List[int]]:
        """
        Run value iteration algorithm.
        
        Returns:
            (value_history, iteration_history)
        """
        value_history = []
        iteration_history = []
        
        for iteration in range(max_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for state_idx in range(self.num_states):
                if self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                    continue
                
                # Compute action values
                action_values = np.zeros(self.num_actions)
                for action_idx in range(self.num_actions):
                    expected_reward = self.R[state_idx, action_idx]
                    expected_next_value = 0
                    for next_state_idx in range(self.num_states):
                        expected_next_value += (self.P[state_idx, action_idx, next_state_idx] * 
                                              V_old[next_state_idx])
                    action_values[action_idx] = expected_reward + self.gamma * expected_next_value
                
                # Update value function
                self.V[state_idx] = np.max(action_values)
                delta = max(delta, abs(V_old[state_idx] - self.V[state_idx]))
            
            value_history.append(self.V.copy())
            iteration_history.append(iteration)
            
            if delta < self.theta:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        # Extract greedy policy
        for state_idx in range(self.num_states):
            if self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                continue
                
            action_values = np.zeros(self.num_actions)
            for action_idx in range(self.num_actions):
                expected_reward = self.R[state_idx, action_idx]
                expected_next_value = 0
                for next_state_idx in range(self.num_states):
                    expected_next_value += (self.P[state_idx, action_idx, next_state_idx] * 
                                          self.V[next_state_idx])
                action_values[action_idx] = expected_reward + self.gamma * expected_next_value
            
            best_action = np.argmax(action_values)
            new_policy = np.zeros(self.num_actions)
            new_policy[best_action] = 1.0
            self.policy[state_idx] = new_policy
        
        return value_history, iteration_history
    
    def get_optimal_policy(self) -> np.ndarray:
        """Get the optimal policy."""
        return self.policy.copy()
    
    def get_value_function(self) -> np.ndarray:
        """Get the value function."""
        return self.V.copy()
    
    def evaluate_policy_performance(self, num_episodes: int = 100) -> List[float]:
        """Evaluate the current policy by running episodes."""
        episode_returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                state_idx = self.env.get_state_index(state)
                action = np.argmax(self.policy[state_idx])
                
                next_state, reward, done, _ = self.env.step(action)
                total_return += reward
                
                if done:
                    break
                    
                state = next_state
                steps += 1
            
            episode_returns.append(total_return)
        
        return episode_returns


def run_dp_experiment():
    """Run DP control experiment."""
    print("Running Dynamic Programming Control Experiment")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # Policy Iteration
    print("\n1. Policy Iteration:")
    dp_pi = DPControl(env)
    value_history_pi, iter_history_pi = dp_pi.policy_iteration()
    
    print(f"Final value at start state: {dp_pi.V[env.get_state_index(env.start)]:.2f}")
    
    # Evaluate policy performance
    returns_pi = dp_pi.evaluate_policy_performance(100)
    print(f"Average return over 100 episodes: {np.mean(returns_pi):.2f} ± {np.std(returns_pi):.2f}")
    
    # Value Iteration
    print("\n2. Value Iteration:")
    dp_vi = DPControl(env)
    value_history_vi, iter_history_vi = dp_vi.value_iteration()
    
    print(f"Final value at start state: {dp_vi.V[env.get_state_index(env.start)]:.2f}")
    
    # Evaluate policy performance
    returns_vi = dp_vi.evaluate_policy_performance(100)
    print(f"Average return over 100 episodes: {np.mean(returns_vi):.2f} ± {np.std(returns_vi):.2f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Value function convergence
    ax1 = axes[0, 0]
    start_state_idx = env.get_state_index(env.start)
    pi_values = [vh[start_state_idx] for vh in value_history_pi]
    vi_values = [vh[start_state_idx] for vh in value_history_vi]
    
    ax1.plot(iter_history_pi, pi_values, 'b-', label='Policy Iteration', linewidth=2)
    ax1.plot(iter_history_vi, vi_values, 'r-', label='Value Iteration', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value at Start State')
    ax1.set_title('Value Function Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Policy performance comparison
    ax2 = axes[0, 1]
    ax2.hist(returns_pi, bins=20, alpha=0.7, label='Policy Iteration', color='blue')
    ax2.hist(returns_vi, bins=20, alpha=0.7, label='Value Iteration', color='red')
    ax2.set_xlabel('Episode Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Policy Performance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Value function heatmap (Policy Iteration)
    ax3 = axes[1, 0]
    value_grid_pi = dp_pi.V.reshape(env.height, env.width)
    im1 = ax3.imshow(value_grid_pi, cmap='viridis', aspect='equal')
    ax3.set_title('Value Function (Policy Iteration)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im1, ax=ax3)
    
    # Add start and goal markers
    ax3.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax3.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    # Value function heatmap (Value Iteration)
    ax4 = axes[1, 1]
    value_grid_vi = dp_vi.V.reshape(env.height, env.width)
    im2 = ax4.imshow(value_grid_vi, cmap='viridis', aspect='equal')
    ax4.set_title('Value Function (Value Iteration)')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im2, ax=ax4)
    
    # Add start and goal markers
    ax4.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax4.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    # Render optimal policies
    print("\nOptimal Policy (Policy Iteration):")
    env.render(policy=dp_pi.policy, values=dp_pi.V)
    
    print("\nOptimal Policy (Value Iteration):")
    env.render(policy=dp_vi.policy, values=dp_vi.V)
    
    return dp_pi, dp_vi


if __name__ == "__main__":
    dp_pi, dp_vi = run_dp_experiment()
