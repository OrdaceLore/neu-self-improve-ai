"""
TD(0) On-Policy Control for Windy Gridworld
Based on Sutton & Barto Chapter 6.4
"""

import numpy as np
from windy_gridworld import WindyGridworld, EpsilonGreedyPolicy
from typing import List, Tuple
import matplotlib.pyplot as plt
import random


class TDOnPolicyControl:
    """
    TD(0) On-Policy Control using Sarsa algorithm.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # Policy
        self.policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
    def sarsa_update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> None:
        """
        Update Q-function using Sarsa algorithm.
        """
        # Sarsa update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        current_q = self.Q[state, action]
        next_q = self.Q[next_state, next_action]
        target = reward + self.gamma * next_q
        self.Q[state, action] = current_q + self.alpha * (target - current_q)
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train the agent using Sarsa.
        
        Returns:
            (episode_returns, evaluation_returns)
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training TD(0) On-Policy Control (Sarsa) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Initialize episode
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            action = self.policy.select_action(self.Q, state_idx)
            
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_return += reward
                
                if done:
                    # Terminal state: Q(S',A') = 0
                    self.Q[state_idx, action] += self.alpha * (reward - self.Q[state_idx, action])
                    break
                else:
                    # Select next action
                    next_action = self.policy.select_action(self.Q, next_state_idx)
                    
                    # Sarsa update
                    self.sarsa_update(state_idx, action, reward, next_state_idx, next_action)
                    
                    # Move to next state
                    state_idx = next_state_idx
                    action = next_action
                
                steps += 1
            
            episode_returns.append(total_return)
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def evaluate_policy(self, num_episodes: int = 100) -> float:
        """
        Evaluate the current policy by running episodes greedily.
        """
        total_returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Use greedy policy for evaluation
                action = np.argmax(self.Q[state_idx])
                
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                
                total_return += reward
                
                if done:
                    break
                    
                state_idx = next_state_idx
                steps += 1
            
            total_returns.append(total_return)
        
        return np.mean(total_returns)
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        Get the greedy policy.
        """
        greedy_policy = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            if not self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                best_action = np.argmax(self.Q[state_idx])
                greedy_policy[state_idx, best_action] = 1.0
        
        return greedy_policy


class QLearning:
    """
    Q-Learning algorithm (off-policy TD control).
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # Policy for action selection
        self.policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
    def q_learning_update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-function using Q-learning algorithm.
        """
        # Q-learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state, :])
        target = reward + self.gamma * max_next_q
        self.Q[state, action] = current_q + self.alpha * (target - current_q)
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train the agent using Q-learning.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Q-Learning for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Initialize episode
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Select action using epsilon-greedy policy
                action = self.policy.select_action(self.Q, state_idx)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_return += reward
                
                # Q-learning update
                self.q_learning_update(state_idx, action, reward, next_state_idx)
                
                if done:
                    break
                    
                # Move to next state
                state_idx = next_state_idx
                steps += 1
            
            episode_returns.append(total_return)
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def evaluate_policy(self, num_episodes: int = 100) -> float:
        """
        Evaluate the current policy by running episodes greedily.
        """
        total_returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Use greedy policy for evaluation
                action = np.argmax(self.Q[state_idx])
                
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                
                total_return += reward
                
                if done:
                    break
                    
                state_idx = next_state_idx
                steps += 1
            
            total_returns.append(total_return)
        
        return np.mean(total_returns)
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        Get the greedy policy.
        """
        greedy_policy = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            if not self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                best_action = np.argmax(self.Q[state_idx])
                greedy_policy[state_idx, best_action] = 1.0
        
        return greedy_policy


class ExpectedSarsa:
    """
    Expected Sarsa algorithm.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # Policy
        self.policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
    def expected_sarsa_update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-function using Expected Sarsa algorithm.
        """
        # Expected Sarsa update: Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]
        current_q = self.Q[state, action]
        
        # Calculate expected value of next state
        expected_next_q = 0
        for next_action in range(self.num_actions):
            action_prob = self.policy.get_probability(self.Q, next_state, next_action)
            expected_next_q += action_prob * self.Q[next_state, next_action]
        
        target = reward + self.gamma * expected_next_q
        self.Q[state, action] = current_q + self.alpha * (target - current_q)
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train the agent using Expected Sarsa.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Expected Sarsa for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Initialize episode
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Select action using epsilon-greedy policy
                action = self.policy.select_action(self.Q, state_idx)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_return += reward
                
                if done:
                    # Terminal state: expected value is 0
                    self.Q[state_idx, action] += self.alpha * (reward - self.Q[state_idx, action])
                    break
                else:
                    # Expected Sarsa update
                    self.expected_sarsa_update(state_idx, action, reward, next_state_idx)
                    
                    # Move to next state
                    state_idx = next_state_idx
                
                steps += 1
            
            episode_returns.append(total_return)
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def evaluate_policy(self, num_episodes: int = 100) -> float:
        """
        Evaluate the current policy by running episodes greedily.
        """
        total_returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            total_return = 0
            steps = 0
            
            while steps < 1000:  # Max steps per episode
                # Use greedy policy for evaluation
                action = np.argmax(self.Q[state_idx])
                
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                
                total_return += reward
                
                if done:
                    break
                    
                state_idx = next_state_idx
                steps += 1
            
            total_returns.append(total_return)
        
        return np.mean(total_returns)
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        Get the greedy policy.
        """
        greedy_policy = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            if not self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                best_action = np.argmax(self.Q[state_idx])
                greedy_policy[state_idx, best_action] = 1.0
        
        return greedy_policy


def run_td_on_policy_experiment():
    """Run TD(0) on-policy control experiment."""
    print("Running TD(0) On-Policy Control Experiment")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # Sarsa
    print("\n1. Sarsa (On-Policy TD Control):")
    sarsa = TDOnPolicyControl(env, alpha=0.1, epsilon=0.1)
    returns_sarsa, eval_returns_sarsa = sarsa.train(num_episodes=1000)
    
    final_eval_sarsa = sarsa.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_sarsa:.2f}")
    
    # Q-Learning
    print("\n2. Q-Learning (Off-Policy TD Control):")
    q_learning = QLearning(env, alpha=0.1, epsilon=0.1)
    returns_q_learning, eval_returns_q_learning = q_learning.train(num_episodes=1000)
    
    final_eval_q_learning = q_learning.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_q_learning:.2f}")
    
    # Expected Sarsa
    print("\n3. Expected Sarsa:")
    expected_sarsa = ExpectedSarsa(env, alpha=0.1, epsilon=0.1)
    returns_expected_sarsa, eval_returns_expected_sarsa = expected_sarsa.train(num_episodes=1000)
    
    final_eval_expected_sarsa = expected_sarsa.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_expected_sarsa:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Episode returns
    ax1 = axes[0, 0]
    ax1.plot(returns_sarsa, alpha=0.7, label='Sarsa', color='blue')
    ax1.plot(returns_q_learning, alpha=0.7, label='Q-Learning', color='red')
    ax1.plot(returns_expected_sarsa, alpha=0.7, label='Expected Sarsa', color='green')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Episode Returns During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation returns
    ax2 = axes[0, 1]
    eval_episodes = range(100, 1001, 100)
    ax2.plot(eval_episodes, eval_returns_sarsa, 'b-o', label='Sarsa', linewidth=2)
    ax2.plot(eval_episodes, eval_returns_q_learning, 'r-s', label='Q-Learning', linewidth=2)
    ax2.plot(eval_episodes, eval_returns_expected_sarsa, 'g-^', label='Expected Sarsa', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Return (10 episodes)')
    ax2.set_title('Policy Evaluation During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Q-function heatmap (Sarsa)
    ax3 = axes[1, 0]
    q_max_sarsa = np.max(sarsa.Q, axis=1).reshape(env.height, env.width)
    im1 = ax3.imshow(q_max_sarsa, cmap='viridis', aspect='equal')
    ax3.set_title('Max Q-Values (Sarsa)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im1, ax=ax3)
    
    # Add start and goal markers
    ax3.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax3.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    # Q-function heatmap (Q-Learning)
    ax4 = axes[1, 1]
    q_max_q_learning = np.max(q_learning.Q, axis=1).reshape(env.height, env.width)
    im2 = ax4.imshow(q_max_q_learning, cmap='viridis', aspect='equal')
    ax4.set_title('Max Q-Values (Q-Learning)')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im2, ax=ax4)
    
    # Add start and goal markers
    ax4.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax4.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    # Render learned policies
    print("\nLearned Policy (Sarsa):")
    env.render(policy=sarsa.get_greedy_policy(), values=np.max(sarsa.Q, axis=1))
    
    print("\nLearned Policy (Q-Learning):")
    env.render(policy=q_learning.get_greedy_policy(), values=np.max(q_learning.Q, axis=1))
    
    return sarsa, q_learning, expected_sarsa


if __name__ == "__main__":
    sarsa, q_learning, expected_sarsa = run_td_on_policy_experiment()
