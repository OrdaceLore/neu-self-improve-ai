"""
Monte Carlo On-Policy Control for Windy Gridworld
Based on Sutton & Barto Chapter 5.4
"""

import numpy as np
from windy_gridworld import WindyGridworld, EpsilonGreedyPolicy
from typing import List, Tuple
import matplotlib.pyplot as plt
import random


class MCOnPolicyControl:
    """
    Monte Carlo On-Policy Control using epsilon-greedy policy.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function and policy
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
        # For tracking returns
        self.returns = {}  # Dictionary of (state, action) -> list of returns
        self.episode_count = 0
        
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Generate an episode following the current policy.
        
        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()
        state_idx = self.env.get_state_index(state)
        
        while True:
            # Select action using current policy
            action = self.policy.select_action(self.Q, state_idx)
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_state)
            
            episode.append((state_idx, action, reward))
            
            if done:
                break
                
            state_idx = next_state_idx
        
        return episode
    
    def update_q_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Update Q-function using the episode.
        """
        # Calculate returns for each state-action pair
        G = 0
        visited_pairs = set()
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state_idx, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Only update if this is the first visit to (state, action) in this episode
            if (state_idx, action) not in visited_pairs:
                visited_pairs.add((state_idx, action))
                
                # Add return to the list
                if (state_idx, action) not in self.returns:
                    self.returns[(state_idx, action)] = []
                self.returns[(state_idx, action)].append(G)
                
                # Update Q-function (average of all returns)
                self.Q[state_idx, action] = np.mean(self.returns[(state_idx, action)])
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train the agent using Monte Carlo on-policy control.
        
        Args:
            num_episodes: Number of training episodes
            eval_interval: Interval for evaluation
            
        Returns:
            (episode_returns, evaluation_returns)
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Monte Carlo On-Policy Control for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Generate episode
            episode_data = self.generate_episode()
            
            # Calculate episode return
            episode_return = sum(reward for _, _, reward in episode_data)
            episode_returns.append(episode_return)
            
            # Update Q-function
            self.update_q_function(episode_data)
            
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
    
    def get_policy(self) -> np.ndarray:
        """
        Get the current policy as a probability matrix.
        """
        policy_matrix = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            for action in range(self.num_actions):
                policy_matrix[state_idx, action] = self.policy.get_probability(
                    self.Q, state_idx, action
                )
        
        return policy_matrix
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        Get the greedy policy (for visualization).
        """
        greedy_policy = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            if not self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                best_action = np.argmax(self.Q[state_idx])
                greedy_policy[state_idx, best_action] = 1.0
        
        return greedy_policy


class MCEveryVisitControl:
    """
    Monte Carlo Every-Visit On-Policy Control.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function and policy
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
        # For tracking returns
        self.returns = {}  # Dictionary of (state, action) -> list of returns
        
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """Generate an episode following the current policy."""
        episode = []
        state = self.env.reset()
        state_idx = self.env.get_state_index(state)
        
        while True:
            action = self.policy.select_action(self.Q, state_idx)
            next_state, reward, done, _ = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_state)
            
            episode.append((state_idx, action, reward))
            
            if done:
                break
                
            state_idx = next_state_idx
        
        return episode
    
    def update_q_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """Update Q-function using every-visit Monte Carlo."""
        # Calculate returns for each state-action pair
        G = 0
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state_idx, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Add return to the list (every visit)
            if (state_idx, action) not in self.returns:
                self.returns[(state_idx, action)] = []
            self.returns[(state_idx, action)].append(G)
            
            # Update Q-function (average of all returns)
            self.Q[state_idx, action] = np.mean(self.returns[(state_idx, action)])
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """Train the agent using every-visit Monte Carlo."""
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Monte Carlo Every-Visit On-Policy Control for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_data = self.generate_episode()
            episode_return = sum(reward for _, _, reward in episode_data)
            episode_returns.append(episode_return)
            
            self.update_q_function(episode_data)
            
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def evaluate_policy(self, num_episodes: int = 100) -> float:
        """Evaluate the current policy."""
        total_returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            total_return = 0
            steps = 0
            
            while steps < 1000:
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
        """Get the greedy policy."""
        greedy_policy = np.zeros((self.num_states, self.num_actions))
        
        for state_idx in range(self.num_states):
            if not self.env.is_terminal(self.env.get_state_from_index(state_idx)):
                best_action = np.argmax(self.Q[state_idx])
                greedy_policy[state_idx, best_action] = 1.0
        
        return greedy_policy


def run_mc_on_policy_experiment():
    """Run Monte Carlo on-policy control experiment."""
    print("Running Monte Carlo On-Policy Control Experiment")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # First-Visit Monte Carlo
    print("\n1. First-Visit Monte Carlo On-Policy Control:")
    mc_first_visit = MCOnPolicyControl(env, epsilon=0.1)
    returns_first, eval_returns_first = mc_first_visit.train(num_episodes=1000)
    
    final_eval_first = mc_first_visit.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_first:.2f}")
    
    # Every-Visit Monte Carlo
    print("\n2. Every-Visit Monte Carlo On-Policy Control:")
    mc_every_visit = MCEveryVisitControl(env, epsilon=0.1)
    returns_every, eval_returns_every = mc_every_visit.train(num_episodes=1000)
    
    final_eval_every = mc_every_visit.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_every:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Episode returns
    ax1 = axes[0, 0]
    ax1.plot(returns_first, alpha=0.7, label='First-Visit MC', color='blue')
    ax1.plot(returns_every, alpha=0.7, label='Every-Visit MC', color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Episode Returns During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation returns
    ax2 = axes[0, 1]
    eval_episodes = range(100, 1001, 100)
    ax2.plot(eval_episodes, eval_returns_first, 'b-o', label='First-Visit MC', linewidth=2)
    ax2.plot(eval_episodes, eval_returns_every, 'r-s', label='Every-Visit MC', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Return (10 episodes)')
    ax2.set_title('Policy Evaluation During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Q-function heatmap (First-Visit)
    ax3 = axes[1, 0]
    q_max_first = np.max(mc_first_visit.Q, axis=1).reshape(env.height, env.width)
    im1 = ax3.imshow(q_max_first, cmap='viridis', aspect='equal')
    ax3.set_title('Max Q-Values (First-Visit MC)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im1, ax=ax3)
    
    # Add start and goal markers
    ax3.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax3.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    # Q-function heatmap (Every-Visit)
    ax4 = axes[1, 1]
    q_max_every = np.max(mc_every_visit.Q, axis=1).reshape(env.height, env.width)
    im2 = ax4.imshow(q_max_every, cmap='viridis', aspect='equal')
    ax4.set_title('Max Q-Values (Every-Visit MC)')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im2, ax=ax4)
    
    # Add start and goal markers
    ax4.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax4.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    # Render learned policies
    print("\nLearned Policy (First-Visit MC):")
    env.render(policy=mc_first_visit.get_greedy_policy(), values=np.max(mc_first_visit.Q, axis=1))
    
    print("\nLearned Policy (Every-Visit MC):")
    env.render(policy=mc_every_visit.get_greedy_policy(), values=np.max(mc_every_visit.Q, axis=1))
    
    return mc_first_visit, mc_every_visit


if __name__ == "__main__":
    mc_first, mc_every = run_mc_on_policy_experiment()
