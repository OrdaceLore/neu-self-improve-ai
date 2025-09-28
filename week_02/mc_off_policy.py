"""
Monte Carlo Off-Policy Control for Windy Gridworld
Based on Sutton & Barto Chapter 5.7
"""

import numpy as np
from windy_gridworld import WindyGridworld, EpsilonGreedyPolicy, RandomPolicy
from typing import List, Tuple
import matplotlib.pyplot as plt
import random


class MCOffPolicyControl:
    """
    Monte Carlo Off-Policy Control using importance sampling.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # Target policy (greedy with epsilon exploration)
        self.target_policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
        # Behavior policy (random for exploration)
        self.behavior_policy = RandomPolicy(self.num_actions)
        
        # For tracking returns and weights
        self.returns = {}  # Dictionary of (state, action) -> list of returns
        self.weights = {}  # Dictionary of (state, action) -> list of importance weights
        
    def generate_episode(self) -> List[Tuple[int, int, float, float]]:
        """
        Generate an episode following the behavior policy.
        
        Returns:
            List of (state, action, reward, importance_weight) tuples
        """
        episode = []
        state = self.env.reset()
        state_idx = self.env.get_state_index(state)
        
        while True:
            # Select action using behavior policy
            action = self.behavior_policy.select_action(self.Q, state_idx)
            
            # Calculate importance weight
            behavior_prob = self.behavior_policy.get_probability(self.Q, state_idx, action)
            target_prob = self.target_policy.get_probability(self.Q, state_idx, action)
            importance_weight = target_prob / behavior_prob if behavior_prob > 0 else 0
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_state)
            
            episode.append((state_idx, action, reward, importance_weight))
            
            if done:
                break
                
            state_idx = next_state_idx
        
        return episode
    
    def update_q_function_unweighted(self, episode: List[Tuple[int, int, float, float]]) -> None:
        """
        Update Q-function using unweighted importance sampling.
        """
        G = 0
        W = 1  # Cumulative importance weight
        visited_pairs = set()
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state_idx, action, reward, importance_weight = episode[t]
            G = self.gamma * G + reward
            W *= importance_weight
            
            # Only update if this is the first visit to (state, action) in this episode
            if (state_idx, action) not in visited_pairs:
                visited_pairs.add((state_idx, action))
                
                # Add weighted return to the list
                if (state_idx, action) not in self.returns:
                    self.returns[(state_idx, action)] = []
                self.returns[(state_idx, action)].append(W * G)
                
                # Update Q-function (average of weighted returns)
                self.Q[state_idx, action] = np.mean(self.returns[(state_idx, action)])
    
    def update_q_function_weighted(self, episode: List[Tuple[int, int, float, float]]) -> None:
        """
        Update Q-function using weighted importance sampling.
        """
        G = 0
        W = 1  # Cumulative importance weight
        visited_pairs = set()
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state_idx, action, reward, importance_weight = episode[t]
            G = self.gamma * G + reward
            W *= importance_weight
            
            # Only update if this is the first visit to (state, action) in this episode
            if (state_idx, action) not in visited_pairs:
                visited_pairs.add((state_idx, action))
                
                # Add return and weight to the lists
                if (state_idx, action) not in self.returns:
                    self.returns[(state_idx, action)] = []
                    self.weights[(state_idx, action)] = []
                
                self.returns[(state_idx, action)].append(G)
                self.weights[(state_idx, action)].append(W)
                
                # Update Q-function using weighted average
                returns = np.array(self.returns[(state_idx, action)])
                weights = np.array(self.weights[(state_idx, action)])
                
                if np.sum(weights) > 0:
                    self.Q[state_idx, action] = np.sum(weights * returns) / np.sum(weights)
    
    def train_unweighted(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train using unweighted importance sampling.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Monte Carlo Off-Policy Control (Unweighted) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_data = self.generate_episode()
            episode_return = sum(reward for _, _, reward, _ in episode_data)
            episode_returns.append(episode_return)
            
            self.update_q_function_unweighted(episode_data)
            
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def train_weighted(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train using weighted importance sampling.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Monte Carlo Off-Policy Control (Weighted) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_data = self.generate_episode()
            episode_return = sum(reward for _, _, reward, _ in episode_data)
            episode_returns.append(episode_return)
            
            self.update_q_function_weighted(episode_data)
            
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def evaluate_policy(self, num_episodes: int = 100) -> float:
        """
        Evaluate the current target policy.
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


class MCOffPolicyControlOrdinary:
    """
    Monte Carlo Off-Policy Control using ordinary importance sampling.
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 1.0, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_states = env.get_num_states()
        self.num_actions = env.get_num_actions()
        
        # Initialize Q-function
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # Target policy (greedy with epsilon exploration)
        self.target_policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
        # Behavior policy (random for exploration)
        self.behavior_policy = RandomPolicy(self.num_actions)
        
        # For tracking returns
        self.returns = {}  # Dictionary of (state, action) -> list of returns
        
    def generate_episode(self) -> List[Tuple[int, int, float, float]]:
        """Generate an episode following the behavior policy."""
        episode = []
        state = self.env.reset()
        state_idx = self.env.get_state_index(state)
        
        while True:
            action = self.behavior_policy.select_action(self.Q, state_idx)
            
            # Calculate importance weight
            behavior_prob = self.behavior_policy.get_probability(self.Q, state_idx, action)
            target_prob = self.target_policy.get_probability(self.Q, state_idx, action)
            importance_weight = target_prob / behavior_prob if behavior_prob > 0 else 0
            
            next_state, reward, done, _ = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_state)
            
            episode.append((state_idx, action, reward, importance_weight))
            
            if done:
                break
                
            state_idx = next_state_idx
        
        return episode
    
    def update_q_function(self, episode: List[Tuple[int, int, float, float]]) -> None:
        """Update Q-function using ordinary importance sampling."""
        G = 0
        W = 1  # Cumulative importance weight
        visited_pairs = set()
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state_idx, action, reward, importance_weight = episode[t]
            G = self.gamma * G + reward
            W *= importance_weight
            
            # Only update if this is the first visit to (state, action) in this episode
            if (state_idx, action) not in visited_pairs:
                visited_pairs.add((state_idx, action))
                
                # Add weighted return to the list
                if (state_idx, action) not in self.returns:
                    self.returns[(state_idx, action)] = []
                self.returns[(state_idx, action)].append(W * G)
                
                # Update Q-function (average of weighted returns)
                self.Q[state_idx, action] = np.mean(self.returns[(state_idx, action)])
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """Train using ordinary importance sampling."""
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training Monte Carlo Off-Policy Control (Ordinary) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_data = self.generate_episode()
            episode_return = sum(reward for _, _, reward, _ in episode_data)
            episode_returns.append(episode_return)
            
            self.update_q_function(episode_data)
            
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate_policy(10)
                evaluation_returns.append(eval_return)
                print(f"Episode {episode + 1}: Average return = {eval_return:.2f}")
        
        return episode_returns, evaluation_returns
    
    def evaluate_policy(self, num_episodes: int = 100) -> float:
        """Evaluate the current target policy."""
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


def run_mc_off_policy_experiment():
    """Run Monte Carlo off-policy control experiment."""
    print("Running Monte Carlo Off-Policy Control Experiment")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # Ordinary Importance Sampling
    print("\n1. Monte Carlo Off-Policy Control (Ordinary Importance Sampling):")
    mc_ordinary = MCOffPolicyControlOrdinary(env, epsilon=0.1)
    returns_ordinary, eval_returns_ordinary = mc_ordinary.train(num_episodes=1000)
    
    final_eval_ordinary = mc_ordinary.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_ordinary:.2f}")
    
    # Unweighted Importance Sampling
    print("\n2. Monte Carlo Off-Policy Control (Unweighted Importance Sampling):")
    mc_unweighted = MCOffPolicyControl(env, epsilon=0.1)
    returns_unweighted, eval_returns_unweighted = mc_unweighted.train_unweighted(num_episodes=1000)
    
    final_eval_unweighted = mc_unweighted.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_unweighted:.2f}")
    
    # Weighted Importance Sampling
    print("\n3. Monte Carlo Off-Policy Control (Weighted Importance Sampling):")
    mc_weighted = MCOffPolicyControl(env, epsilon=0.1)
    returns_weighted, eval_returns_weighted = mc_weighted.train_weighted(num_episodes=1000)
    
    final_eval_weighted = mc_weighted.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_weighted:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Episode returns
    ax1 = axes[0, 0]
    ax1.plot(returns_ordinary, alpha=0.7, label='Ordinary IS', color='blue')
    ax1.plot(returns_unweighted, alpha=0.7, label='Unweighted IS', color='red')
    ax1.plot(returns_weighted, alpha=0.7, label='Weighted IS', color='green')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Episode Returns During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation returns
    ax2 = axes[0, 1]
    eval_episodes = range(100, 1001, 100)
    ax2.plot(eval_episodes, eval_returns_ordinary, 'b-o', label='Ordinary IS', linewidth=2)
    ax2.plot(eval_episodes, eval_returns_unweighted, 'r-s', label='Unweighted IS', linewidth=2)
    ax2.plot(eval_episodes, eval_returns_weighted, 'g-^', label='Weighted IS', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Return (10 episodes)')
    ax2.set_title('Policy Evaluation During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Q-function heatmap (Ordinary)
    ax3 = axes[1, 0]
    q_max_ordinary = np.max(mc_ordinary.Q, axis=1).reshape(env.height, env.width)
    im1 = ax3.imshow(q_max_ordinary, cmap='viridis', aspect='equal')
    ax3.set_title('Max Q-Values (Ordinary IS)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im1, ax=ax3)
    
    # Add start and goal markers
    ax3.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax3.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    # Q-function heatmap (Weighted)
    ax4 = axes[1, 1]
    q_max_weighted = np.max(mc_weighted.Q, axis=1).reshape(env.height, env.width)
    im2 = ax4.imshow(q_max_weighted, cmap='viridis', aspect='equal')
    ax4.set_title('Max Q-Values (Weighted IS)')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im2, ax=ax4)
    
    # Add start and goal markers
    ax4.plot(env.start[1], env.start[0], 'ws', markersize=10, markeredgecolor='black')
    ax4.plot(env.goal[1], env.goal[0], 'w*', markersize=15, markeredgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    # Render learned policies
    print("\nLearned Policy (Ordinary Importance Sampling):")
    env.render(policy=mc_ordinary.get_greedy_policy(), values=np.max(mc_ordinary.Q, axis=1))
    
    print("\nLearned Policy (Weighted Importance Sampling):")
    env.render(policy=mc_weighted.get_greedy_policy(), values=np.max(mc_weighted.Q, axis=1))
    
    return mc_ordinary, mc_unweighted, mc_weighted


if __name__ == "__main__":
    mc_ordinary, mc_unweighted, mc_weighted = run_mc_off_policy_experiment()
