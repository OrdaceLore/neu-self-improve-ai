

import numpy as np
from windy_gridworld import WindyGridworld, EpsilonGreedyPolicy, RandomPolicy
from typing import List, Tuple
import matplotlib.pyplot as plt
import random


class TDOffPolicyControl:
    """
    TD(0) Off-Policy Control using importance sampling.
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
        
        # Target policy (greedy with epsilon exploration)
        self.target_policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
        # Behavior policy (random for exploration)
        self.behavior_policy = RandomPolicy(self.num_actions)
        
    def td_off_policy_update_unweighted(self, state: int, action: int, reward: float, 
                                       next_state: int, importance_weight: float) -> None:
        """
        Update Q-function using TD(0) with unweighted importance sampling.
        """
        # TD(0) off-policy update: Q(S,A) ← Q(S,A) + α * W * [R + γ * max_a Q(S',a) - Q(S,A)]
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state, :])
        target = reward + self.gamma * max_next_q
        self.Q[state, action] = current_q + self.alpha * importance_weight * (target - current_q)
    
    def td_off_policy_update_weighted(self, state: int, action: int, reward: float, 
                                     next_state: int, importance_weight: float, 
                                     cumulative_weight: float) -> None:
        """
        Update Q-function using TD(0) with weighted importance sampling.
        """
        # TD(0) off-policy update with weighted importance sampling
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state, :])
        target = reward + self.gamma * max_next_q
        
        if cumulative_weight > 0:
            self.Q[state, action] = current_q + (self.alpha * importance_weight / cumulative_weight) * (target - current_q)
    
    def train_unweighted(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train using TD(0) with unweighted importance sampling.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training TD(0) Off-Policy Control (Unweighted IS) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Initialize episode
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            
            total_return = 0
            steps = 0
            cumulative_weight = 1.0
            
            while steps < 1000:  # Max steps per episode
                # Select action using behavior policy
                action = self.behavior_policy.select_action(self.Q, state_idx)
                
                # Calculate importance weight
                behavior_prob = self.behavior_policy.get_probability(self.Q, state_idx, action)
                target_prob = self.target_policy.get_probability(self.Q, state_idx, action)
                importance_weight = target_prob / behavior_prob if behavior_prob > 0 else 0
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_return += reward
                
                # Update cumulative weight
                cumulative_weight *= importance_weight
                
                # TD(0) off-policy update
                self.td_off_policy_update_unweighted(state_idx, action, reward, next_state_idx, importance_weight)
                
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
    
    def train_weighted(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train using TD(0) with weighted importance sampling.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training TD(0) Off-Policy Control (Weighted IS) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Initialize episode
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            
            total_return = 0
            steps = 0
            cumulative_weight = 1.0
            
            while steps < 1000:  # Max steps per episode
                # Select action using behavior policy
                action = self.behavior_policy.select_action(self.Q, state_idx)
                
                # Calculate importance weight
                behavior_prob = self.behavior_policy.get_probability(self.Q, state_idx, action)
                target_prob = self.target_policy.get_probability(self.Q, state_idx, action)
                importance_weight = target_prob / behavior_prob if behavior_prob > 0 else 0
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_return += reward
                
                # Update cumulative weight
                cumulative_weight *= importance_weight
                
                # TD(0) off-policy update with weighted importance sampling
                self.td_off_policy_update_weighted(state_idx, action, reward, next_state_idx, 
                                                 importance_weight, cumulative_weight)
                
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


class TDOffPolicyControlOrdinary:
    """
    TD(0) Off-Policy Control using ordinary importance sampling.
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
        
        # Target policy (greedy with epsilon exploration)
        self.target_policy = EpsilonGreedyPolicy(self.num_actions, epsilon)
        
        # Behavior policy (random for exploration)
        self.behavior_policy = RandomPolicy(self.num_actions)
        
    def td_off_policy_update(self, state: int, action: int, reward: float, 
                            next_state: int, importance_weight: float) -> None:
        """
        Update Q-function using TD(0) with ordinary importance sampling.
        """
        # TD(0) off-policy update: Q(S,A) ← Q(S,A) + α * W * [R + γ * max_a Q(S',a) - Q(S,A)]
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state, :])
        target = reward + self.gamma * max_next_q
        self.Q[state, action] = current_q + self.alpha * importance_weight * (target - current_q)
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train using TD(0) with ordinary importance sampling.
        """
        episode_returns = []
        evaluation_returns = []
        
        print(f"Training TD(0) Off-Policy Control (Ordinary IS) for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Initialize episode
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            
            total_return = 0
            steps = 0
            cumulative_weight = 1.0
            
            while steps < 1000:  # Max steps per episode
                # Select action using behavior policy
                action = self.behavior_policy.select_action(self.Q, state_idx)
                
                # Calculate importance weight
                behavior_prob = self.behavior_policy.get_probability(self.Q, state_idx, action)
                target_prob = self.target_policy.get_probability(self.Q, state_idx, action)
                importance_weight = target_prob / behavior_prob if behavior_prob > 0 else 0
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_return += reward
                
                # Update cumulative weight
                cumulative_weight *= importance_weight
                
                # TD(0) off-policy update
                self.td_off_policy_update(state_idx, action, reward, next_state_idx, importance_weight)
                
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


def run_td_off_policy_experiment():
    """Run TD(0) off-policy control experiment."""
    print("Running TD(0) Off-Policy Control Experiment")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # Ordinary Importance Sampling
    print("\n1. TD(0) Off-Policy Control (Ordinary Importance Sampling):")
    td_ordinary = TDOffPolicyControlOrdinary(env, alpha=0.1, epsilon=0.1)
    returns_ordinary, eval_returns_ordinary = td_ordinary.train(num_episodes=1000)
    
    final_eval_ordinary = td_ordinary.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_ordinary:.2f}")
    
    # Unweighted Importance Sampling
    print("\n2. TD(0) Off-Policy Control (Unweighted Importance Sampling):")
    td_unweighted = TDOffPolicyControl(env, alpha=0.1, epsilon=0.1)
    returns_unweighted, eval_returns_unweighted = td_unweighted.train_unweighted(num_episodes=1000)
    
    final_eval_unweighted = td_unweighted.evaluate_policy(100)
    print(f"Final evaluation (100 episodes): {final_eval_unweighted:.2f}")
    
    # Weighted Importance Sampling
    print("\n3. TD(0) Off-Policy Control (Weighted Importance Sampling):")
    td_weighted = TDOffPolicyControl(env, alpha=0.1, epsilon=0.1)
    returns_weighted, eval_returns_weighted = td_weighted.train_weighted(num_episodes=1000)
    
    final_eval_weighted = td_weighted.evaluate_policy(100)
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
    q_max_ordinary = np.max(td_ordinary.Q, axis=1).reshape(env.height, env.width)
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
    q_max_weighted = np.max(td_weighted.Q, axis=1).reshape(env.height, env.width)
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
    env.render(policy=td_ordinary.get_greedy_policy(), values=np.max(td_ordinary.Q, axis=1))
    
    print("\nLearned Policy (Weighted Importance Sampling):")
    env.render(policy=td_weighted.get_greedy_policy(), values=np.max(td_weighted.Q, axis=1))
    
    return td_ordinary, td_unweighted, td_weighted


if __name__ == "__main__":
    td_ordinary, td_unweighted, td_weighted = run_td_off_policy_experiment()
