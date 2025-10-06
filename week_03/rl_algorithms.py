"""
Reinforcement Learning Control Algorithms
Implementation of DP, MC, and TD control methods for Windy Gridworld
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
from windy_gridworld import WindyGridworld

class RLAlgorithm:
    """Base class for RL algorithms"""
    
    def __init__(self, env, gamma=1.0, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = defaultdict(lambda: np.zeros(len(env.actions)))
        
        # Initialize policy (epsilon-greedy)
        self.policy = defaultdict(lambda: np.ones(len(env.actions)) / len(env.actions))
        
    def get_action(self, state, use_policy=True):
        """Get action using epsilon-greedy policy"""
        if use_policy and random.random() > self.epsilon:
            # Greedy action
            return np.argmax(self.Q[state])
        else:
            # Random action
            return random.randint(0, len(self.env.actions) - 1)
    
    def update_policy(self, state):
        """Update policy to be epsilon-greedy with respect to Q"""
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.ones(len(self.env.actions)) * (self.epsilon / len(self.env.actions))
        self.policy[state][best_action] += (1 - self.epsilon)
    
    def get_optimal_policy(self):
        """Get deterministic optimal policy"""
        optimal_policy = {}
        for state in self.Q:
            optimal_policy[state] = np.argmax(self.Q[state])
        return optimal_policy
    
    def evaluate_policy(self, num_episodes=100):
        """Evaluate current policy by running episodes"""
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 1000
            
            while not self.env.is_terminal(state) and steps < max_steps:
                action = self.get_action(state, use_policy=True)
                next_state = self.env.get_next_state(state, self.env.actions[action])
                reward = self.env.get_reward(state, self.env.actions[action], next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards), np.std(total_rewards)

class DPControl(RLAlgorithm):
    """Dynamic Programming Control (Policy Iteration)"""
    
    def __init__(self, env, gamma=1.0, theta=1e-6):
        super().__init__(env, gamma)
        self.theta = theta
    
    def policy_evaluation(self, policy):
        """Evaluate policy using iterative policy evaluation"""
        V = defaultdict(float)
        
        while True:
            delta = 0
            V_old = V.copy()
            
            for state in self.env.states:
                if self.env.is_terminal(state):
                    continue
                
                v = 0
                for action_idx, action in enumerate(self.env.actions):
                    next_state = self.env.get_next_state(state, action)
                    reward = self.env.get_reward(state, action, next_state)
                    v += policy[state][action_idx] * (reward + self.gamma * V_old[next_state])
                
                V[state] = v
                delta = max(delta, abs(V_old[state] - V[state]))
            
            if delta < self.theta:
                break
        
        return V
    
    def policy_improvement(self, V):
        """Improve policy to be greedy with respect to V"""
        policy = {}
        
        for state in self.env.states:
            if self.env.is_terminal(state):
                policy[state] = np.ones(len(self.env.actions)) / len(self.env.actions)
                continue
            
            # Find best action(s)
            action_values = []
            for action in self.env.actions:
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                action_values.append(reward + self.gamma * V[next_state])
            
            # Create greedy policy
            max_value = max(action_values)
            best_actions = [i for i, v in enumerate(action_values) if v == max_value]
            
            policy[state] = np.zeros(len(self.env.actions))
            for action_idx in best_actions:
                policy[state][action_idx] = 1.0 / len(best_actions)
        
        return policy
    
    def train(self, max_iterations=100):
        """Run policy iteration"""
        # Initialize random policy
        policy = {}
        for state in self.env.states:
            policy[state] = np.ones(len(self.env.actions)) / len(self.env.actions)
        
        iteration = 0
        while iteration < max_iterations:
            # Policy evaluation
            V = self.policy_evaluation(policy)
            
            # Policy improvement
            new_policy = self.policy_improvement(V)
            
            # Check for convergence
            policy_stable = True
            for state in policy:
                if not np.allclose(policy[state], new_policy[state]):
                    policy_stable = False
                    break
            
            if policy_stable:
                break
            
            policy = new_policy
            iteration += 1
        
        # Update Q-values based on final policy
        for state in self.env.states:
            if not self.env.is_terminal(state):
                for action_idx, action in enumerate(self.env.actions):
                    next_state = self.env.get_next_state(state, action)
                    reward = self.env.get_reward(state, action, next_state)
                    self.Q[state][action_idx] = reward + self.gamma * V[next_state]
        
        return iteration

class MCOnPolicyControl(RLAlgorithm):
    """Monte Carlo On-Policy Control (SARSA-like)"""
    
    def __init__(self, env, gamma=1.0, epsilon=0.1):
        super().__init__(env, gamma, epsilon)
        self.returns = defaultdict(list)
    
    def generate_episode(self):
        """Generate an episode following current policy"""
        episode = []
        state = self.env.reset()
        
        while not self.env.is_terminal(state):
            action = self.get_action(state, use_policy=True)
            next_state = self.env.get_next_state(state, self.env.actions[action])
            reward = self.env.get_reward(state, self.env.actions[action], next_state)
            
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def train(self, num_episodes=1000):
        """Train using Monte Carlo on-policy control"""
        episode_rewards = []
        
        for episode_num in range(num_episodes):
            episode = self.generate_episode()
            
            # Calculate returns
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # First-visit MC
                if (state, action) not in [(s, a) for s, a, _ in episode[:t]]:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
            
            # Update policy
            for state, action, _ in episode:
                self.update_policy(state)
            
            # Track episode reward
            total_reward = sum(reward for _, _, reward in episode)
            episode_rewards.append(total_reward)
        
        return episode_rewards

class MCOffPolicyControl(RLAlgorithm):
    """Monte Carlo Off-Policy Control using Importance Sampling"""
    
    def __init__(self, env, gamma=1.0, epsilon=0.1):
        super().__init__(env, gamma, epsilon)
        self.returns = defaultdict(list)
        self.behavior_policy = defaultdict(lambda: np.ones(len(env.actions)) / len(env.actions))
    
    def generate_episode(self, policy):
        """Generate episode following given policy"""
        episode = []
        state = self.env.reset()
        
        while not self.env.is_terminal(state):
            action = np.random.choice(len(self.env.actions), p=policy[state])
            next_state = self.env.get_next_state(state, self.env.actions[action])
            reward = self.env.get_reward(state, self.env.actions[action], next_state)
            
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def train(self, num_episodes=1000):
        """Train using Monte Carlo off-policy control"""
        episode_rewards = []
        
        for episode_num in range(num_episodes):
            # Generate episode using behavior policy
            episode = self.generate_episode(self.behavior_policy)
            
            # Calculate returns and importance sampling ratios
            G = 0
            W = 1.0
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # First-visit MC with importance sampling
                if (state, action) not in [(s, a) for s, a, _ in episode[:t]]:
                    # Calculate importance sampling ratio
                    target_prob = self.policy[state][action]
                    behavior_prob = self.behavior_policy[state][action]
                    
                    if behavior_prob > 0:
                        W *= target_prob / behavior_prob
                        self.returns[(state, action)].append(G * W)
                        self.Q[state][action] = np.mean(self.returns[(state, action)])
            
            # Update target policy
            for state, action, _ in episode:
                self.update_policy(state)
            
            # Update behavior policy (epsilon-greedy)
            for state in self.Q:
                self.behavior_policy[state] = self.policy[state].copy()
            
            # Track episode reward
            total_reward = sum(reward for _, _, reward in episode)
            episode_rewards.append(total_reward)
        
        return episode_rewards

class TDOnPolicyControl(RLAlgorithm):
    """TD(0) On-Policy Control (SARSA)"""
    
    def __init__(self, env, gamma=1.0, epsilon=0.1, alpha=0.1):
        super().__init__(env, gamma, epsilon)
        self.alpha = alpha
    
    def train(self, num_episodes=1000):
        """Train using SARSA"""
        episode_rewards = []
        
        for episode_num in range(num_episodes):
            state = self.env.reset()
            action = self.get_action(state, use_policy=True)
            total_reward = 0
            
            while not self.env.is_terminal(state):
                next_state = self.env.get_next_state(state, self.env.actions[action])
                reward = self.env.get_reward(state, self.env.actions[action], next_state)
                next_action = self.get_action(next_state, use_policy=True)
                
                # SARSA update
                td_target = reward + self.gamma * self.Q[next_state][next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                # Update policy
                self.update_policy(state)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        return episode_rewards

class TDOffPolicyControl(RLAlgorithm):
    """TD(0) Off-Policy Control with Importance Sampling"""
    
    def __init__(self, env, gamma=1.0, epsilon=0.1, alpha=0.1, weighted=True):
        super().__init__(env, gamma, epsilon)
        self.alpha = alpha
        self.weighted = weighted  # True for weighted, False for unweighted
        self.behavior_policy = defaultdict(lambda: np.ones(len(env.actions)) / len(env.actions))
    
    def train(self, num_episodes=1000):
        """Train using TD(0) off-policy control"""
        episode_rewards = []
        
        for episode_num in range(num_episodes):
            state = self.env.reset()
            action = self.get_action(state, use_policy=True)
            total_reward = 0
            W = 1.0
            
            while not self.env.is_terminal(state):
                next_state = self.env.get_next_state(state, self.env.actions[action])
                reward = self.env.get_reward(state, self.env.actions[action], next_state)
                
                # Choose next action using behavior policy
                next_action = np.random.choice(len(self.env.actions), 
                                             p=self.behavior_policy[next_state])
                
                # Calculate importance sampling ratio
                target_prob = self.policy[state][action]
                behavior_prob = self.behavior_policy[state][action]
                
                if behavior_prob > 0:
                    W *= target_prob / behavior_prob
                    
                    # Q-learning update with importance sampling
                    if self.weighted:
                        # Weighted importance sampling
                        td_target = reward + self.gamma * self.Q[next_state][next_action]
                        td_error = td_target - self.Q[state][action]
                        self.Q[state][action] += self.alpha * W * td_error
                    else:
                        # Unweighted importance sampling
                        td_target = reward + self.gamma * self.Q[next_state][next_action]
                        td_error = td_target - self.Q[state][action]
                        self.Q[state][action] += self.alpha * td_error
                
                # Update target policy
                self.update_policy(state)
                
                # Update behavior policy
                self.behavior_policy[state] = self.policy[state].copy()
                
                state = next_state
                action = next_action
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        return episode_rewards

def compare_algorithms():
    """Compare all implemented algorithms"""
    print("Comparing RL Algorithms on Windy Gridworld")
    
    # Create environment
    env = WindyGridworld()
    
    # Initialize algorithms
    algorithms = {
        'DP Control': DPControl(env),
        'MC On-Policy': MCOnPolicyControl(env, epsilon=0.1),
        'MC Off-Policy': MCOffPolicyControl(env, epsilon=0.1),
        'TD(0) On-Policy': TDOnPolicyControl(env, epsilon=0.1, alpha=0.1),
        'TD(0) Off-Policy (Unweighted)': TDOffPolicyControl(env, epsilon=0.1, alpha=0.1, weighted=False),
        'TD(0) Off-Policy (Weighted)': TDOffPolicyControl(env, epsilon=0.1, alpha=0.1, weighted=True)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\nTraining {name}...")
        
        if name == 'DP Control':
            iterations = algorithm.train()
            print(f"DP converged in {iterations} iterations")
        else:
            episode_rewards = algorithm.train(num_episodes=1000)
            print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
        
        # Evaluate final policy
        mean_reward, std_reward = algorithm.evaluate_policy(num_episodes=100)
        results[name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'algorithm': algorithm
        }
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    names = list(results.keys())
    means = [results[name]['mean_reward'] for name in names]
    stds = [results[name]['std_reward'] for name in names]
    
    plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title('Algorithm Comparison on Windy Gridworld')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    results = compare_algorithms()
