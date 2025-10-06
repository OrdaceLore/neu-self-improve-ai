"""
Replication of Figure 4.1 from Sutton & Barto
Policy Iteration in Gridworld Environment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

class Gridworld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size-1, size-1)  # Bottom-right corner
        self.terminal = (0, 0)  # Top-left corner
        
        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['↑', '↓', '←', '→']
        
        # Initialize value function
        self.V = np.zeros((size, size))
        
    def is_terminal(self, state):
        return state == self.goal or state == self.terminal
    
    def get_next_state(self, state, action):
        """Get next state given current state and action"""
        if self.is_terminal(state):
            return state
        
        next_state = (state[0] + action[0], state[1] + action[1])
        
        # Check bounds
        if (0 <= next_state[0] < self.size and 
            0 <= next_state[1] < self.size):
            return next_state
        else:
            return state  # Stay in place if hitting wall
    
    def get_reward(self, state, action, next_state):
        """Get reward for transition"""
        if next_state == self.goal:
            return 0  # Goal reward
        else:
            return -1  # Step cost

class PolicyIteration:
    def __init__(self, gridworld, gamma=1.0, theta=1e-6):
        self.gw = gridworld
        self.gamma = gamma
        self.theta = theta
        self.size = gridworld.size
        
    def policy_evaluation(self, policy):
        """Evaluate policy using iterative policy evaluation"""
        V = np.zeros((self.size, self.size))
        
        while True:
            delta = 0
            V_old = V.copy()
            
            for i in range(self.size):
                for j in range(self.size):
                    state = (i, j)
                    if self.gw.is_terminal(state):
                        continue
                    
                    v = 0
                    for action_idx, action in enumerate(self.gw.actions):
                        next_state = self.gw.get_next_state(state, action)
                        reward = self.gw.get_reward(state, action, next_state)
                        v += policy[state][action_idx] * (reward + self.gamma * V_old[next_state])
                    
                    V[state] = v
                    delta = max(delta, abs(V_old[state] - V[state]))
            
            if delta < self.theta:
                break
        
        return V
    
    def policy_improvement(self, V):
        """Improve policy to be greedy with respect to V"""
        policy = {}
        
        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if self.gw.is_terminal(state):
                    policy[state] = [0.25, 0.25, 0.25, 0.25]  # Random for terminal
                    continue
                
                # Find best action(s)
                action_values = []
                for action in self.gw.actions:
                    next_state = self.gw.get_next_state(state, action)
                    reward = self.gw.get_reward(state, action, next_state)
                    action_values.append(reward + self.gamma * V[next_state])
                
                # Create greedy policy
                max_value = max(action_values)
                best_actions = [i for i, v in enumerate(action_values) if v == max_value]
                
                policy[state] = [1.0/len(best_actions) if i in best_actions else 0.0 
                               for i in range(4)]
        
        return policy
    
    def policy_iteration(self, max_iterations=100):
        """Run policy iteration algorithm"""
        # Initialize random policy
        policy = {}
        for i in range(self.size):
            for j in range(self.size):
                policy[(i, j)] = [0.25, 0.25, 0.25, 0.25]
        
        iteration = 0
        V_history = []
        policy_history = []
        
        while iteration < max_iterations:
            # Policy evaluation
            V = self.policy_evaluation(policy)
            V_history.append(V.copy())
            
            # Policy improvement
            new_policy = self.policy_improvement(V)
            policy_history.append({k: v.copy() for k, v in new_policy.items()})
            
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
        
        return V_history, policy_history, iteration

def visualize_figure_4_1(V_history, policy_history, save_path=None):
    """Replicate Figure 4.1 visualization"""
    n_iterations = len(V_history)
    fig, axes = plt.subplots(n_iterations, 2, figsize=(12, 3*n_iterations))
    
    if n_iterations == 1:
        axes = axes.reshape(1, -1)
    
    for k in range(n_iterations):
        V = V_history[k]
        policy = policy_history[k]
        
        # Value function heatmap
        im1 = axes[k, 0].imshow(V, cmap='viridis', aspect='equal')
        axes[k, 0].set_title(f'k={k} - v_k for the random policy')
        
        # Add value annotations
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if (i, j) == (0, 0) or (i, j) == (V.shape[0]-1, V.shape[1]-1):
                    axes[k, 0].text(j, i, '', ha='center', va='center', 
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='gray'))
                else:
                    axes[k, 0].text(j, i, f'{V[i, j]:.1f}', ha='center', va='center',
                                  color='white' if V[i, j] < -5 else 'black')
        
        # Policy visualization
        axes[k, 1].set_xlim(-0.5, V.shape[1]-0.5)
        axes[k, 1].set_ylim(-0.5, V.shape[0]-0.5)
        axes[k, 1].set_title(f'k={k} - greedy policy w.r.t. v_k')
        axes[k, 1].invert_yaxis()
        
        # Draw grid
        for i in range(V.shape[0] + 1):
            axes[k, 1].axhline(i-0.5, color='black', linewidth=0.5)
        for j in range(V.shape[1] + 1):
            axes[k, 1].axvline(j-0.5, color='black', linewidth=0.5)
        
        # Draw policy arrows
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                state = (i, j)
                if state == (0, 0) or state == (V.shape[0]-1, V.shape[1]-1):
                    # Terminal states - gray background
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='gray', alpha=0.5)
                    axes[k, 1].add_patch(rect)
                else:
                    # Draw arrows for actions with probability > 0
                    for action_idx, prob in enumerate(policy[state]):
                        if prob > 0:
                            action = [(-1, 0), (1, 0), (0, -1), (0, 1)][action_idx]
                            arrow = FancyArrowPatch((j, i), 
                                                  (j + action[1]*0.3, i + action[0]*0.3),
                                                  arrowstyle='->', mutation_scale=15,
                                                  color='red', linewidth=2)
                            axes[k, 1].add_patch(arrow)
        
        axes[k, 0].set_xticks(range(V.shape[1]))
        axes[k, 0].set_yticks(range(V.shape[0]))
        axes[k, 1].set_xticks(range(V.shape[1]))
        axes[k, 1].set_yticks(range(V.shape[0]))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run the Figure 4.1 replication"""
    print("Replicating Figure 4.1: Policy Iteration in Gridworld")
    
    # Create gridworld and policy iteration
    gw = Gridworld(size=4)
    pi = PolicyIteration(gw)
    
    # Run policy iteration
    V_history, policy_history, iterations = pi.policy_iteration()
    
    print(f"Policy iteration converged in {iterations} iterations")
    print(f"Final value function:\n{V_history[-1]}")
    
    # Visualize results
    visualize_figure_4_1(V_history, policy_history, 
                        save_path='figure_4_1_replication.png')
    
    return V_history, policy_history

if __name__ == "__main__":
    V_history, policy_history = main()
