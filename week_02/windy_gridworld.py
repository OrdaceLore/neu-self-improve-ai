"""
Windy Gridworld Environment
Based on Sutton & Barto Chapter 6.5
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random

class WindyGridworld:
    """
    Windy Gridworld environment as described in Sutton & Barto.
    
    The grid is 7x10 with wind effects in columns 3-9.
    Wind strength: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    """
    
    def __init__(self, height: int = 7, width: int = 10):
        self.height = height
        self.width = width
        
        # Wind strength for each column (0-indexed)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        
        # Actions: up, down, left, right
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # (row, col)
        self.action_names = ['up', 'down', 'left', 'right']
        
        # Start and goal positions
        self.start = (3, 0)  # (row, col)
        self.goal = (3, 7)   # (row, col)
        
        # Current state
        self.current_state = self.start
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 1000
        
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to start state."""
        self.current_state = self.start
        self.episode_steps = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.episode_steps >= self.max_episode_steps:
            return self.current_state, 0, True, {'timeout': True}
        
        row, col = self.current_state
        action_delta = self.actions[action]
        
        # Apply action
        new_row = row + action_delta[0]
        new_col = col + action_delta[1]
        
        # Apply wind effect
        wind_strength = self.wind[col]
        new_row -= wind_strength
        
        # Keep within bounds
        new_row = max(0, min(self.height - 1, new_row))
        new_col = max(0, min(self.width - 1, new_col))
        
        self.current_state = (new_row, new_col)
        self.episode_steps += 1
        
        # Check if goal reached
        done = (self.current_state == self.goal)
        reward = -1 if not done else 0  # -1 for each step, 0 at goal
        
        info = {
            'wind_applied': wind_strength,
            'action_taken': self.action_names[action]
        }
        
        return self.current_state, reward, done, info
    
    def get_state_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to linear index."""
        row, col = state
        return row * self.width + col
    
    def get_state_from_index(self, index: int) -> Tuple[int, int]:
        """Convert linear index to (row, col) state."""
        row = index // self.width
        col = index % self.width
        return (row, col)
    
    def get_num_states(self) -> int:
        """Get total number of states."""
        return self.height * self.width
    
    def get_num_actions(self) -> int:
        """Get total number of actions."""
        return len(self.actions)
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal (goal)."""
        return state == self.goal
    
    def render(self, policy: Optional[np.ndarray] = None, values: Optional[np.ndarray] = None):
        """
        Render the gridworld with optional policy and value function.
        
        Args:
            policy: Policy array of shape (num_states, num_actions)
            values: Value function array of shape (num_states,)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grid
        grid = np.zeros((self.height, self.width))
        
        # Mark start and goal
        grid[self.start] = 1
        grid[self.goal] = 2
        
        # Show wind strength
        wind_display = np.array(self.wind).reshape(1, -1)
        
        # Create subplot layout
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
        ax_main = fig.add_subplot(gs[0, :])
        ax_wind = fig.add_subplot(gs[1, 0])
        ax_legend = fig.add_subplot(gs[1, 1])
        
        # Main grid
        im = ax_main.imshow(grid, cmap='viridis', aspect='equal')
        ax_main.set_title('Windy Gridworld')
        ax_main.set_xlabel('Column')
        ax_main.set_ylabel('Row')
        
        # Add wind information
        wind_colors = ['white', 'lightblue', 'blue']
        for col in range(self.width):
            wind_strength = self.wind[col]
            if wind_strength > 0:
                for row in range(self.height):
                    rect = plt.Rectangle((col-0.4, row-0.4), 0.8, 0.8, 
                                       facecolor=wind_colors[wind_strength], 
                                       alpha=0.3, edgecolor='black')
                    ax_main.add_patch(rect)
        
        # Add start and goal labels
        ax_main.text(self.start[1], self.start[0], 'S', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white')
        ax_main.text(self.goal[1], self.goal[0], 'G', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Add policy arrows if provided
        if policy is not None:
            arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
            for state_idx in range(self.get_num_states()):
                if not self.is_terminal(self.get_state_from_index(state_idx)):
                    row, col = self.get_state_from_index(state_idx)
                    best_action = np.argmax(policy[state_idx])
                    ax_main.text(col, row, arrow_map[best_action], ha='center', va='center',
                               fontsize=12, fontweight='bold')
        
        # Wind strength bar
        wind_display = np.array(self.wind).reshape(1, -1)
        ax_wind.imshow(wind_display, cmap='Blues', aspect='auto')
        ax_wind.set_title('Wind Strength')
        ax_wind.set_xlabel('Column')
        ax_wind.set_xticks(range(self.width))
        ax_wind.set_yticks([])
        
        # Add wind strength values
        for col in range(self.width):
            ax_wind.text(col, 0, str(self.wind[col]), ha='center', va='center',
                        fontweight='bold', color='white' if self.wind[col] > 1 else 'black')
        
        # Legend
        ax_legend.axis('off')
        legend_text = "Legend:\n"
        legend_text += "S = Start\n"
        legend_text += "G = Goal\n"
        legend_text += "↑↓←→ = Policy\n"
        legend_text += "Blue = Wind"
        ax_legend.text(0.1, 0.5, legend_text, fontsize=12, va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Print value function if provided
        if values is not None:
            print("\nValue Function:")
            value_grid = values.reshape(self.height, self.width)
            for row in range(self.height):
                print(" ".join([f"{value_grid[row, col]:6.2f}" for col in range(self.width)]))


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy for exploration."""
    
    def __init__(self, num_actions: int, epsilon: float = 0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
    
    def select_action(self, q_values: np.ndarray, state: int) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(q_values[state])
    
    def get_probability(self, q_values: np.ndarray, state: int, action: int) -> float:
        """Get action probability under epsilon-greedy policy."""
        best_action = np.argmax(q_values[state])
        if action == best_action:
            return 1.0 - self.epsilon + self.epsilon / self.num_actions
        else:
            return self.epsilon / self.num_actions


class RandomPolicy:
    """Random policy for behavior in off-policy methods."""
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
    
    def select_action(self, q_values: np.ndarray, state: int) -> int:
        """Select random action."""
        return random.randint(0, self.num_actions - 1)
    
    def get_probability(self, q_values: np.ndarray, state: int, action: int) -> float:
        """Get action probability under random policy."""
        return 1.0 / self.num_actions


if __name__ == "__main__":
    # Test the environment
    env = WindyGridworld()
    
    print("Testing Windy Gridworld Environment")
    print(f"Grid size: {env.height}x{env.width}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Wind: {env.wind}")
    print(f"Number of states: {env.get_num_states()}")
    print(f"Number of actions: {env.get_num_actions()}")
    
    # Test a few steps
    state = env.reset()
    print(f"\nInitial state: {state}")
    
    for step in range(5):
        action = random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        print(f"Step {step+1}: Action {env.action_names[action]}, "
              f"State {next_state}, Reward {reward}, Done {done}")
        if done:
            break
        state = next_state
    
    # Render the environment
    env.render()
