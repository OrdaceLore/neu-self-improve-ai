"""
Windy Gridworld Environment
Based on Example 6.5 from Sutton & Barto
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

class WindyGridworld:
    def __init__(self, height=7, width=10, wind_strength=None):
        """
        Initialize Windy Gridworld
        
        Args:
            height: Grid height (default 7)
            width: Grid width (default 10) 
            wind_strength: List of wind strength for each column (default from book)
        """
        self.height = height
        self.width = width
        
        # Default wind strength from Sutton & Barto Example 6.5
        if wind_strength is None:
            self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind_strength = wind_strength
        
        # Start and goal positions
        self.start = (3, 0)  # Row 3, Column 0
        self.goal = (3, 7)   # Row 3, Column 7
        
        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['↑', '↓', '←', '→']
        
        # State space
        self.states = [(i, j) for i in range(height) for j in range(width)]
        
    def is_terminal(self, state):
        """Check if state is terminal (goal)"""
        return state == self.goal
    
    def get_next_state(self, state, action):
        """
        Get next state given current state and action
        Includes wind effect
        """
        if self.is_terminal(state):
            return state
        
        # Apply action
        next_row = state[0] + action[0]
        next_col = state[1] + action[1]
        
        # Apply wind
        wind_effect = self.wind_strength[state[1]]
        next_row -= wind_effect
        
        # Check bounds and clip
        next_row = max(0, min(self.height - 1, next_row))
        next_col = max(0, min(self.width - 1, next_col))
        
        return (next_row, next_col)
    
    def get_reward(self, state, action, next_state):
        """Get reward for transition"""
        if next_state == self.goal:
            return 0  # Goal reward
        else:
            return -1  # Step cost
    
    def reset(self):
        """Reset environment to start state"""
        return self.start
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action index (0-3)
            
        Returns:
            next_state, reward, done, info
        """
        if isinstance(action, int):
            action = self.actions[action]
        
        next_state = self.get_next_state(self.start, action)
        reward = self.get_reward(self.start, action, next_state)
        done = self.is_terminal(next_state)
        
        # Update current state
        self.start = next_state
        
        return next_state, reward, done, {}
    
    def visualize(self, path=None, title="Windy Gridworld"):
        """Visualize the gridworld with optional path"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw grid
        for i in range(self.height + 1):
            ax.axhline(i-0.5, color='black', linewidth=0.5)
        for j in range(self.width + 1):
            ax.axvline(j-0.5, color='black', linewidth=0.5)
        
        # Color cells based on wind strength
        for j in range(self.width):
            wind = self.wind_strength[j]
            color_intensity = wind / max(self.wind_strength) if max(self.wind_strength) > 0 else 0
            for i in range(self.height):
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                       facecolor=plt.cm.Blues(color_intensity * 0.3),
                                       edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # Mark start and goal
        start_rect = patches.Rectangle((self.start[1]-0.4, self.start[0]-0.4), 0.8, 0.8,
                                     facecolor='green', edgecolor='black', linewidth=2)
        ax.add_patch(start_rect)
        ax.text(self.start[1], self.start[0], 'S', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
        
        goal_rect = patches.Rectangle((self.goal[1]-0.4, self.goal[0]-0.4), 0.8, 0.8,
                                    facecolor='red', edgecolor='black', linewidth=2)
        ax.add_patch(goal_rect)
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')
        
        # Draw path if provided
        if path:
            for i in range(len(path) - 1):
                start_pos = (path[i][1], path[i][0])
                end_pos = (path[i+1][1], path[i+1][0])
                arrow = FancyArrowPatch(start_pos, end_pos,
                                      arrowstyle='->', mutation_scale=20,
                                      color='orange', linewidth=3)
                ax.add_patch(arrow)
        
        # Add wind strength labels
        for j in range(self.width):
            ax.text(j, -0.3, f'W={self.wind_strength[j]}', ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig, ax

def test_windy_gridworld():
    """Test the Windy Gridworld environment"""
    print("Testing Windy Gridworld Environment")
    
    env = WindyGridworld()
    print(f"Grid size: {env.height} x {env.width}")
    print(f"Start: {env.start}")
    print(f"Goal: {env.goal}")
    print(f"Wind strength: {env.wind_strength}")
    
    # Test some transitions
    print("\nTesting transitions:")
    test_state = (3, 3)  # Middle of grid
    for i, action in enumerate(env.actions):
        next_state = env.get_next_state(test_state, action)
        reward = env.get_reward(test_state, action, next_state)
        print(f"Action {env.action_names[i]}: {test_state} -> {next_state}, reward: {reward}")
    
    # Visualize
    fig, ax = env.visualize()
    plt.savefig('windy_gridworld.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return env

if __name__ == "__main__":
    env = test_windy_gridworld()
