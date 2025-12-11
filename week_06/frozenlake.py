"""
FrozenLake Environment - Full Implementation
4x4 and 8x8 variants with proper reward shaping for RAGEN
"""
import torch
import random
from typing import Tuple, Dict, List


class FrozenLakeEnv:
    """
    FrozenLake environment for RAGEN evaluation.
    
    Grid cells:
    - S: Start (0,0)
    - F: Frozen/Safe
    - H: Hole (terminal, negative reward)
    - G: Goal (terminal, positive reward)
    
    Actions: 0=Left, 1=Down, 2=Right, 3=Up
    """
    
    # Standard 4x4 map from OpenAI Gym
    MAP_4x4 = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    
    # Standard 8x8 map
    MAP_8x8 = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
    
    def __init__(self, map_name: str = "4x4", is_slippery: bool = False):
        """
        Initialize FrozenLake.
        
        Args:
            map_name: "4x4" or "8x8"
            is_slippery: If True, actions have stochastic outcomes
        """
        self.map = self.MAP_4x4 if map_name == "4x4" else self.MAP_8x8
        self.size = len(self.map)
        self.is_slippery = is_slippery
        
        self.n_states = self.size * self.size
        self.n_actions = 4
        
        # Find start and goal positions
        self.start_pos = None
        self.goal_pos = None
        self.holes = []
        
        for i, row in enumerate(self.map):
            for j, cell in enumerate(row):
                if cell == 'S':
                    self.start_pos = (i, j)
                elif cell == 'G':
                    self.goal_pos = (i, j)
                elif cell == 'H':
                    self.holes.append((i, j))
        
        self.reset()
    
    def reset(self) -> torch.Tensor:
        """Reset to start position"""
        self.pos = list(self.start_pos)
        self.done = False
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """One-hot encoding of current position"""
        state = torch.zeros(self.n_states)
        idx = self.pos[0] * self.size + self.pos[1]
        state[idx] = 1.0
        return state
    
    def _pos_to_idx(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to state index"""
        return pos[0] * self.size + pos[1]
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take action in environment.
        
        Actions: 0=Left, 1=Down, 2=Right, 3=Up
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        self.steps += 1
        
        # Handle slippery ice (stochastic transitions)
        if self.is_slippery and random.random() < 0.33:
            action = random.randint(0, 3)
        
        # Movement deltas: Left, Down, Right, Up
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        delta = deltas[action]
        
        new_row = self.pos[0] + delta[0]
        new_col = self.pos[1] + delta[1]
        
        # Bounds check
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.pos = [new_row, new_col]
        
        # Check cell type
        cell = self.map[self.pos[0]][self.pos[1]]
        
        info = {"cell": cell}
        
        if cell == 'H':  # Hole
            reward = -1.0
            self.done = True
            info["success"] = False
        elif cell == 'G':  # Goal
            reward = 1.0
            self.done = True
            info["success"] = True
        else:  # Safe cell
            # Reward shaping: small penalty + distance-based bonus
            step_penalty = -0.01
            
            # Manhattan distance to goal
            dist = abs(self.pos[0] - self.goal_pos[0]) + abs(self.pos[1] - self.goal_pos[1])
            max_dist = (self.size - 1) * 2
            progress = (max_dist - dist) / max_dist
            
            reward = step_penalty + 0.02 * progress
        
        # Timeout
        if self.steps >= 100:
            self.done = True
            info["timeout"] = True
        
        return self._get_state(), reward, self.done, info
    
    def render(self) -> str:
        """Render current state as string"""
        result = []
        for i, row in enumerate(self.map):
            line = ""
            for j, cell in enumerate(row):
                if [i, j] == self.pos:
                    line += "A"  # Agent
                else:
                    line += cell
            result.append(line)
        return "\n".join(result)
    
    @property
    def state_dim(self) -> int:
        return self.n_states
    
    @property
    def action_dim(self) -> int:
        return self.n_actions


def test_environment():
    """Test the FrozenLake environment"""
    print("Testing FrozenLake Environment")
    print("=" * 40)
    
    env = FrozenLakeEnv(map_name="4x4")
    state = env.reset()
    
    print(f"State shape: {state.shape}")
    print(f"Initial position: {env.pos}")
    print(f"\nInitial state:")
    print(env.render())
    
    # Random episode
    print("\nRunning random episode:")
    total_reward = 0
    for step in range(20):
        action = random.randint(0, 3)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        action_names = ["Left", "Down", "Right", "Up"]
        print(f"Step {step+1}: {action_names[action]}, Reward: {reward:.3f}, Pos: {env.pos}")
        
        if done:
            print(f"\nEpisode ended! Total reward: {total_reward:.3f}")
            print(f"Success: {info.get('success', False)}")
            break
    
    return env


if __name__ == "__main__":
    test_environment()

