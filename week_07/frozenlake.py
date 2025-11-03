"""
Simplified FrozenLake environment
4x4 grid for minimal computation
"""
import torch
import random

class SimpleFrozenLake:
    """4x4 FrozenLake: agent moves to goal, avoids holes"""
    def __init__(self):
        # 0=start, 1=safe, 2=hole, 3=goal
        self.grid = [
            [0, 1, 1, 1],
            [1, 2, 1, 2],
            [1, 1, 1, 2],
            [2, 1, 1, 3]
        ]
        self.reset()
    
    def reset(self):
        self.pos = [0, 0]
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """State: one-hot of position"""
        state = torch.zeros(16)  # 4x4 = 16
        state[self.pos[0] * 4 + self.pos[1]] = 1.0
        return state
    
    def step(self, action):
        """Actions: 0=up, 1=down, 2=left, 3=right"""
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        move = moves[action]
        new_pos = [self.pos[0] + move[0], self.pos[1] + move[1]]
        
        # Bounds check
        if 0 <= new_pos[0] < 4 and 0 <= new_pos[1] < 4:
            self.pos = new_pos
        
        cell = self.grid[self.pos[0]][self.pos[1]]
        if cell == 2:  # hole
            reward = -1.0
            self.done = True
        elif cell == 3:  # goal
            reward = 1.2  # High reward for goal (matching working code)
            self.done = True
        else:
            # Reward shaping: step penalty + progress bonus (inspired by working code)
            step_penalty = -0.02
            goal_pos = [3, 3]
            dist_to_goal = abs(self.pos[0] - goal_pos[0]) + abs(self.pos[1] - goal_pos[1])
            max_dist = 6  # Max distance from (0,0) to (3,3)
            progress_bonus = 0.1 * (1.0 - dist_to_goal / max_dist)
            reward = step_penalty + progress_bonus
        
        return self._get_state(), reward, self.done, {}
