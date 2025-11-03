"""
Mock WebArena environment - simplified for fast execution
Simulates realistic web interaction tasks
"""
import torch
import random

class MockWebArena:
    """Simplified WebArena: realistic web tasks"""
    def __init__(self):
        self.tasks = ['login', 'search', 'navigate', 'form_fill']
        self.reset()
    
    def reset(self):
        self.task = random.choice(self.tasks)
        self.stage = 0  # 0=start, 1=middle, 2=complete
        self.steps = 0
        self.max_steps = 15
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """State: task_type, stage, progress"""
        task_idx = self.tasks.index(self.task) / len(self.tasks)
        return torch.tensor([task_idx, self.stage / 2.0, self.steps / self.max_steps])
    
    def step(self, action):
        """Actions: 0=click, 1=type, 2=scroll, 3=submit"""
        self.steps += 1
        reward = -0.1
        
        # Simple logic: need right sequence of actions
        if self.task == 'login':
            if action == 0 and self.stage == 0:  # Click login
                self.stage = 1
                reward = 0.5
            elif action == 1 and self.stage == 1:  # Type credentials
                self.stage = 2
                reward = 0.5
            elif action == 3 and self.stage == 2:  # Submit
                reward = 5.0
                self.done = True
        
        elif self.task == 'search':
            if action == 0 and self.stage == 0:  # Click search box
                self.stage = 1
                reward = 0.5
            elif action == 1 and self.stage == 1:  # Type query
                self.stage = 2
                reward = 0.5
            elif action == 3 and self.stage == 2:  # Submit
                reward = 5.0
                self.done = True
        
        # Similar for other tasks...
        
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._get_state(), reward, self.done, {}
