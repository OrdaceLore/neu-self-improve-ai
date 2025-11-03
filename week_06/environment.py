"""
Minimal environments for countdown and multiplication tasks
Simplified for fast execution
"""
import torch
import random

class CountdownEnv:
    """Simple countdown task: count from N to 0"""
    def __init__(self, max_value=10):
        self.max_value = max_value
        self.reset()
    
    def reset(self):
        self.value = random.randint(5, self.max_value)
        self.target = 0
        self.done = False
        return torch.tensor([self.value])
    
    def step(self, action):
        """Action: -1 (decrement) or 0 (keep)"""
        if action == 0:  # decrement
            self.value -= 1
        elif action == 1:  # keep (wrong)
            pass
        
        reward = 1.0 if self.value == self.target else 0.0
        self.done = self.value <= self.target
        
        return torch.tensor([self.value]), reward, self.done, {}

class MultiplicationEnv:
    """Simple multiplication: compute a * b"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.a = random.randint(2, 5)  # Smaller numbers for simplicity
        self.b = random.randint(2, 5)
        self.target = self.a * self.b
        self.current_guess = None
        self.done = False
        return torch.tensor([self.a, self.b])
    
    def step(self, action):
        """Action: guess the product"""
        self.current_guess = action.item() if isinstance(action, torch.Tensor) else action
        reward = 1.0 if self.current_guess == self.target else -0.1
        self.done = True
        return torch.tensor([self.a, self.b]), reward, self.done, {}
