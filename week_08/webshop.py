"""
Mock WebShop environment - simplified for fast execution
Simulates e-commerce shopping task
"""
import torch
import random

class MockWebShop:
    """Simplified WebShop: navigate and purchase items"""
    def __init__(self):
        self.items = {
            'laptop': {'price': 800, 'rating': 4.5},
            'phone': {'price': 600, 'rating': 4.2},
            'tablet': {'price': 400, 'rating': 4.0},
            'headphones': {'price': 100, 'rating': 4.3},
        }
        self.reset()
    
    def reset(self):
        self.task = random.choice(['find_cheap', 'find_high_rated'])
        self.current_item = None
        self.steps = 0
        self.max_steps = 10
        self.done = False
        # State: task_type, item_selected, price, rating
        return self._get_state()
    
    def _get_state(self):
        """State encoding"""
        task_val = 1.0 if self.task == 'find_cheap' else 2.0
        item_idx = list(self.items.keys()).index(self.current_item) if self.current_item else 0
        price = self.items[self.current_item]['price'] / 1000.0 if self.current_item else 0.0
        rating = self.items[self.current_item]['rating'] / 5.0 if self.current_item else 0.0
        return torch.tensor([task_val, item_idx / 4.0, price, rating, self.steps / self.max_steps])
    
    def step(self, action):
        """Actions: 0=select_laptop, 1=select_phone, 2=select_tablet, 3=select_headphones, 4=purchase"""
        self.steps += 1
        reward = -0.1  # Step penalty
        
        if action < 4:
            item_name = list(self.items.keys())[action]
            self.current_item = item_name
        
        elif action == 4:  # Purchase
            if self.current_item:
                item = self.items[self.current_item]
                if self.task == 'find_cheap' and item['price'] < 500:
                    reward = 10.0  # Success
                elif self.task == 'find_high_rated' and item['rating'] > 4.2:
                    reward = 10.0  # Success
                else:
                    reward = -5.0  # Wrong choice
                self.done = True
        
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._get_state(), reward, self.done, {}
