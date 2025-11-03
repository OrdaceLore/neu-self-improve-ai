"""
Policy network for RAGEN on FrozenLake
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RagenPolicy(nn.Module):
    """Simple policy for FrozenLake"""
    def __init__(self, state_dim=16, action_dim=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Initialize with small positive bias to encourage exploration
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, state):
        return self.net(state)
    
    def sample(self, state, temperature=1.0):
        """Sample action"""
        logits = self.forward(state)
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
