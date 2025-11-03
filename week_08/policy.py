"""Policy network for RAGEN on WebShop/WebArena"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WebPolicy(nn.Module):
    """Policy for web interaction tasks"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)
    
    def sample(self, state, temperature=1.0):
        logits = self.forward(state)
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
