"""
Simple policy network for TinyZero tasks
Minimal MLP for countdown and multiplication
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPolicy(nn.Module):
    """Simple policy network for arithmetic tasks"""
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.head(x[:, -1])
    
    def sample(self, state, temperature=1.0):
        """Sample action from policy"""
        logits = self.forward(state)
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
