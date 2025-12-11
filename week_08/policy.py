"""
Policy Networks for Web Tasks - Improved Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WebPolicy(nn.Module):
    """
    Policy network for web interaction tasks.
    
    Architecture:
    - Multi-layer MLP with layer normalization
    - Separate value head for actor-critic
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits and value estimate.
        
        Args:
            state: State tensor [batch, state_dim]
            
        Returns:
            logits: Action logits [batch, action_dim]
            value: Value estimate [batch, 1]
        """
        features = self.feature_net(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value
    
    def sample(
        self, 
        state: torch.Tensor, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            temperature: Sampling temperature (lower = more greedy)
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        logits, _ = self.forward(state)
        
        # Apply temperature
        logits = logits / max(temperature, 0.01)
        
        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        logits, _ = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate only"""
        _, value = self.forward(state)
        return value.squeeze(-1)


class WebShopPolicy(WebPolicy):
    """Policy specifically tuned for WebShop environment"""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__(state_dim=8, action_dim=5, hidden_dim=hidden_dim)


class WebArenaPolicy(WebPolicy):
    """Policy specifically tuned for WebArena environment"""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__(state_dim=10, action_dim=6, hidden_dim=hidden_dim)

