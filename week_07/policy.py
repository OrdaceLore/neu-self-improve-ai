"""
Policy Network for RAGEN - Pure PyTorch Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RAGENPolicy(nn.Module):
    """
    Policy network for RAGEN.
    
    Architecture:
    - Feature extractor (MLP)
    - Policy head (actor)
    - Value head (critic)
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 128,
        n_layers: int = 2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature extractor
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        
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
        Forward pass.
        
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
    
    def get_action(
        self, 
        state: torch.Tensor, 
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            state: State tensor
            temperature: Sampling temperature
            deterministic: If True, return argmax action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        logits, value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        else:
            # Apply temperature
            logits_scaled = logits / max(temperature, 0.01)
            probs = F.softmax(logits_scaled, dim=-1)
            
            # Sample action
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO-style updates.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            
        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
            entropy: Policy entropy
        """
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class FrozenLakePolicy(RAGENPolicy):
    """Policy specifically for FrozenLake 4x4"""
    def __init__(self, hidden_dim: int = 128):
        super().__init__(state_dim=16, action_dim=4, hidden_dim=hidden_dim)


class FrozenLake8x8Policy(RAGENPolicy):
    """Policy for FrozenLake 8x8"""
    def __init__(self, hidden_dim: int = 128):
        super().__init__(state_dim=64, action_dim=4, hidden_dim=hidden_dim)

