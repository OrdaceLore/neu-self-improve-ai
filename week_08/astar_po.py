"""
A*-PO (A-Star Policy Optimization) for Web Tasks - Improved Version
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_astar_po_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    kl_weight: float = 0.1,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    A*-PO loss with proper advantage weighting.
    
    Key features:
    - Exponential advantage weighting (A* style)
    - KL regularization to prevent large policy updates
    - Optional value function loss
    - Entropy bonus for exploration
    
    Args:
        log_probs: Log probabilities of actions under current policy
        advantages: Computed advantages (returns - values)
        old_log_probs: Log probabilities under old policy (for KL)
        values: Value estimates from critic
        returns: Actual returns
        kl_weight: Weight for KL penalty
        value_weight: Weight for value loss
        entropy_weight: Weight for entropy bonus
    
    Returns:
        total_loss: Combined loss
        metrics: Dictionary of loss components
    """
    metrics = {}
    
    # A*-PO style: exponential advantage weighting
    # This emphasizes trajectories with high advantages
    beta = 0.5  # Temperature for advantage weighting
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    weights = F.softmax(advantages_normalized / beta, dim=0)
    
    # Policy gradient loss with A* weighting
    policy_loss = -(weights.detach() * log_probs * advantages.detach()).sum()
    metrics["policy_loss"] = policy_loss.item()
    
    # KL penalty if old log probs provided
    if old_log_probs is not None:
        # Approximate KL divergence
        log_ratio = log_probs - old_log_probs
        kl_div = (torch.exp(log_ratio) - 1 - log_ratio).mean()
        kl_loss = kl_weight * kl_div
        metrics["kl_loss"] = kl_loss.item()
    else:
        kl_loss = 0.0
    
    # Value loss if provided
    if values is not None and returns is not None:
        value_loss = value_weight * F.mse_loss(values, returns)
        metrics["value_loss"] = value_loss.item()
    else:
        value_loss = 0.0
    
    # Entropy bonus
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    entropy_loss = -entropy_weight * entropy
    metrics["entropy"] = entropy.item()
    
    # Total loss
    total_loss = policy_loss + kl_loss + value_loss + entropy_loss
    metrics["total_loss"] = total_loss.item()
    
    return total_loss, metrics


class AStarPOOptimizer:
    """
    A*-PO optimizer for web task policies.
    """
    
    def __init__(
        self,
        policy,
        lr: float = 1e-3,
        gamma: float = 0.99,
        kl_weight: float = 0.1,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.kl_weight = kl_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        
        # For tracking baseline
        self.baseline = 0.0
        self.baseline_alpha = 0.01
    
    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Update policy using A*-PO.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            old_log_probs: Log probs from behavior policy (optional)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        # Compute returns and advantages
        returns = self.compute_returns(rewards)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get current policy outputs
        logits, values = self.policy(states)
        values = values.squeeze(-1)
        
        # Compute log probs of taken actions
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Update baseline
        self.baseline = (1 - self.baseline_alpha) * self.baseline + \
                       self.baseline_alpha * returns.mean().item()
        
        # Compute A*-PO loss
        loss, metrics = compute_astar_po_loss(
            log_probs=log_probs,
            advantages=advantages,
            old_log_probs=old_log_probs,
            values=values,
            returns=returns,
            kl_weight=self.kl_weight,
            value_weight=self.value_weight,
            entropy_weight=self.entropy_weight
        )
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), 
            self.max_grad_norm
        )
        
        self.optimizer.step()
        
        metrics["baseline"] = self.baseline
        return metrics

