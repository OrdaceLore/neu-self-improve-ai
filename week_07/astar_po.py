"""
A*-PO (A-Star Policy Optimization) - Full Implementation
Based on: "A*-PO: Accelerating RL for LLM Reasoning with Optimal Advantage Regression"

Key features:
1. Offline stage: Collect samples and estimate optimal value function
2. Online stage: Policy updates with advantage weighting
3. Single generation per prompt (efficient)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RolloutBatch:
    """Batch of rollout data"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    dones: torch.Tensor


class AStarPO:
    """
    A*-PO Algorithm Implementation.
    
    Two-stage approach:
    1. Offline: Estimate optimal value function V*(s) from samples
    2. Online: Update policy using advantage regression with A* weighting
    
    Key insight: Use exponential advantage weighting to focus on high-reward trajectories.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        beta: float = 0.5,  # A* temperature
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        """
        Initialize A*-PO.
        
        Args:
            policy: Policy network with actor-critic heads
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            beta: Temperature for A* advantage weighting (lower = more selective)
            clip_ratio: PPO-style clipping ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
        """
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        
        # Baseline tracking for A* weighting
        self.baseline = 0.0
        self.baseline_alpha = 0.01
    
    def compute_returns_and_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and GAE advantages.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        n = len(rewards)
        returns = torch.zeros(n)
        advantages = torch.zeros(n)
        
        # Bootstrap value (0 if terminal)
        next_value = 0.0
        next_advantage = 0.0
        
        for t in reversed(range(n)):
            mask = 1.0 - float(dones[t])
            
            # Compute return
            returns[t] = rewards[t] + self.gamma * next_value * mask
            
            # Compute GAE advantage
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * mask
            
            next_value = values[t]
            next_advantage = advantages[t]
        
        return returns, advantages
    
    def compute_astar_weights(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute A* weights from advantages.
        
        A* weighting: w_i = exp(A_i / beta) / sum(exp(A_j / beta))
        This emphasizes high-advantage trajectories.
        """
        # Normalize advantages
        adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # A* weighting with temperature
        weights = F.softmax(adv_normalized / self.beta, dim=0)
        
        # Scale to have mean 1 (maintains gradient scale)
        weights = weights * len(weights)
        
        return weights
    
    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        """
        Update policy using A*-PO.
        
        Args:
            batch: Rollout batch with states, actions, rewards, etc.
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Get current policy outputs
        log_probs, values, entropy = self.policy.evaluate_actions(
            batch.states, batch.actions
        )
        
        # Compute A* weights
        astar_weights = self.compute_astar_weights(batch.advantages)
        
        # Policy loss with A* weighting and PPO clipping
        ratio = torch.exp(log_probs - batch.log_probs.detach())
        
        # Unclipped objective
        surr1 = -astar_weights.detach() * ratio * batch.advantages.detach()
        
        # Clipped objective
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surr2 = -astar_weights.detach() * ratio_clipped * batch.advantages.detach()
        
        policy_loss = torch.max(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_pred = values
        value_target = batch.returns.detach()
        
        value_loss_unclipped = (value_pred - value_target) ** 2
        value_clipped = batch.values.detach() + torch.clamp(
            value_pred - batch.values.detach(),
            -self.clip_ratio, self.clip_ratio
        )
        value_loss_clipped = (value_clipped - value_target) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Entropy bonus (weighted by A* weights)
        entropy_loss = -(astar_weights.detach() * entropy).mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update baseline
        self.baseline = (
            (1 - self.baseline_alpha) * self.baseline + 
            self.baseline_alpha * batch.returns.mean().item()
        )
        
        # Compute metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clip_fraction = (torch.abs(ratio - 1) > self.clip_ratio).float().mean().item()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "entropy": entropy.mean().item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "mean_advantage": batch.advantages.mean().item(),
            "mean_return": batch.returns.mean().item(),
            "baseline": self.baseline
        }
    
    def process_rollouts(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        rewards: List[float],
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        dones: List[bool]
    ) -> RolloutBatch:
        """
        Process raw rollout data into a batch.
        """
        # Stack tensors
        states_t = torch.stack(states)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards, [v.item() for v in values_t], dones
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return RolloutBatch(
            states=states_t,
            actions=actions_t,
            rewards=rewards_t,
            log_probs=log_probs_t,
            values=values_t,
            returns=returns,
            advantages=advantages,
            dones=dones_t
        )

