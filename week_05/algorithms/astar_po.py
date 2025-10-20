"""
A*-PO (A*-Policy Optimization) algorithm implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class AStarPOBatch:
    """Batch data for A*-PO training"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor


class AStarPO:
    """
    A*-PO: A two-stage policy optimization algorithm
    
    Stage 1 (Offline): Estimate optimal value function using offline samples
    Stage 2 (Online): Perform policy updates using advantage regression
    """
    
    def __init__(self, config, policy_model, value_model, reward_model):
        self.config = config
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.astar_po.policy_learning_rate,
            eps=1e-5
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=config.astar_po.value_learning_rate,
            eps=1e-5
        )
        
        # Schedulers
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=config.astar_po.policy_epochs
        )
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=config.astar_po.value_epochs
        )
        
        # Storage for offline samples
        self.offline_samples = []
        self.optimal_values = []
        
    def collect_offline_samples(self, dataloader, num_samples: int) -> None:
        """Collect offline samples for value function estimation"""
        print(f"Collecting {num_samples} offline samples...")
        
        self.offline_samples = []
        self.optimal_values = []
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            # Generate responses
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config.model.max_length,
                    temperature=self.config.model.temperature,
                    do_sample=True,
                    pad_token_id=self.policy_model.config.eos_token_id
                )
            
            # Compute rewards
            rewards = self._compute_batch_rewards(batch, outputs)
            
            # Store samples
            for i in range(len(batch["input_ids"])):
                if sample_count >= num_samples:
                    break
                    
                sample = {
                    "input_ids": batch["input_ids"][i],
                    "attention_mask": batch["attention_mask"][i],
                    "generated_ids": outputs[i],
                    "reward": rewards[i].item()
                }
                self.offline_samples.append(sample)
                sample_count += 1
        
        print(f"Collected {len(self.offline_samples)} offline samples")
    
    def estimate_optimal_values(self) -> None:
        """Estimate optimal value function using offline samples"""
        print("Estimating optimal value function...")
        
        # Prepare data for value function training
        input_ids = torch.stack([s["input_ids"] for s in self.offline_samples])
        attention_mask = torch.stack([s["attention_mask"] for s in self.offline_samples])
        rewards = torch.tensor([s["reward"] for s in self.offline_samples])
        
        # Train value function
        self.value_model.train()
        for epoch in range(self.config.astar_po.value_epochs):
            # Forward pass
            values = self.value_model(input_ids, attention_mask)
            
            # Compute loss (MSE between predicted and actual rewards)
            value_loss = F.mse_loss(values.squeeze(), rewards)
            
            # Backward pass
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
            self.value_optimizer.step()
            self.value_scheduler.step()
            
            if epoch % 5 == 0:
                print(f"Value epoch {epoch}, loss: {value_loss.item():.4f}")
        
        # Store optimal values
        with torch.no_grad():
            self.optimal_values = self.value_model(input_ids, attention_mask).squeeze().tolist()
        
        print("Optimal value function estimated")
    
    def compute_advantages(self, batch: AStarPOBatch) -> torch.Tensor:
        """Compute advantages using the estimated optimal value function"""
        # Get current value estimates
        with torch.no_grad():
            current_values = self.value_model(batch.input_ids, batch.attention_mask).squeeze()
        
        # Compute advantages using optimal values as targets
        advantages = batch.rewards - current_values
        
        # Apply GAE if needed
        if hasattr(self.config.astar_po, 'lambda_gae'):
            advantages = self._compute_gae_advantages(advantages, batch.rewards)
        
        return advantages
    
    def _compute_gae_advantages(self, advantages: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        # Simplified GAE computation
        gamma = self.config.astar_po.gamma
        lambda_gae = self.config.astar_po.lambda_gae
        
        # For simplicity, we'll use a basic advantage computation
        # In practice, you'd want to implement full GAE
        return advantages * lambda_gae
    
    def policy_update(self, batch: AStarPOBatch) -> Dict[str, float]:
        """Perform policy update using advantage regression"""
        self.policy_model.train()
        
        # Get current policy outputs
        outputs = self.policy_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.target_ids
        )
        
        log_probs = outputs.logits.log_softmax(dim=-1)
        old_log_probs = batch.old_log_probs
        
        # Compute policy loss using advantage regression
        ratio = torch.exp(log_probs - old_log_probs)
        advantages = batch.advantages
        
        # Clipped policy loss
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(
            ratio, 1 - self.config.astar_po.clip_ratio, 1 + self.config.astar_po.clip_ratio
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # Entropy bonus
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        entropy_loss = -self.config.astar_po.entropy_coef * entropy
        
        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_policy_loss": total_policy_loss.item(),
            "mean_advantage": advantages.mean().item(),
            "mean_ratio": ratio.mean().item()
        }
    
    def value_update(self, batch: AStarPOBatch) -> Dict[str, float]:
        """Perform value function update"""
        self.value_model.train()
        
        # Get current value estimates
        values = self.value_model(batch.input_ids, batch.attention_mask).squeeze()
        
        # Compute value loss
        value_loss = F.mse_loss(values, batch.rewards)
        
        # Clipped value loss (optional)
        if hasattr(self.config.astar_po, 'value_clip_ratio'):
            old_values = batch.old_values
            value_clipped = old_values + torch.clamp(
                values - old_values, -self.config.astar_po.value_clip_ratio, self.config.astar_po.value_clip_ratio
            )
            value_loss_clipped = F.mse_loss(value_clipped, batch.rewards)
            value_loss = torch.max(value_loss, value_loss_clipped)
        
        # Backward pass
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.value_optimizer.step()
        
        return {
            "value_loss": value_loss.item(),
            "mean_value": values.mean().item(),
            "mean_reward": batch.rewards.mean().item()
        }
    
    def _compute_batch_rewards(self, batch: Dict[str, torch.Tensor], generated_outputs: torch.Tensor) -> torch.Tensor:
        """Compute rewards for a batch of generated outputs"""
        rewards = []
        
        for i in range(len(batch["input_ids"])):
            # Decode generated text
            generated_text = self.policy_model.tokenizer.decode(
                generated_outputs[i], skip_special_tokens=True
            )
            
            # Get problem and solution
            problem = batch["problems"][i]
            solution = batch["solutions"][i]
            
            # Compute reward
            reward_dict = self.reward_model.compute_reward(problem, solution, generated_text)
            rewards.append(reward_dict["total_reward"])
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step combining offline and online stages"""
        # Generate responses
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.config.model.max_length,
                temperature=self.config.model.temperature,
                do_sample=True,
                pad_token_id=self.policy_model.config.eos_token_id
            )
        
        # Compute rewards
        rewards = self._compute_batch_rewards(batch, outputs)
        
        # Get current log probabilities
        with torch.no_grad():
            current_outputs = self.policy_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["target_ids"]
            )
            current_log_probs = current_outputs.logits.log_softmax(dim=-1)
        
        # Get current values
        with torch.no_grad():
            current_values = self.value_model(batch["input_ids"], batch["attention_mask"]).squeeze()
        
        # Create A*-PO batch
        astar_batch = AStarPOBatch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_ids=batch["target_ids"],
            rewards=rewards,
            advantages=torch.zeros_like(rewards),  # Will be computed
            values=current_values,
            old_log_probs=current_log_probs,
            old_values=current_values
        )
        
        # Compute advantages
        astar_batch.advantages = self.compute_advantages(astar_batch)
        
        # Perform updates
        policy_metrics = self.policy_update(astar_batch)
        value_metrics = self.value_update(astar_batch)
        
        # Combine metrics
        metrics = {**policy_metrics, **value_metrics}
        metrics["mean_reward"] = rewards.mean().item()
        metrics["mean_advantage"] = astar_batch.advantages.mean().item()
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoints"""
        checkpoint = {
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoints"""
        checkpoint = torch.load(path, map_location="cpu")
        self.policy_model.load_state_dict(checkpoint["policy_model"])
        self.value_model.load_state_dict(checkpoint["value_model"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
