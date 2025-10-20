"""
Reward model for mathematical reasoning quality assessment
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import AutoModel, AutoTokenizer


class MathematicalRewardModel(nn.Module):
    """Reward model for assessing mathematical reasoning quality"""
    
    def __init__(self, config, base_model_name: str = None):
        super().__init__()
        self.config = config
        
        # Initialize without base model for now (can be added later)
        self.tokenizer = None
        self.base_model = None
        hidden_size = 768  # Default hidden size
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Reasoning quality head
        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the reward model"""
        # Simplified forward pass without base model
        batch_size = input_ids.size(0)
        pooled_output = torch.randn(batch_size, 768)  # Random features for now
        
        # Get rewards
        reward = self.reward_head(pooled_output)
        reasoning_quality = self.reasoning_head(pooled_output)
        
        return {
            "reward": reward.squeeze(-1),
            "reasoning_quality": reasoning_quality.squeeze(-1)
        }
    
    def compute_reward(self, problem: str, solution: str, generated_text: str) -> Dict[str, float]:
        """Compute reward for a generated solution"""
        # Simplified reward computation without tokenization
        # Compute additional reward components
        correctness_reward = self._compute_correctness_reward(problem, solution, generated_text)
        reasoning_reward = self._compute_reasoning_reward(generated_text)
        step_penalty = self._compute_step_penalty(generated_text)
        
        # Random neural reward for now
        neural_reward = 0.5
        reasoning_quality = 0.7
        
        # Combine rewards
        total_reward = (
            self.config.reward.correctness_weight * correctness_reward +
            self.config.reward.reasoning_weight * reasoning_reward +
            self.config.reward.step_penalty * step_penalty +
            neural_reward * 0.1  # Small contribution from neural reward
        ) * self.config.reward.final_reward_scale
        
        return {
            "total_reward": total_reward,
            "correctness_reward": correctness_reward,
            "reasoning_reward": reasoning_reward,
            "step_penalty": step_penalty,
            "neural_reward": neural_reward,
            "reasoning_quality": reasoning_quality
        }
    
    def _compute_correctness_reward(self, problem: str, solution: str, generated_text: str) -> float:
        """Compute correctness-based reward"""
        # Extract final answer from generated text
        generated_answer = self._extract_final_answer(generated_text)
        ground_truth = self._extract_final_answer(solution)
        
        # Check if answers match
        if self._normalize_answer(generated_answer) == self._normalize_answer(ground_truth):
            return 1.0
        
        # Partial credit for similar answers
        similarity = self._compute_answer_similarity(generated_answer, ground_truth)
        return similarity * 0.5  # Partial credit
    
    def _compute_reasoning_reward(self, generated_text: str) -> float:
        """Compute reasoning quality reward"""
        # Check for step-by-step reasoning
        step_indicators = ["step", "first", "second", "next", "then", "therefore", "thus"]
        step_count = sum(1 for indicator in step_indicators if indicator in generated_text.lower())
        
        # Check for mathematical expressions
        math_indicators = ["=", "+", "-", "*", "/", "^", "sqrt", "log", "sin", "cos"]
        math_count = sum(1 for indicator in math_indicators if indicator in generated_text)
        
        # Check for explanation quality
        explanation_indicators = ["because", "since", "given", "we know", "from", "using"]
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in generated_text.lower())
        
        # Normalize and combine
        reasoning_score = min(1.0, (step_count + math_count + explanation_count) / 10.0)
        return reasoning_score
    
    def _compute_step_penalty(self, generated_text: str) -> float:
        """Compute penalty for too many steps (encourage efficiency)"""
        lines = generated_text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Penalty increases with number of steps, but caps at reasonable level
        step_penalty = min(0, -len(non_empty_lines) * 0.01)
        return step_penalty
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from text"""
        # Look for common answer patterns
        patterns = [
            r"the answer is\s*([^\n\.]+)",
            r"answer:\s*([^\n\.]+)",
            r"final answer:\s*([^\n\.]+)",
            r"answer\s*=\s*([^\n\.]+)",
            r"=\s*([^\n\.]+)$"
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the last line
        lines = text.strip().split('\n')
        if lines:
            return lines[-1].strip()
        
        return text.strip()
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        answer = answer.strip().lower()
        
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "final answer:", "answer ="]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        return answer
    
    def _compute_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Compute similarity between two answers"""
        norm1 = self._normalize_answer(answer1)
        norm2 = self._normalize_answer(answer2)
        
        if norm1 == norm2:
            return 1.0
        
        # Simple character-level similarity
        if len(norm1) == 0 or len(norm2) == 0:
            return 0.0
        
        # Jaccard similarity
        set1 = set(norm1.split())
        set2 = set(norm2.split())
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class RewardModelTrainer:
    """Trainer for the reward model"""
    
    def __init__(self, model: MathematicalRewardModel, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.reward.learning_rate if hasattr(config.reward, 'learning_rate') else 1e-4
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Compute loss (this would need to be adapted based on your training data)
        # For now, we'll use a simple MSE loss
        target_rewards = batch.get("target_rewards", torch.zeros_like(outputs["reward"]))
        loss = F.mse_loss(outputs["reward"], target_rewards)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reward_mean": outputs["reward"].mean().item(),
            "reasoning_quality_mean": outputs["reasoning_quality"].mean().item()
        }
