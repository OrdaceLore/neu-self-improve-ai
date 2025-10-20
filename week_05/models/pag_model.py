"""
Policy as Generative Verifier (PAG) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class PAGTurn:
    """Single turn in PAG multi-turn process"""
    problem: str
    generated_solution: str
    verification_score: float
    is_correct: bool
    confidence: float
    reasoning_steps: List[str]


class PolicyAsGenerativeVerifier(nn.Module):
    """
    Policy as Generative Verifier (PAG) implementation
    
    Multi-turn self-correction framework where the model:
    1. Generates an initial solution
    2. Verifies the solution using the same model
    3. Revises the solution if needed
    4. Repeats until satisfied or max turns reached
    """
    
    def __init__(self, policy_model, value_model, reward_model, config):
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.config = config
        
        # Verification head for self-assessment
        self.verification_head = nn.Sequential(
            nn.Linear(policy_model.base_model.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Confidence head for uncertainty estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(policy_model.base_model.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for verification and confidence estimation"""
        # Get hidden states from policy model
        with torch.no_grad():
            outputs = self.policy_model.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
        
        # Pool hidden states
        pooled_output = self._pool_hidden_states(hidden_states, attention_mask)
        
        # Get verification and confidence scores
        verification_score = self.verification_head(pooled_output)
        confidence = self.confidence_head(pooled_output)
        
        return {
            "verification_score": verification_score.squeeze(-1),
            "confidence": confidence.squeeze(-1)
        }
    
    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states using attention mask"""
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden = hidden_states * mask_expanded
        
        summed = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        pooled = summed / lengths
        
        return pooled
    
    def multi_turn_reasoning(self, problem: str, max_turns: int = None) -> List[PAGTurn]:
        """Perform multi-turn reasoning with self-verification"""
        if max_turns is None:
            max_turns = self.config.pag.max_turns
        
        turns = []
        current_problem = problem
        
        for turn in range(max_turns):
            # Generate solution
            generated_solution = self._generate_solution(current_problem)
            
            # Verify solution
            verification_result = self._verify_solution(problem, generated_solution)
            
            # Create turn record
            pag_turn = PAGTurn(
                problem=current_problem,
                generated_solution=generated_solution,
                verification_score=verification_result["verification_score"],
                is_correct=verification_result["is_correct"],
                confidence=verification_result["confidence"],
                reasoning_steps=verification_result["reasoning_steps"]
            )
            turns.append(pag_turn)
            
            # Check if we should stop
            if verification_result["is_correct"] or verification_result["confidence"] > self.config.pag.verification_threshold:
                break
            
            # Prepare for next turn (self-correction)
            current_problem = self._create_correction_prompt(problem, generated_solution, verification_result)
        
        return turns
    
    def _generate_solution(self, problem: str) -> str:
        """Generate a solution for the given problem"""
        # Create prompt
        prompt = f"""Solve the following mathematical problem step by step. Show your reasoning clearly.

Problem: {problem}

Solution:"""
        
        # Tokenize
        inputs = self.policy_model.base_model.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.model.max_length,
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.model.max_length,
                temperature=self.config.model.temperature,
                do_sample=True,
                pad_token_id=self.policy_model.base_model.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.policy_model.base_model.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        
        # Extract solution part
        if "Solution:" in generated_text:
            solution = generated_text.split("Solution:")[-1].strip()
        else:
            solution = generated_text
        
        return solution
    
    def _verify_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        """Verify the correctness of a solution"""
        # Create verification prompt
        verification_prompt = f"""Verify the correctness of the following mathematical solution. 
        Rate the solution on a scale of 0 to 1 and explain your reasoning.

Problem: {problem}

Solution: {solution}

Verification:"""
        
        # Tokenize
        inputs = self.policy_model.base_model.tokenizer(
            verification_prompt,
            return_tensors="pt",
            max_length=self.config.model.max_length,
            truncation=True
        )
        
        # Get verification scores
        with torch.no_grad():
            verification_outputs = self.forward(inputs["input_ids"], inputs["attention_mask"])
            verification_score = verification_outputs["verification_score"].item()
            confidence = verification_outputs["confidence"].item()
        
        # Generate verification reasoning
        with torch.no_grad():
            verification_text = self.policy_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.policy_model.base_model.tokenizer.pad_token_id
            )
            
            verification_reasoning = self.policy_model.base_model.tokenizer.decode(
                verification_text[0], skip_special_tokens=True
            )
        
        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(verification_reasoning)
        
        # Determine if correct
        is_correct = verification_score > self.config.pag.verification_threshold
        
        return {
            "verification_score": verification_score,
            "confidence": confidence,
            "is_correct": is_correct,
            "reasoning_steps": reasoning_steps,
            "verification_reasoning": verification_reasoning
        }
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from verification text"""
        # Simple extraction - look for numbered steps
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', 'Step', 'First', 'Second', 'Third']):
                steps.append(line)
        
        return steps
    
    def _create_correction_prompt(self, original_problem: str, previous_solution: str, 
                                verification_result: Dict[str, Any]) -> str:
        """Create a prompt for self-correction"""
        correction_prompt = f"""The previous solution was incorrect or had low confidence. 
        Please solve the problem again with a different approach.

Original Problem: {original_problem}

Previous Solution: {previous_solution}

Issues with previous solution: {verification_result.get('verification_reasoning', 'Low confidence')}

Please provide a corrected solution:"""
        
        return correction_prompt
    
    def compute_pag_reward(self, turns: List[PAGTurn], ground_truth: str) -> Dict[str, float]:
        """Compute reward for PAG multi-turn process"""
        if not turns:
            return {"total_reward": 0.0, "correctness_reward": 0.0, "efficiency_reward": 0.0}
        
        # Correctness reward (based on final turn)
        final_turn = turns[-1]
        correctness_reward = 1.0 if final_turn.is_correct else 0.0
        
        # Efficiency reward (fewer turns is better)
        efficiency_reward = max(0.0, 1.0 - (len(turns) - 1) * 0.2)
        
        # Verification quality reward
        verification_rewards = [turn.verification_score for turn in turns]
        avg_verification = sum(verification_rewards) / len(verification_rewards)
        
        # Confidence reward
        confidence_rewards = [turn.confidence for turn in turns]
        avg_confidence = sum(confidence_rewards) / len(confidence_rewards)
        
        # Combine rewards
        total_reward = (
            correctness_reward * 0.5 +
            efficiency_reward * 0.2 +
            avg_verification * 0.2 +
            avg_confidence * 0.1
        )
        
        return {
            "total_reward": total_reward,
            "correctness_reward": correctness_reward,
            "efficiency_reward": efficiency_reward,
            "verification_reward": avg_verification,
            "confidence_reward": avg_confidence,
            "num_turns": len(turns)
        }
    
    def train_verification_head(self, batch: Dict[str, torch.Tensor], 
                               target_scores: torch.Tensor) -> Dict[str, float]:
        """Train the verification head"""
        self.verification_head.train()
        
        # Forward pass
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        predicted_scores = outputs["verification_score"]
        
        # Compute loss
        loss = F.mse_loss(predicted_scores, target_scores)
        
        # Backward pass
        loss.backward()
        
        return {
            "verification_loss": loss.item(),
            "mean_predicted_score": predicted_scores.mean().item(),
            "mean_target_score": target_scores.mean().item()
        }
    
    def train_confidence_head(self, batch: Dict[str, torch.Tensor], 
                            target_confidences: torch.Tensor) -> Dict[str, float]:
        """Train the confidence head"""
        self.confidence_head.train()
        
        # Forward pass
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        predicted_confidences = outputs["confidence"]
        
        # Compute loss
        loss = F.mse_loss(predicted_confidences, target_confidences)
        
        # Backward pass
        loss.backward()
        
        return {
            "confidence_loss": loss.item(),
            "mean_predicted_confidence": predicted_confidences.mean().item(),
            "mean_target_confidence": target_confidences.mean().item()
        }
