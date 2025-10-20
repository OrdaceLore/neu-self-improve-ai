"""
Evaluation module for PAG with A*-PO
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from models.qwen_model import PolicyModel, ValueModel
from models.pag_model import PolicyAsGenerativeVerifier
from data.math_dataset import MATHDataset
from data.reward_model import MathematicalRewardModel


class PAGEvaluator:
    """Evaluator for PAG with A*-PO model"""
    
    def __init__(self, config, model_path: str = None):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Initialize models
        self._initialize_models()
        
        # Load models if path provided
        if model_path:
            self.load_models(model_path)
        
        # Initialize dataset
        self.dataset = MATHDataset(config)
        
        # Evaluation metrics
        self.metrics = {}
    
    def _initialize_models(self):
        """Initialize models for evaluation"""
        from models.qwen_model import QwenModel
        
        # Base model
        self.base_model = QwenModel(self.config)
        
        # Policy model
        self.policy_model = PolicyModel(self.base_model)
        
        # Value model
        self.value_model = ValueModel(self.base_model)
        
        # Reward model
        self.reward_model = MathematicalRewardModel(self.config)
        
        # PAG model
        self.pag_model = PolicyAsGenerativeVerifier(
            policy_model=self.policy_model,
            value_model=self.value_model,
            reward_model=self.reward_model,
            config=self.config
        )
        
        # Move to device
        self.policy_model = self.policy_model.to(self.device)
        self.value_model = self.value_model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.pag_model = self.pag_model.to(self.device)
    
    def load_models(self, model_path: str):
        """Load trained models"""
        print(f"Loading models from {model_path}")
        
        # Load policy model
        self.policy_model.load_pretrained(os.path.join(model_path, "policy_model"))
        
        # Load value model
        self.value_model.load_state_dict(
            torch.load(os.path.join(model_path, "value_model.pt"), map_location=self.device)
        )
        
        # Load reward model
        self.reward_model.load_state_dict(
            torch.load(os.path.join(model_path, "reward_model.pt"), map_location=self.device)
        )
        
        print("Models loaded successfully")
    
    def evaluate_on_math500(self, output_file: str = None) -> Dict[str, Any]:
        """Evaluate on MATH 500 dataset"""
        print("Evaluating on MATH 500 dataset...")
        
        # Load test dataset
        _, test_dataset = self.dataset.load_dataset()
        test_dataloader = self.dataset.get_dataloader(test_dataset, shuffle=False)
        
        # Evaluation results
        results = {
            "total_samples": 0,
            "correct_samples": 0,
            "accuracy": 0.0,
            "per_problem_results": [],
            "level_accuracy": {},
            "type_accuracy": {},
            "reward_metrics": {
                "total_rewards": [],
                "correctness_rewards": [],
                "reasoning_rewards": [],
                "step_penalties": []
            }
        }
        
        # Set models to eval mode
        self.policy_model.eval()
        self.value_model.eval()
        self.reward_model.eval()
        self.pag_model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating MATH 500")):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Evaluate each sample in the batch
                for i in range(len(batch["input_ids"])):
                    problem = batch["problems"][i]
                    ground_truth = batch["solutions"][i]
                    level = batch["levels"][i]
                    problem_type = batch["types"][i]
                    
                    # Generate solution using PAG
                    turns = self.pag_model.multi_turn_reasoning(problem)
                    
                    # Get final solution
                    if turns:
                        final_solution = turns[-1].generated_solution
                    else:
                        # Fallback to single generation
                        final_solution = self._generate_single_solution(problem)
                    
                    # Extract answer
                    generated_answer = self.dataset.extract_answer(final_solution)
                    
                    # Check correctness
                    is_correct = self.dataset.is_correct(generated_answer, ground_truth)
                    
                    # Compute rewards
                    reward_dict = self.reward_model.compute_reward(problem, ground_truth, final_solution)
                    
                    # Store results
                    problem_result = {
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "generated_solution": final_solution,
                        "generated_answer": generated_answer,
                        "is_correct": is_correct,
                        "level": level,
                        "type": problem_type,
                        "num_turns": len(turns),
                        "verification_scores": [turn.verification_score for turn in turns],
                        "confidence_scores": [turn.confidence for turn in turns],
                        "rewards": reward_dict
                    }
                    
                    results["per_problem_results"].append(problem_result)
                    
                    # Update counters
                    results["total_samples"] += 1
                    if is_correct:
                        results["correct_samples"] += 1
                    
                    # Update level accuracy
                    if level not in results["level_accuracy"]:
                        results["level_accuracy"][level] = {"correct": 0, "total": 0}
                    results["level_accuracy"][level]["total"] += 1
                    if is_correct:
                        results["level_accuracy"][level]["correct"] += 1
                    
                    # Update type accuracy
                    if problem_type not in results["type_accuracy"]:
                        results["type_accuracy"][problem_type] = {"correct": 0, "total": 0}
                    results["type_accuracy"][problem_type]["total"] += 1
                    if is_correct:
                        results["type_accuracy"][problem_type]["correct"] += 1
                    
                    # Store reward metrics
                    results["reward_metrics"]["total_rewards"].append(reward_dict["total_reward"])
                    results["reward_metrics"]["correctness_rewards"].append(reward_dict["correctness_reward"])
                    results["reward_metrics"]["reasoning_rewards"].append(reward_dict["reasoning_reward"])
                    results["reward_metrics"]["step_penalties"].append(reward_dict["step_penalty"])
        
        # Compute final metrics
        results["accuracy"] = results["correct_samples"] / results["total_samples"]
        
        # Compute level accuracies
        for level in results["level_accuracy"]:
            level_data = results["level_accuracy"][level]
            results["level_accuracy"][level]["accuracy"] = level_data["correct"] / level_data["total"]
        
        # Compute type accuracies
        for problem_type in results["type_accuracy"]:
            type_data = results["type_accuracy"][problem_type]
            results["type_accuracy"][problem_type]["accuracy"] = type_data["correct"] / type_data["total"]
        
        # Compute reward statistics
        results["reward_metrics"]["mean_total_reward"] = np.mean(results["reward_metrics"]["total_rewards"])
        results["reward_metrics"]["mean_correctness_reward"] = np.mean(results["reward_metrics"]["correctness_rewards"])
        results["reward_metrics"]["mean_reasoning_reward"] = np.mean(results["reward_metrics"]["reasoning_rewards"])
        results["reward_metrics"]["mean_step_penalty"] = np.mean(results["reward_metrics"]["step_penalties"])
        
        # Print results
        self._print_evaluation_results(results)
        
        # Save results
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _generate_single_solution(self, problem: str) -> str:
        """Generate a single solution without PAG"""
        prompt = f"""Solve the following mathematical problem step by step. Show your reasoning clearly.

Problem: {problem}

Solution:"""
        
        inputs = self.policy_model.base_model.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.model.max_length,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                max_length=self.config.model.max_length,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.policy_model.base_model.tokenizer.pad_token_id
            )
        
        generated_text = self.policy_model.base_model.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        
        if "Solution:" in generated_text:
            solution = generated_text.split("Solution:")[-1].strip()
        else:
            solution = generated_text
        
        return solution
    
    def _print_evaluation_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Samples: {results['correct_samples']}")
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        
        print("\nLevel-wise Accuracy:")
        for level, data in results["level_accuracy"].items():
            print(f"  {level}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
        
        print("\nType-wise Accuracy:")
        for problem_type, data in results["type_accuracy"].items():
            print(f"  {problem_type}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
        
        print("\nReward Metrics:")
        reward_metrics = results["reward_metrics"]
        print(f"  Mean Total Reward: {reward_metrics['mean_total_reward']:.4f}")
        print(f"  Mean Correctness Reward: {reward_metrics['mean_correctness_reward']:.4f}")
        print(f"  Mean Reasoning Reward: {reward_metrics['mean_reasoning_reward']:.4f}")
        print(f"  Mean Step Penalty: {reward_metrics['mean_step_penalty']:.4f}")
        
        # PAG-specific metrics
        num_turns_list = [result["num_turns"] for result in results["per_problem_results"]]
        mean_turns = np.mean(num_turns_list)
        print(f"\nPAG Metrics:")
        print(f"  Mean Number of Turns: {mean_turns:.2f}")
        
        # Verification scores
        all_verification_scores = []
        all_confidence_scores = []
        for result in results["per_problem_results"]:
            all_verification_scores.extend(result["verification_scores"])
            all_confidence_scores.extend(result["confidence_scores"])
        
        if all_verification_scores:
            print(f"  Mean Verification Score: {np.mean(all_verification_scores):.4f}")
        if all_confidence_scores:
            print(f"  Mean Confidence Score: {np.mean(all_confidence_scores):.4f}")
        
        print("="*50)
    
    def _save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert results
        results_serializable = json.loads(json.dumps(results, default=convert_numpy))
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def evaluate_single_problem(self, problem: str, ground_truth: str = None) -> Dict[str, Any]:
        """Evaluate on a single problem"""
        print(f"Evaluating single problem...")
        
        # Generate solution using PAG
        turns = self.pag_model.multi_turn_reasoning(problem)
        
        # Get final solution
        if turns:
            final_solution = turns[-1].generated_solution
        else:
            final_solution = self._generate_single_solution(problem)
        
        # Extract answer
        generated_answer = self.dataset.extract_answer(final_solution)
        
        # Check correctness if ground truth provided
        is_correct = None
        if ground_truth:
            is_correct = self.dataset.is_correct(generated_answer, ground_truth)
        
        # Compute rewards
        reward_dict = self.reward_model.compute_reward(problem, ground_truth or "", final_solution)
        
        result = {
            "problem": problem,
            "generated_solution": final_solution,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "num_turns": len(turns),
            "turns": [
                {
                    "solution": turn.generated_solution,
                    "verification_score": turn.verification_score,
                    "confidence": turn.confidence,
                    "is_correct": turn.is_correct
                }
                for turn in turns
            ],
            "rewards": reward_dict
        }
        
        return result
