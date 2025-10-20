"""
Training loop for PAG with A*-PO
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import os
import json
import wandb
from datetime import datetime

from models.qwen_model import QwenModel, PolicyModel, ValueModel
from models.pag_model import PolicyAsGenerativeVerifier
from algorithms.astar_po import AStarPO
from data.math_dataset import MATHDataset
from data.reward_model import MathematicalRewardModel


class PAGTrainer:
    """Trainer for PAG with A*-PO"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize A*-PO
        self.astar_po = AStarPO(
            config=config,
            policy_model=self.policy_model,
            value_model=self.value_model,
            reward_model=self.reward_model
        )
        
        # Initialize PAG
        self.pag_model = PolicyAsGenerativeVerifier(
            policy_model=self.policy_model,
            value_model=self.value_model,
            reward_model=self.reward_model,
            config=config
        )
        
        # Initialize dataset
        self.dataset = MATHDataset(config)
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_accuracy = 0.0
        
    def _initialize_models(self):
        """Initialize all models"""
        print("Initializing models...")
        
        # Base Qwen model
        self.base_model = QwenModel(self.config)
        
        # Policy model
        self.policy_model = PolicyModel(self.base_model)
        
        # Value model
        self.value_model = ValueModel(self.base_model)
        
        # Reward model
        self.reward_model = MathematicalRewardModel(self.config)
        
        # Move to device
        self.policy_model = self.policy_model.to(self.device)
        self.value_model = self.value_model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        
        print("Models initialized successfully")
    
    def _setup_logging(self):
        """Setup logging and wandb"""
        if self.config.training.use_wandb:
            wandb.init(
                project="pag-astar-po",
                config=self.config,
                name=f"pag-astar-po-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    
    def load_dataset(self):
        """Load and prepare dataset"""
        print("Loading MATH dataset...")
        self.train_dataset, self.test_dataset = self.dataset.load_dataset()
        
        self.train_dataloader = self.dataset.get_dataloader(self.train_dataset, shuffle=True)
        self.test_dataloader = self.dataset.get_dataloader(self.test_dataset, shuffle=False)
        
        print(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Load dataset
        self.load_dataset()
        
        # Collect offline samples for A*-PO
        print("Collecting offline samples...")
        self.astar_po.collect_offline_samples(
            self.train_dataloader, 
            self.config.astar_po.offline_samples
        )
        
        # Estimate optimal values
        print("Estimating optimal value function...")
        self.astar_po.estimate_optimal_values()
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train for one epoch
            self._train_epoch()
            
            # Evaluate
            if (epoch + 1) % self.config.training.eval_steps == 0:
                self._evaluate()
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_steps == 0:
                self._save_checkpoint()
        
        print("Training completed!")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.policy_model.train()
        self.value_model.train()
        self.reward_model.train()
        
        epoch_losses = []
        epoch_rewards = []
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = self._training_step(batch)
            
            # Update progress
            epoch_losses.append(metrics.get("total_policy_loss", 0))
            epoch_rewards.append(metrics.get("mean_reward", 0))
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{metrics.get('total_policy_loss', 0):.4f}",
                "reward": f"{metrics.get('mean_reward', 0):.4f}",
                "advantage": f"{metrics.get('mean_advantage', 0):.4f}"
            })
            
            # Log metrics
            if self.global_step % self.config.training.logging_steps == 0:
                self._log_metrics(metrics)
            
            self.global_step += 1
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        
        print(f"Epoch {self.epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
        
        if self.config.training.use_wandb:
            wandb.log({
                "epoch": self.epoch + 1,
                "avg_loss": avg_loss,
                "avg_reward": avg_reward
            })
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # A*-PO training step
        astar_metrics = self.astar_po.train_step(batch)
        
        # PAG multi-turn reasoning (for a subset of samples)
        if self.global_step % 10 == 0:  # Every 10 steps
            pag_metrics = self._pag_training_step(batch)
            astar_metrics.update(pag_metrics)
        
        return astar_metrics
    
    def _pag_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PAG training step with multi-turn reasoning"""
        self.pag_model.train()
        
        # Select a subset of samples for PAG training
        batch_size = batch["input_ids"].size(0)
        pag_indices = torch.randperm(batch_size)[:batch_size // 2]  # Half the batch
        
        pag_batch = {
            k: v[pag_indices] if isinstance(v, torch.Tensor) else [v[i] for i in pag_indices]
            for k, v in batch.items()
        }
        
        # Multi-turn reasoning
        all_turns = []
        all_rewards = []
        
        for i in range(len(pag_batch["input_ids"])):
            problem = pag_batch["problems"][i]
            ground_truth = pag_batch["solutions"][i]
            
            # Perform multi-turn reasoning
            turns = self.pag_model.multi_turn_reasoning(problem)
            all_turns.append(turns)
            
            # Compute PAG reward
            pag_reward = self.pag_model.compute_pag_reward(turns, ground_truth)
            all_rewards.append(pag_reward["total_reward"])
        
        # Compute PAG loss
        pag_loss = self._compute_pag_loss(pag_batch, all_turns, all_rewards)
        
        return {
            "pag_loss": pag_loss,
            "mean_pag_reward": np.mean(all_rewards),
            "mean_turns": np.mean([len(turns) for turns in all_turns])
        }
    
    def _compute_pag_loss(self, batch: Dict[str, torch.Tensor], 
                         turns: List, rewards: List[float]) -> float:
        """Compute PAG-specific loss"""
        # This would involve training the verification and confidence heads
        # For now, return a simple loss
        return torch.tensor(0.0, requires_grad=True)
    
    def _evaluate(self):
        """Evaluate the model"""
        print("Evaluating model...")
        
        self.policy_model.eval()
        self.value_model.eval()
        self.reward_model.eval()
        
        total_correct = 0
        total_samples = 0
        all_rewards = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate responses
                outputs = self.policy_model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config.model.max_length,
                    temperature=0.1,  # Low temperature for evaluation
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.policy_model.base_model.tokenizer.pad_token_id
                )
                
                # Compute rewards and accuracy
                for i in range(len(batch["input_ids"])):
                    generated_text = self.policy_model.base_model.tokenizer.decode(
                        outputs[i], skip_special_tokens=True
                    )
                    
                    # Extract answer
                    generated_answer = self.dataset.extract_answer(generated_text)
                    ground_truth = batch["solutions"][i]
                    
                    # Check correctness
                    is_correct = self.dataset.is_correct(generated_answer, ground_truth)
                    if is_correct:
                        total_correct += 1
                    
                    total_samples += 1
                    
                    # Compute reward
                    reward_dict = self.reward_model.compute_reward(
                        batch["problems"][i], ground_truth, generated_text
                    )
                    all_rewards.append(reward_dict["total_reward"])
        
        # Compute metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
        print(f"  Average Reward: {avg_reward:.4f}")
        
        # Log metrics
        if self.config.training.use_wandb:
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/avg_reward": avg_reward,
                "eval/total_samples": total_samples
            })
        
        # Save best model
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_best_model()
        
        return {
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "total_samples": total_samples
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        if self.config.training.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Print to console
        print(f"Step {self.global_step}: {metrics}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.training.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        self.policy_model.save_pretrained(os.path.join(checkpoint_dir, "policy_model"))
        torch.save(self.value_model.state_dict(), os.path.join(checkpoint_dir, "value_model.pt"))
        torch.save(self.reward_model.state_dict(), os.path.join(checkpoint_dir, "reward_model.pt"))
        
        # Save A*-PO state
        self.astar_po.save_checkpoint(os.path.join(checkpoint_dir, "astar_po.pt"))
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_accuracy": self.best_accuracy,
            "config": self.config
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_best_model(self):
        """Save best model"""
        best_dir = os.path.join(self.config.training.output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        
        self.policy_model.save_pretrained(os.path.join(best_dir, "policy_model"))
        torch.save(self.value_model.state_dict(), os.path.join(best_dir, "value_model.pt"))
        torch.save(self.reward_model.state_dict(), os.path.join(best_dir, "reward_model.pt"))
        
        print(f"Best model saved to {best_dir} (accuracy: {self.best_accuracy:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load training state
        with open(os.path.join(checkpoint_path, "training_state.json"), "r") as f:
            training_state = json.load(f)
        
        self.global_step = training_state["global_step"]
        self.epoch = training_state["epoch"]
        self.best_accuracy = training_state["best_accuracy"]
        
        # Load models
        self.policy_model.load_pretrained(os.path.join(checkpoint_path, "policy_model"))
        self.value_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "value_model.pt")))
        self.reward_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "reward_model.pt")))
        
        # Load A*-PO state
        self.astar_po.load_checkpoint(os.path.join(checkpoint_path, "astar_po.pt"))
        
        print("Checkpoint loaded successfully")
