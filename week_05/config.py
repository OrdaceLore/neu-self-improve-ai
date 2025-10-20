"""
Configuration settings for PAG with A*-PO training
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the Qwen2.5-1.5B-Instruct model"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None

@dataclass
class DataConfig:
    """Configuration for MATH dataset"""
    dataset_name: str = "hendrycks/competition_math"
    train_split: str = "train"
    test_split: str = "test"
    max_train_samples: int = 10000  # Limit for faster training
    max_test_samples: int = 500
    batch_size: int = 4
    num_workers: int = 4

@dataclass
class AStarPOConfig:
    """Configuration for A*-PO algorithm"""
    # Offline stage parameters
    offline_samples: int = 1000
    value_learning_rate: float = 1e-4
    value_epochs: int = 10
    
    # Online stage parameters
    policy_learning_rate: float = 5e-6
    policy_epochs: int = 3
    advantage_regression_weight: float = 1.0
    
    # General parameters
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5

@dataclass
class PAGConfig:
    """Configuration for PAG framework"""
    max_turns: int = 3
    verification_threshold: float = 0.8
    self_correction_weight: float = 0.5
    use_verifier: bool = True

@dataclass
class TrainingConfig:
    """Configuration for training"""
    num_epochs: int = 5
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    use_wandb: bool = False

@dataclass
class RewardConfig:
    """Configuration for reward model"""
    correctness_weight: float = 1.0
    reasoning_weight: float = 0.5
    step_penalty: float = -0.01
    final_reward_scale: float = 1.0
    learning_rate: float = 1e-4

# Global configuration
class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.astar_po = AStarPOConfig()
        self.pag = PAGConfig()
        self.training = TrainingConfig()
        self.reward = RewardConfig()
    
    def __getitem__(self, key):
        return getattr(self, key)

config = Config()
