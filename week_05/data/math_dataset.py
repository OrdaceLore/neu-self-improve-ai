"""
MATH dataset loading and preprocessing for mathematical reasoning
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np


class MATHDataset:
    """Dataset class for MATH mathematical reasoning problems"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.train_dataset = None
        self.test_dataset = None
        
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load MATH dataset from HuggingFace"""
        print("Loading MATH dataset...")
        
        # Load the dataset
        dataset = load_dataset(self.config.data.dataset_name)
        
        # Process training data
        train_data = dataset[self.config.data.train_split]
        if self.config.data.max_train_samples:
            train_data = train_data.select(range(min(self.config.data.max_train_samples, len(train_data))))
        
        # Process test data
        test_data = dataset[self.config.data.test_split]
        if self.config.data.max_test_samples:
            test_data = test_data.select(range(min(self.config.data.max_test_samples, len(test_data))))
        
        # Preprocess the data
        self.train_dataset = self._preprocess_dataset(train_data)
        self.test_dataset = self._preprocess_dataset(test_data)
        
        print(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")
        return self.train_dataset, self.test_dataset
    
    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset for training"""
        def process_example(example):
            # Extract problem and solution
            problem = example["problem"]
            solution = example["solution"]
            
            # Create input prompt
            prompt = self._create_prompt(problem)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                max_length=self.config.model.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create target for training
            target_text = f"Problem: {problem}\nSolution: {solution}"
            targets = self.tokenizer(
                target_text,
                max_length=self.config.model.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "target_ids": targets["input_ids"].squeeze(0),
                "problem": problem,
                "solution": solution,
                "level": example.get("level", "unknown"),
                "type": example.get("type", "unknown")
            }
        
        return dataset.map(process_example, batched=False)
    
    def _create_prompt(self, problem: str) -> str:
        """Create a prompt for the mathematical problem"""
        return f"""Solve the following mathematical problem step by step. Show your reasoning clearly.

Problem: {problem}

Solution:"""
    
    def get_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for the dataset"""
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        target_ids = torch.stack([item["target_ids"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "problems": [item["problem"] for item in batch],
            "solutions": [item["solution"] for item in batch],
            "levels": [item["level"] for item in batch],
            "types": [item["type"] for item in batch]
        }
    
    def extract_answer(self, text: str) -> str:
        """Extract the final answer from generated text"""
        # Look for patterns like "The answer is X" or "Answer: X"
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
    
    def is_correct(self, generated_answer: str, ground_truth: str) -> bool:
        """Check if the generated answer is correct"""
        # Normalize answers for comparison
        gen_norm = self._normalize_answer(generated_answer)
        gt_norm = self._normalize_answer(ground_truth)
        
        return gen_norm == gt_norm
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Remove extra whitespace
        answer = answer.strip()
        
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "final answer:", "answer ="]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove punctuation at the end
        answer = answer.rstrip(".,!?")
        
        return answer
