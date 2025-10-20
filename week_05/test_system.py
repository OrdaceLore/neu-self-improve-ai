"""
Test script to verify the PAG with A*-PO system
"""

import sys
import os
import torch
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data.math_dataset import MATHDataset
from data.reward_model import MathematicalRewardModel
from models.qwen_model import QwenModel, PolicyModel, ValueModel
from models.pag_model import PolicyAsGenerativeVerifier


def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        from algorithms.astar_po import AStarPO
        from algorithms.monte_carlo import MonteCarloGradientEstimator
        from training.trainer import PAGTrainer
        from training.evaluator import PAGEvaluator
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        print(f"  Model: {config['model'].model_name}")
        print(f"  Dataset: {config['data'].dataset_name}")
        print(f"  Device: {config['training'].device}")
        print(f"  Max length: {config['model'].max_length}")
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("Testing dataset loading...")
    try:
        # Create a small test config
        test_config = config
        test_config.data.max_train_samples = 5
        test_config.data.max_test_samples = 3
        
        dataset = MATHDataset(test_config)
        train_dataset, test_dataset = dataset.load_dataset()
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print("✓ Dataset loading successful")
        return True
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        return False


def test_models():
    """Test model initialization"""
    print("Testing model initialization...")
    try:
        # Test Qwen model
        base_model = QwenModel(config)
        print("  ✓ Base model loaded")
        
        # Test policy model
        policy_model = PolicyModel(base_model)
        print("  ✓ Policy model created")
        
        # Test value model
        value_model = ValueModel(base_model)
        print("  ✓ Value model created")
        
        # Test reward model
        reward_model = MathematicalRewardModel(config)
        print("  ✓ Reward model created")
        
        # Test PAG model
        pag_model = PolicyAsGenerativeVerifier(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            config=config
        )
        print("  ✓ PAG model created")
        
        print("✓ All models initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        return False


def test_simple_generation():
    """Test simple text generation"""
    print("Testing simple generation...")
    try:
        # Initialize model
        base_model = QwenModel(config)
        
        # Simple test prompt
        test_prompt = "What is 2 + 2?"
        
        # Tokenize
        inputs = base_model.tokenizer(
            test_prompt,
            return_tensors="pt",
            max_length=100,
            truncation=True
        )
        
        # Generate (with very short length for testing)
        with torch.no_grad():
            outputs = base_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=base_model.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = base_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input: {test_prompt}")
        print(f"  Output: {generated_text[:100]}...")
        print("✓ Simple generation successful")
        return True
    except Exception as e:
        print(f"✗ Generation error: {e}")
        return False


def test_reward_model():
    """Test reward model"""
    print("Testing reward model...")
    try:
        reward_model = MathematicalRewardModel(config)
        
        # Test reward computation
        problem = "What is 2 + 2?"
        solution = "2 + 2 = 4"
        generated_text = "To solve this, I add 2 and 2 to get 4."
        
        reward_dict = reward_model.compute_reward(problem, solution, generated_text)
        
        print(f"  Problem: {problem}")
        print(f"  Solution: {solution}")
        print(f"  Generated: {generated_text}")
        print(f"  Total Reward: {reward_dict['total_reward']:.4f}")
        print("✓ Reward model working")
        return True
    except Exception as e:
        print(f"✗ Reward model error: {e}")
        return False


def test_pag_single_turn():
    """Test PAG single turn reasoning"""
    print("Testing PAG single turn...")
    try:
        # Initialize models
        base_model = QwenModel(config)
        policy_model = PolicyModel(base_model)
        value_model = ValueModel(base_model)
        reward_model = MathematicalRewardModel(config)
        
        pag_model = PolicyAsGenerativeVerifier(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            config=config
        )
        
        # Test problem
        problem = "Solve: 3x + 7 = 22"
        
        # Single turn reasoning
        turns = pag_model.multi_turn_reasoning(problem, max_turns=1)
        
        print(f"  Problem: {problem}")
        print(f"  Number of turns: {len(turns)}")
        if turns:
            print(f"  Generated solution: {turns[0].generated_solution[:100]}...")
            print(f"  Verification score: {turns[0].verification_score:.4f}")
            print(f"  Confidence: {turns[0].confidence:.4f}")
        
        print("✓ PAG single turn successful")
        return True
    except Exception as e:
        print(f"✗ PAG single turn error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("PAG with A*-PO System Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset", test_dataset),
        ("Models", test_models),
        ("Simple Generation", test_simple_generation),
        ("Reward Model", test_reward_model),
        ("PAG Single Turn", test_pag_single_turn),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
