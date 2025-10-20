"""
Simple test script that doesn't require torch 2.6+
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports without torch operations"""
    print("Testing basic imports...")
    try:
        from config import config
        print("✓ Config imported")
        
        from data.math_dataset import MATHDataset
        print("✓ MATHDataset imported")
        
        from models.qwen_model import QwenModel, PolicyModel, ValueModel
        print("✓ Qwen models imported")
        
        from models.pag_model import PolicyAsGenerativeVerifier
        print("✓ PAG model imported")
        
        from algorithms.astar_po import AStarPO
        print("✓ A*-PO imported")
        
        from algorithms.monte_carlo import MonteCarloGradientEstimator
        print("✓ Monte Carlo imported")
        
        from training.trainer import PAGTrainer
        print("✓ Trainer imported")
        
        from training.evaluator import PAGEvaluator
        print("✓ Evaluator imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_access():
    """Test configuration access"""
    print("Testing configuration access...")
    try:
        from config import config
        
        print(f"  Model: {config.model.model_name}")
        print(f"  Dataset: {config.data.dataset_name}")
        print(f"  Device: {config.training.device}")
        print(f"  Max length: {config.model.max_length}")
        print(f"  Training epochs: {config.training.num_epochs}")
        print(f"  A*-PO learning rate: {config.astar_po.policy_learning_rate}")
        print(f"  PAG max turns: {config.pag.max_turns}")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_simple_math():
    """Test simple mathematical operations"""
    print("Testing simple math...")
    try:
        import numpy as np
        
        # Test basic operations
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 3, 4, 5, 6])
        c = a + b
        
        print(f"  Array addition: {a} + {b} = {c}")
        
        # Test mean
        mean_val = np.mean(c)
        print(f"  Mean: {mean_val}")
        
        return True
    except Exception as e:
        print(f"✗ Math error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("Testing file structure...")
    try:
        required_files = [
            "config.py",
            "requirements.txt",
            "README.md",
            "main.py",
            "test_system.py",
            "example.py",
            "data/__init__.py",
            "data/math_dataset.py",
            "data/reward_model.py",
            "models/__init__.py",
            "models/qwen_model.py",
            "models/pag_model.py",
            "algorithms/__init__.py",
            "algorithms/astar_po.py",
            "algorithms/monte_carlo.py",
            "training/__init__.py",
            "training/trainer.py",
            "training/evaluator.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
            return False
        else:
            print("✓ All required files present")
            return True
            
    except Exception as e:
        print(f"✗ File structure error: {e}")
        return False

def main():
    """Run simple tests"""
    print("="*60)
    print("Simple System Test (No Torch Operations)")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Access", test_config_access),
        ("Simple Math", test_simple_math),
        ("File Structure", test_file_structure),
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
        print("🎉 All basic tests passed! System structure is correct.")
        print("Note: For full functionality, you may need to upgrade PyTorch to 2.6+")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
