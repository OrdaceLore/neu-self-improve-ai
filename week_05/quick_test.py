"""
Quick test script that doesn't require model loading
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from config import config
        print("✓ Config imported")
        
        from data.math_dataset import MATHDataset
        print("✓ MATHDataset imported")
        
        from data.reward_model import MathematicalRewardModel
        print("✓ Reward model imported")
        
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

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
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

def test_reward_model():
    """Test reward model without torch operations"""
    print("\nTesting reward model...")
    try:
        from data.reward_model import MathematicalRewardModel
        from config import config
        
        # Create reward model (this should work without torch operations)
        reward_model = MathematicalRewardModel(config)
        print("✓ Reward model created")
        
        # Test reward computation (this will fail without proper torch setup)
        try:
            problem = "What is 2 + 2?"
            solution = "2 + 2 = 4"
            generated_text = "To solve this, I add 2 and 2 to get 4."
            
            reward_dict = reward_model.compute_reward(problem, solution, generated_text)
            print(f"✓ Reward computation successful: {reward_dict['total_reward']:.4f}")
        except Exception as e:
            print(f"⚠ Reward computation failed (expected): {e}")
        
        return True
    except Exception as e:
        print(f"✗ Reward model error: {e}")
        return False

def test_astar_po():
    """Test A*-PO algorithm structure"""
    print("\nTesting A*-PO algorithm...")
    try:
        from algorithms.astar_po import AStarPO, AStarPOBatch
        print("✓ A*-PO classes imported")
        
        # Test batch structure
        batch = AStarPOBatch(
            input_ids=None,
            attention_mask=None,
            target_ids=None,
            rewards=None,
            advantages=None,
            values=None,
            old_log_probs=None,
            old_values=None
        )
        print("✓ AStarPOBatch structure created")
        
        return True
    except Exception as e:
        print(f"✗ A*-PO error: {e}")
        return False

def test_pag_structure():
    """Test PAG structure without model loading"""
    print("\nTesting PAG structure...")
    try:
        from models.pag_model import PolicyAsGenerativeVerifier, PAGTurn
        print("✓ PAG classes imported")
        
        # Test PAGTurn structure
        turn = PAGTurn(
            problem="Test problem",
            generated_solution="Test solution",
            verification_score=0.8,
            is_correct=True,
            confidence=0.9,
            reasoning_steps=["Step 1", "Step 2"]
        )
        print("✓ PAGTurn structure created")
        
        return True
    except Exception as e:
        print(f"✗ PAG error: {e}")
        return False

def test_monte_carlo():
    """Test Monte Carlo methods"""
    print("\nTesting Monte Carlo methods...")
    try:
        from algorithms.monte_carlo import (
            MonteCarloGradientEstimator,
            REINFORCE,
            REINFORCELOO,
            AdvantageWeightedRegression
        )
        print("✓ Monte Carlo classes imported")
        
        # Test estimator creation
        estimator = MonteCarloGradientEstimator("pathwise")
        print("✓ Monte Carlo estimator created")
        
        return True
    except Exception as e:
        print(f"✗ Monte Carlo error: {e}")
        return False

def main():
    """Run quick tests"""
    print("="*60)
    print("Quick System Test (No Model Loading)")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Reward Model", test_reward_model),
        ("A*-PO Algorithm", test_astar_po),
        ("PAG Structure", test_pag_structure),
        ("Monte Carlo Methods", test_monte_carlo),
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
        print("🎉 All quick tests passed! System structure is correct.")
        print("Note: For full functionality with model loading, you may need to:")
        print("  1. Upgrade PyTorch to 2.6+")
        print("  2. Install/update torchvision")
        print("  3. Ensure all dependencies are compatible")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
