"""
Example usage of PAG with A*-PO system
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.qwen_model import QwenModel, PolicyModel, ValueModel
from models.pag_model import PolicyAsGenerativeVerifier
from data.reward_model import MathematicalRewardModel


def example_single_problem():
    """Example: Solve a single mathematical problem using PAG"""
    print("="*60)
    print("Example: Single Problem Solving with PAG")
    print("="*60)
    
    # Initialize models
    print("Initializing models...")
    base_model = QwenModel(config)
    policy_model = PolicyModel(base_model)
    value_model = ValueModel(base_model)
    reward_model = MathematicalRewardModel(config)
    
    # Initialize PAG model
    pag_model = PolicyAsGenerativeVerifier(
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        config=config
    )
    
    # Example problem
    problem = """
    A store sells apples for $2 per pound and oranges for $3 per pound. 
    If Sarah buys 4 pounds of apples and 3 pounds of oranges, 
    how much does she spend in total?
    """
    
    print(f"Problem: {problem.strip()}")
    print("\nSolving with PAG multi-turn reasoning...")
    
    # Solve using PAG
    turns = pag_model.multi_turn_reasoning(problem, max_turns=3)
    
    print(f"\nNumber of reasoning turns: {len(turns)}")
    
    for i, turn in enumerate(turns):
        print(f"\n--- Turn {i+1} ---")
        print(f"Generated Solution: {turn.generated_solution}")
        print(f"Verification Score: {turn.verification_score:.4f}")
        print(f"Confidence: {turn.confidence:.4f}")
        print(f"Is Correct: {turn.is_correct}")
    
    # Compute PAG reward
    if turns:
        final_solution = turns[-1].generated_solution
        reward_dict = pag_model.compute_pag_reward(turns, "")
        
        print(f"\n--- Final Results ---")
        print(f"Final Solution: {final_solution}")
        print(f"Total Reward: {reward_dict['total_reward']:.4f}")
        print(f"Correctness Reward: {reward_dict['correctness_reward']:.4f}")
        print(f"Efficiency Reward: {reward_dict['efficiency_reward']:.4f}")


def example_batch_processing():
    """Example: Process a batch of problems"""
    print("\n" + "="*60)
    print("Example: Batch Problem Processing")
    print("="*60)
    
    # Sample problems
    problems = [
        "What is 15 + 27?",
        "If a rectangle has length 8 and width 5, what is its area?",
        "Solve for x: 2x - 5 = 11"
    ]
    
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
    
    print(f"Processing {len(problems)} problems...")
    
    for i, problem in enumerate(problems):
        print(f"\n--- Problem {i+1} ---")
        print(f"Problem: {problem}")
        
        # Solve with PAG
        turns = pag_model.multi_turn_reasoning(problem, max_turns=2)
        
        if turns:
            final_solution = turns[-1].generated_solution
            print(f"Solution: {final_solution}")
            print(f"Turns used: {len(turns)}")
            print(f"Final verification score: {turns[-1].verification_score:.4f}")
        else:
            print("No solution generated")


def example_reward_analysis():
    """Example: Analyze reward components"""
    print("\n" + "="*60)
    print("Example: Reward Analysis")
    print("="*60)
    
    # Initialize reward model
    reward_model = MathematicalRewardModel(config)
    
    # Test cases with different quality solutions
    test_cases = [
        {
            "problem": "What is 12 × 8?",
            "solution": "12 × 8 = 96",
            "generated": "I multiply 12 by 8 to get 96."
        },
        {
            "problem": "What is 12 × 8?",
            "solution": "12 × 8 = 96", 
            "generated": "12 times 8 equals 96."
        },
        {
            "problem": "What is 12 × 8?",
            "solution": "12 × 8 = 96",
            "generated": "I don't know the answer."
        }
    ]
    
    print("Analyzing reward components for different solution qualities...")
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Problem: {case['problem']}")
        print(f"Correct Solution: {case['solution']}")
        print(f"Generated: {case['generated']}")
        
        # Compute rewards
        reward_dict = reward_model.compute_reward(
            case['problem'], 
            case['solution'], 
            case['generated']
        )
        
        print(f"Reward Breakdown:")
        print(f"  Total Reward: {reward_dict['total_reward']:.4f}")
        print(f"  Correctness: {reward_dict['correctness_reward']:.4f}")
        print(f"  Reasoning: {reward_dict['reasoning_reward']:.4f}")
        print(f"  Step Penalty: {reward_dict['step_penalty']:.4f}")


def main():
    """Run all examples"""
    print("PAG with A*-PO Examples")
    print("This demonstrates the key components of the system.")
    
    try:
        # Example 1: Single problem solving
        example_single_problem()
        
        # Example 2: Batch processing
        example_batch_processing()
        
        # Example 3: Reward analysis
        example_reward_analysis()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
