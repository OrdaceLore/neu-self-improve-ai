"""
Main script to run all RL algorithm experiments
Replicates Figure 4.1 and implements all control algorithms on Windy Gridworld
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from figure_4_1_replication import main as run_figure_4_1
from windy_gridworld import test_windy_gridworld
from rl_algorithms import compare_algorithms
from analysis_and_visualization import run_comprehensive_analysis

def main():
    """Run all experiments"""
    print("Reinforcement Learning Algorithm Implementation")
    print("=" * 60)
    print("Based on Sutton & Barto 'Reinforcement Learning: An Introduction'")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # 1. Replicate Figure 4.1
        print("\n1. REPLICATING FIGURE 4.1: Policy Iteration in Gridworld")
        print("-" * 50)
        V_history, policy_history = run_figure_4_1()
        print("✓ Figure 4.1 replication complete!")
        
        # 2. Test Windy Gridworld
        print("\n2. TESTING WINDY GRIDWORLD ENVIRONMENT")
        print("-" * 50)
        env = test_windy_gridworld()
        print("✓ Windy Gridworld environment tested!")
        
        # 3. Compare all algorithms
        print("\n3. COMPARING ALL RL ALGORITHMS")
        print("-" * 50)
        results = compare_algorithms()
        print("✓ Algorithm comparison complete!")
        
        # 4. Comprehensive analysis
        print("\n4. COMPREHENSIVE ANALYSIS AND VISUALIZATION")
        print("-" * 50)
        analysis_results, convergence_data, summary_df = run_comprehensive_analysis()
        print("✓ Comprehensive analysis complete!")
        
        # 5. Summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print("✓ Figure 4.1 Policy Iteration replicated")
        print("✓ Windy Gridworld environment implemented")
        print("✓ All 6 control algorithms implemented:")
        print("  - DP Control (Policy Iteration)")
        print("  - MC On-Policy Control")
        print("  - MC Off-Policy Control")
        print("  - TD(0) On-Policy Control (SARSA)")
        print("  - TD(0) Off-Policy Control (Unweighted IS)")
        print("  - TD(0) Off-Policy Control (Weighted IS)")
        print("✓ Comprehensive analysis and visualizations generated")
        
        print("\nGenerated Files:")
        print("- figure_4_1_replication.png")
        print("- windy_gridworld.png")
        print("- algorithm_comparison.png")
        print("- learning_curves.png")
        print("- optimal_policies.png")
        print("- convergence_analysis.png")
        print("- performance_summary.png")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check the error and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nAll experiments completed successfully!")
    else:
        print("\nExperiments failed. Please check the error messages above.")
