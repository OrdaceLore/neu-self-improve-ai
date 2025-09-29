

import numpy as np
import matplotlib.pyplot as plt
from windy_gridworld import WindyGridworld
from dp_control import DPControl
from mc_on_policy import MCOnPolicyControl, MCEveryVisitControl
from mc_off_policy import MCOffPolicyControl, MCOffPolicyControlOrdinary
from td_on_policy import TDOnPolicyControl, QLearning, ExpectedSarsa
from td_off_policy import TDOffPolicyControl, TDOffPolicyControlOrdinary
from comparison_tool import AlgorithmComparison
import time
import warnings
warnings.filterwarnings('ignore')


def create_figure_4_1_replication():
    """
    Replicate Figure 4.1 from Sutton & Barto showing learning curves
    for different control algorithms on Windy Gridworld.
    """
    print("REPLICATING FIGURE 4.1 FROM SUTTON & BARTO")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # Parameters for experiments
    num_episodes = 1000
    eval_interval = 100
    
    # Store results
    results = {}
    
    # 1. Dynamic Programming (baseline)
    print("\n1. Running Dynamic Programming...")
    dp = DPControl(env)
    dp.policy_iteration()
    dp_returns = dp.evaluate_policy_performance(100)
    results['DP'] = {
        'returns': dp_returns,
        'mean': np.mean(dp_returns),
        'std': np.std(dp_returns),
        'agent': dp
    }
    print(f"   DP Average Return: {np.mean(dp_returns):.2f} ± {np.std(dp_returns):.2f}")
    
    # 2. Monte Carlo On-Policy (First-Visit)
    print("\n2. Running Monte Carlo On-Policy (First-Visit)...")
    mc_first = MCOnPolicyControl(env, epsilon=0.1)
    mc_first_returns, mc_first_eval = mc_first.train(num_episodes, eval_interval)
    results['MC First-Visit'] = {
        'returns': mc_first_returns,
        'eval_returns': mc_first_eval,
        'mean': mc_first.evaluate_policy(100),
        'agent': mc_first
    }
    print(f"   MC First-Visit Final: {results['MC First-Visit']['mean']:.2f}")
    
    # 3. Monte Carlo On-Policy (Every-Visit)
    print("\n3. Running Monte Carlo On-Policy (Every-Visit)...")
    mc_every = MCEveryVisitControl(env, epsilon=0.1)
    mc_every_returns, mc_every_eval = mc_every.train(num_episodes, eval_interval)
    results['MC Every-Visit'] = {
        'returns': mc_every_returns,
        'eval_returns': mc_every_eval,
        'mean': mc_every.evaluate_policy(100),
        'agent': mc_every
    }
    print(f"   MC Every-Visit Final: {results['MC Every-Visit']['mean']:.2f}")
    
    # 4. Monte Carlo Off-Policy (Weighted)
    print("\n4. Running Monte Carlo Off-Policy (Weighted)...")
    mc_off = MCOffPolicyControl(env, epsilon=0.1)
    mc_off_returns, mc_off_eval = mc_off.train_weighted(num_episodes, eval_interval)
    results['MC Off-Policy'] = {
        'returns': mc_off_returns,
        'eval_returns': mc_off_eval,
        'mean': mc_off.evaluate_policy(100),
        'agent': mc_off
    }
    print(f"   MC Off-Policy Final: {results['MC Off-Policy']['mean']:.2f}")
    
    # 5. TD(0) On-Policy (Sarsa)
    print("\n5. Running TD(0) On-Policy (Sarsa)...")
    sarsa = TDOnPolicyControl(env, alpha=0.1, epsilon=0.1)
    sarsa_returns, sarsa_eval = sarsa.train(num_episodes, eval_interval)
    results['Sarsa'] = {
        'returns': sarsa_returns,
        'eval_returns': sarsa_eval,
        'mean': sarsa.evaluate_policy(100),
        'agent': sarsa
    }
    print(f"   Sarsa Final: {results['Sarsa']['mean']:.2f}")
    
    # 6. TD(0) Off-Policy (Q-Learning)
    print("\n6. Running TD(0) Off-Policy (Q-Learning)...")
    q_learning = QLearning(env, alpha=0.1, epsilon=0.1)
    q_learning_returns, q_learning_eval = q_learning.train(num_episodes, eval_interval)
    results['Q-Learning'] = {
        'returns': q_learning_returns,
        'eval_returns': q_learning_eval,
        'mean': q_learning.evaluate_policy(100),
        'agent': q_learning
    }
    print(f"   Q-Learning Final: {results['Q-Learning']['mean']:.2f}")
    
    # 7. TD(0) Off-Policy with Unweighted Importance Sampling
    print("\n7. Running TD(0) Off-Policy (Unweighted IS)...")
    td_off_unweighted = TDOffPolicyControl(env, alpha=0.1, epsilon=0.1)
    td_off_unweighted_returns, td_off_unweighted_eval = td_off_unweighted.train_unweighted(num_episodes, eval_interval)
    results['TD Off-Policy (Unweighted)'] = {
        'returns': td_off_unweighted_returns,
        'eval_returns': td_off_unweighted_eval,
        'mean': td_off_unweighted.evaluate_policy(100),
        'agent': td_off_unweighted
    }
    print(f"   TD Off-Policy (Unweighted) Final: {results['TD Off-Policy (Unweighted)']['mean']:.2f}")
    
    # 8. TD(0) Off-Policy with Weighted Importance Sampling
    print("\n8. Running TD(0) Off-Policy (Weighted IS)...")
    td_off_weighted = TDOffPolicyControl(env, alpha=0.1, epsilon=0.1)
    td_off_weighted_returns, td_off_weighted_eval = td_off_weighted.train_weighted(num_episodes, eval_interval)
    results['TD Off-Policy (Weighted)'] = {
        'returns': td_off_weighted_returns,
        'eval_returns': td_off_weighted_eval,
        'mean': td_off_weighted.evaluate_policy(100),
        'agent': td_off_weighted
    }
    print(f"   TD Off-Policy (Weighted) Final: {results['TD Off-Policy (Weighted)']['mean']:.2f}")
    
    return results, env


def plot_figure_4_1_replication(results, env):
    """
    Create plots similar to Figure 4.1 from Sutton & Barto.
    """
    print("\nGenerating Figure 4.1 Replication...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Windy Gridworld Control Algorithms - Figure 4.1 Replication', fontsize=16, fontweight='bold')
    
    # 1. Learning curves (main plot)
    ax1 = axes[0, 0]
    ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
    
    # Plot evaluation returns for algorithms that have them
    eval_episodes = range(100, 1001, 100)
    
    if 'MC First-Visit' in results:
        ax1.plot(eval_episodes, results['MC First-Visit']['eval_returns'], 'b-o', 
                label='MC First-Visit', linewidth=2, markersize=4)
    
    if 'MC Every-Visit' in results:
        ax1.plot(eval_episodes, results['MC Every-Visit']['eval_returns'], 'c-s', 
                label='MC Every-Visit', linewidth=2, markersize=4)
    
    if 'MC Off-Policy' in results:
        ax1.plot(eval_episodes, results['MC Off-Policy']['eval_returns'], 'm-^', 
                label='MC Off-Policy', linewidth=2, markersize=4)
    
    if 'Sarsa' in results:
        ax1.plot(eval_episodes, results['Sarsa']['eval_returns'], 'g-d', 
                label='Sarsa', linewidth=2, markersize=4)
    
    if 'Q-Learning' in results:
        ax1.plot(eval_episodes, results['Q-Learning']['eval_returns'], 'r-v', 
                label='Q-Learning', linewidth=2, markersize=4)
    
    if 'TD Off-Policy (Unweighted)' in results:
        ax1.plot(eval_episodes, results['TD Off-Policy (Unweighted)']['eval_returns'], 'orange', 
                marker='<', label='TD Off-Policy (Unweighted)', linewidth=2, markersize=4)
    
    if 'TD Off-Policy (Weighted)' in results:
        ax1.plot(eval_episodes, results['TD Off-Policy (Weighted)']['eval_returns'], 'brown', 
                marker='>', label='TD Off-Policy (Weighted)', linewidth=2, markersize=4)
    
    # Add DP baseline
    if 'DP' in results:
        dp_mean = results['DP']['mean']
        ax1.axhline(y=dp_mean, color='black', linestyle='--', linewidth=2, label='DP (Optimal)')
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Return')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Final performance comparison
    ax2 = axes[0, 1]
    ax2.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    
    algorithms = []
    performances = []
    colors = []
    
    for name, result in results.items():
        if name != 'DP':  # DP will be shown separately
            algorithms.append(name)
            performances.append(result['mean'])
            colors.append('skyblue')
    
    # Add DP
    if 'DP' in results:
        algorithms.append('DP (Optimal)')
        performances.append(results['DP']['mean'])
        colors.append('gold')
    
    bars = ax2.bar(range(len(algorithms)), performances, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_ylabel('Average Return')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, perf in zip(bars, performances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{perf:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Optimal policy visualization
    ax3 = axes[0, 2]
    ax3.set_title('Optimal Policy (DP)', fontsize=14, fontweight='bold')
    
    if 'DP' in results:
        optimal_policy = results['DP']['agent'].get_optimal_policy()
        policy_grid = np.argmax(optimal_policy, axis=1).reshape(env.height, env.width)
        
        im = ax3.imshow(policy_grid, cmap='viridis', aspect='equal')
        ax3.set_xlabel('Column')
        ax3.set_ylabel('Row')
        
        # Add start and goal markers
        ax3.plot(env.start[1], env.start[0], 'ws', markersize=12, markeredgecolor='black', markeredgewidth=2)
        ax3.plot(env.goal[1], env.goal[0], 'w*', markersize=16, markeredgecolor='black', markeredgewidth=2)
        
        # Add action labels
        for row in range(env.height):
            for col in range(env.width):
                action = policy_grid[row, col]
                action_symbols = ['↑', '↓', '←', '→']
                ax3.text(col, row, action_symbols[action], ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Action (0=up, 1=down, 2=left, 3=right)')
    
    # 4. Value function visualization
    ax4 = axes[1, 0]
    ax4.set_title('Optimal Value Function (DP)', fontsize=14, fontweight='bold')
    
    if 'DP' in results:
        optimal_values = results['DP']['agent'].get_value_function()
        value_grid = optimal_values.reshape(env.height, env.width)
        
        im = ax4.imshow(value_grid, cmap='viridis', aspect='equal')
        ax4.set_xlabel('Column')
        ax4.set_ylabel('Row')
        
        # Add start and goal markers
        ax4.plot(env.start[1], env.start[0], 'ws', markersize=12, markeredgecolor='black', markeredgewidth=2)
        ax4.plot(env.goal[1], env.goal[0], 'w*', markersize=16, markeredgecolor='black', markeredgewidth=2)
        
        # Add value labels
        for row in range(env.height):
            for col in range(env.width):
                value = value_grid[row, col]
                ax4.text(col, row, f'{value:.1f}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('State Value')
    
    # 5. Wind visualization
    ax5 = axes[1, 1]
    ax5.set_title('Wind Pattern', fontsize=14, fontweight='bold')
    
    # Create wind visualization
    wind_grid = np.array(env.wind).reshape(1, -1)
    im = ax5.imshow(wind_grid, cmap='Blues', aspect='auto')
    ax5.set_xlabel('Column')
    ax5.set_ylabel('Wind Strength')
    ax5.set_yticks([])
    ax5.set_xticks(range(env.width))
    
    # Add wind strength values
    for col in range(env.width):
        wind_strength = env.wind[col]
        ax5.text(col, 0, str(wind_strength), ha='center', va='center',
                fontweight='bold', color='white' if wind_strength > 1 else 'black')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Wind Strength')
    
    # 6. Algorithm summary table
    ax6 = axes[1, 2]
    ax6.set_title('Algorithm Summary', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Create summary data
    summary_data = []
    for name, result in results.items():
        if 'eval_returns' in result:
            # Learning algorithm
            final_eval = result['mean']
            convergence_episode = len(result['eval_returns']) * 100
            summary_data.append([name, f"{final_eval:.2f}", f"{convergence_episode}ep"])
        else:
            # DP
            summary_data.append([name, f"{result['mean']:.2f}", "Optimal"])
    
    # Create table
    if summary_data:
        table = ax6.table(cellText=summary_data,
                         colLabels=['Algorithm', 'Final Return', 'Convergence'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def generate_summary_report(results):
    """Generate a summary report of all results."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY REPORT")
    print("="*60)
    
    print(f"\n{'Algorithm':<25} {'Final Return':<15} {'Type':<15}")
    print("-" * 55)
    
    for name, result in results.items():
        if 'eval_returns' in result:
            algo_type = "Learning"
        else:
            algo_type = "DP"
        print(f"{name:<25} {result['mean']:<15.2f} {algo_type:<15}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    # Find best performing algorithm
    best_algo = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"• Best performing algorithm: {best_algo[0]} ({best_algo[1]['mean']:.2f})")
    
    # Compare to optimal
    if 'DP' in results:
        dp_performance = results['DP']['mean']
        print(f"• Optimal performance (DP): {dp_performance:.2f}")
        
        for name, result in results.items():
            if name != 'DP':
                gap = dp_performance - result['mean']
                print(f"• {name} gap from optimal: {gap:.2f}")
    
    print(f"\n• Total algorithms tested: {len(results)}")
    print(f"• Environment: Windy Gridworld ({env.height}x{env.width})")
    print(f"• Wind pattern: {env.wind}")
    print(f"• Start: {env.start}, Goal: {env.goal}")


def main():
    """Main experiment function."""
    print("WINDY GRIDWORLD CONTROL ALGORITHMS EXPERIMENT")
    print("Replicating Sutton & Barto Figure 4.1")
    print("=" * 60)
    
    # Run all experiments
    results, env = create_figure_4_1_replication()
    
    # Create visualization
    fig = plot_figure_4_1_replication(results, env)
    
    # Generate summary report
    generate_summary_report(results)
    
    # Save results
    print(f"\nSaving results...")
    np.save('experiment_results.npy', results)
    print("Results saved to 'experiment_results.npy'")
    
    return results, env, fig


if __name__ == "__main__":
    results, env, fig = main()
