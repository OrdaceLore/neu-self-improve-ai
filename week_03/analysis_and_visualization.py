"""
Comprehensive Analysis and Visualization for RL Algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from windy_gridworld import WindyGridworld
from rl_algorithms import *

def plot_learning_curves(algorithms, num_episodes=1000):
    """Plot learning curves for all algorithms"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    algorithm_names = [
        'MC On-Policy', 'MC Off-Policy', 
        'TD(0) On-Policy', 'TD(0) Off-Policy (Unweighted)', 
        'TD(0) Off-Policy (Weighted)'
    ]
    
    for i, name in enumerate(algorithm_names):
        if name in algorithms:
            algorithm = algorithms[name]
            episode_rewards = algorithm.train(num_episodes=num_episodes)
            
            # Plot learning curve
            axes[i].plot(episode_rewards, alpha=0.7, linewidth=1)
            
            # Plot moving average
            window = 50
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                axes[i].plot(range(window-1, len(episode_rewards)), moving_avg, 
                           color='red', linewidth=2, label=f'Moving Avg ({window})')
            
            axes[i].set_title(f'{name} Learning Curve')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel('Total Reward')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_optimal_policies(algorithms):
    """Visualize optimal policies found by each algorithm"""
    env = WindyGridworld()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    algorithm_names = list(algorithms.keys())
    
    for i, (name, algorithm) in enumerate(algorithms.items()):
        if i >= 6:  # Only plot first 6 algorithms
            break
            
        ax = axes[i]
        
        # Get optimal policy
        if hasattr(algorithm, 'get_optimal_policy'):
            optimal_policy = algorithm.get_optimal_policy()
        else:
            # For algorithms without explicit optimal policy, use Q-values
            optimal_policy = {}
            for state in algorithm.Q:
                optimal_policy[state] = np.argmax(algorithm.Q[state])
        
        # Draw grid
        for row in range(env.height + 1):
            ax.axhline(row-0.5, color='black', linewidth=0.5)
        for col in range(env.width + 1):
            ax.axvline(col-0.5, color='black', linewidth=0.5)
        
        # Color cells based on wind strength
        for j in range(env.width):
            wind = env.wind_strength[j]
            color_intensity = wind / max(env.wind_strength) if max(env.wind_strength) > 0 else 0
            for i_row in range(env.height):
                rect = plt.Rectangle((j-0.5, i_row-0.5), 1, 1, 
                                   facecolor=plt.cm.Blues(color_intensity * 0.3),
                                   edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # Mark start and goal
        start_rect = plt.Rectangle((env.start[1]-0.4, env.start[0]-0.4), 0.8, 0.8,
                                 facecolor='green', edgecolor='black', linewidth=2)
        ax.add_patch(start_rect)
        ax.text(env.start[1], env.start[0], 'S', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
        
        goal_rect = plt.Rectangle((env.goal[1]-0.4, env.goal[0]-0.4), 0.8, 0.8,
                                facecolor='red', edgecolor='black', linewidth=2)
        ax.add_patch(goal_rect)
        ax.text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')
        
        # Draw policy arrows
        for state, action_idx in optimal_policy.items():
            if state != env.goal:  # Don't draw arrow from goal
                action = env.actions[action_idx]
                arrow = FancyArrowPatch((state[1], state[0]), 
                                      (state[1] + action[1]*0.3, state[0] + action[0]*0.3),
                                      arrowstyle='->', mutation_scale=15,
                                      color='red', linewidth=2)
                ax.add_patch(arrow)
        
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(-0.5, env.height - 0.5)
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
        ax.set_title(f'{name} Optimal Policy')
        ax.invert_yaxis()
    
    # Remove empty subplots
    for i in range(len(algorithm_names), 6):
        axes[i].remove()
    
    plt.tight_layout()
    plt.savefig('optimal_policies.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_convergence_speed(algorithms, num_episodes=1000):
    """Analyze convergence speed of different algorithms"""
    convergence_data = {}
    
    for name, algorithm in algorithms.items():
        if name == 'DP Control':
            # DP converges immediately
            convergence_data[name] = {'episodes': 0, 'final_reward': algorithm.evaluate_policy()[0]}
        else:
            episode_rewards = algorithm.train(num_episodes=num_episodes)
            
            # Find convergence point (when moving average stabilizes)
            window = 50
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                
                # Find when moving average stabilizes (changes by less than 1% for 100 episodes)
                for i in range(100, len(moving_avg)):
                    recent_avg = np.mean(moving_avg[i-50:i])
                    if abs(moving_avg[i] - recent_avg) / abs(recent_avg) < 0.01:
                        convergence_episode = i + window - 1
                        break
                else:
                    convergence_episode = len(episode_rewards) - 1
            else:
                convergence_episode = len(episode_rewards) - 1
            
            convergence_data[name] = {
                'episodes': convergence_episode,
                'final_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else episode_rewards[-1]
            }
    
    # Plot convergence analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    names = list(convergence_data.keys())
    episodes = [convergence_data[name]['episodes'] for name in names]
    rewards = [convergence_data[name]['final_reward'] for name in names]
    
    # Episodes to convergence
    bars1 = ax1.bar(names, episodes, alpha=0.7, color='skyblue')
    ax1.set_title('Episodes to Convergence')
    ax1.set_ylabel('Episodes')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, episodes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value}', ha='center', va='bottom')
    
    # Final performance
    bars2 = ax2.bar(names, rewards, alpha=0.7, color='lightcoral')
    ax2.set_title('Final Performance')
    ax2.set_ylabel('Average Reward')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return convergence_data

def create_summary_table(results):
    """Create a summary table of all results"""
    data = []
    
    for name, result in results.items():
        data.append({
            'Algorithm': name,
            'Mean Reward': f"{result['mean_reward']:.2f}",
            'Std Reward': f"{result['std_reward']:.2f}",
            'Performance': 'High' if result['mean_reward'] > -20 else 'Medium' if result['mean_reward'] > -50 else 'Low'
        })
    
    df = pd.DataFrame(data)
    
    # Create a nice table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color code performance
    for i in range(1, len(df) + 1):
        performance = df.iloc[i-1]['Performance']
        if performance == 'High':
            color = 'lightgreen'
        elif performance == 'Medium':
            color = 'lightyellow'
        else:
            color = 'lightcoral'
        
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Algorithm Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def run_comprehensive_analysis():
    """Run complete analysis of all algorithms"""
    print("Running Comprehensive RL Algorithm Analysis")
    print("=" * 50)
    
    # Create environment
    env = WindyGridworld()
    
    # Initialize and train all algorithms
    algorithms = {
        'DP Control': DPControl(env),
        'MC On-Policy': MCOnPolicyControl(env, epsilon=0.1),
        'MC Off-Policy': MCOffPolicyControl(env, epsilon=0.1),
        'TD(0) On-Policy': TDOnPolicyControl(env, epsilon=0.1, alpha=0.1),
        'TD(0) Off-Policy (Unweighted)': TDOffPolicyControl(env, epsilon=0.1, alpha=0.1, weighted=False),
        'TD(0) Off-Policy (Weighted)': TDOffPolicyControl(env, epsilon=0.1, alpha=0.1, weighted=True)
    }
    
    # Train algorithms
    results = {}
    for name, algorithm in algorithms.items():
        print(f"\nTraining {name}...")
        
        if name == 'DP Control':
            iterations = algorithm.train()
            print(f"DP converged in {iterations} iterations")
        else:
            episode_rewards = algorithm.train(num_episodes=1000)
            print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
        
        # Evaluate final policy
        mean_reward, std_reward = algorithm.evaluate_policy(num_episodes=100)
        results[name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'algorithm': algorithm
        }
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Learning curves
    plot_learning_curves(algorithms)
    
    # 2. Optimal policies
    visualize_optimal_policies(algorithms)
    
    # 3. Convergence analysis
    convergence_data = analyze_convergence_speed(algorithms)
    
    # 4. Summary table
    summary_df = create_summary_table(results)
    
    print("\nAnalysis complete! Generated files:")
    print("- learning_curves.png")
    print("- optimal_policies.png") 
    print("- convergence_analysis.png")
    print("- performance_summary.png")
    
    return results, convergence_data, summary_df

if __name__ == "__main__":
    results, convergence_data, summary_df = run_comprehensive_analysis()
