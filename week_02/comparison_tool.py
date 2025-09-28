"""
Comprehensive Comparison Tool for All Control Algorithms
Replicates Figure 4.1 from Sutton & Barto
"""

import numpy as np
import matplotlib.pyplot as plt
from windy_gridworld import WindyGridworld
from dp_control import DPControl
from mc_on_policy import MCOnPolicyControl, MCEveryVisitControl
from mc_off_policy import MCOffPolicyControl, MCOffPolicyControlOrdinary
from td_on_policy import TDOnPolicyControl, QLearning, ExpectedSarsa
from td_off_policy import TDOffPolicyControl, TDOffPolicyControlOrdinary
import time
from typing import Dict, List, Tuple
import pandas as pd


class AlgorithmComparison:
    """
    Comprehensive comparison of all control algorithms.
    """
    
    def __init__(self, env: WindyGridworld):
        self.env = env
        self.results = {}
        
    def run_dp_experiments(self) -> Dict:
        """Run Dynamic Programming experiments."""
        print("Running Dynamic Programming Experiments...")
        
        # Policy Iteration
        dp_pi = DPControl(self.env)
        start_time = time.time()
        value_history_pi, iter_history_pi = dp_pi.policy_iteration()
        pi_time = time.time() - start_time
        
        # Value Iteration
        dp_vi = DPControl(self.env)
        start_time = time.time()
        value_history_vi, iter_history_vi = dp_vi.value_iteration()
        vi_time = time.time() - start_time
        
        # Evaluate policies
        pi_returns = dp_pi.evaluate_policy_performance(100)
        vi_returns = dp_vi.evaluate_policy_performance(100)
        
        return {
            'policy_iteration': {
                'agent': dp_pi,
                'value_history': value_history_pi,
                'iter_history': iter_history_pi,
                'returns': pi_returns,
                'time': pi_time,
                'final_value': dp_pi.V[self.env.get_state_index(self.env.start)]
            },
            'value_iteration': {
                'agent': dp_vi,
                'value_history': value_history_vi,
                'iter_history': iter_history_vi,
                'returns': vi_returns,
                'time': vi_time,
                'final_value': dp_vi.V[self.env.get_state_index(self.env.start)]
            }
        }
    
    def run_mc_experiments(self, num_episodes: int = 1000) -> Dict:
        """Run Monte Carlo experiments."""
        print("Running Monte Carlo Experiments...")
        
        # First-Visit MC On-Policy
        mc_first = MCOnPolicyControl(self.env, epsilon=0.1)
        start_time = time.time()
        returns_first, eval_returns_first = mc_first.train(num_episodes)
        first_time = time.time() - start_time
        
        # Every-Visit MC On-Policy
        mc_every = MCEveryVisitControl(self.env, epsilon=0.1)
        start_time = time.time()
        returns_every, eval_returns_every = mc_every.train(num_episodes)
        every_time = time.time() - start_time
        
        # MC Off-Policy (Ordinary)
        mc_ordinary = MCOffPolicyControlOrdinary(self.env, epsilon=0.1)
        start_time = time.time()
        returns_ordinary, eval_returns_ordinary = mc_ordinary.train(num_episodes)
        ordinary_time = time.time() - start_time
        
        # MC Off-Policy (Weighted)
        mc_weighted = MCOffPolicyControl(self.env, epsilon=0.1)
        start_time = time.time()
        returns_weighted, eval_returns_weighted = mc_weighted.train_weighted(num_episodes)
        weighted_time = time.time() - start_time
        
        return {
            'first_visit_on_policy': {
                'agent': mc_first,
                'returns': returns_first,
                'eval_returns': eval_returns_first,
                'time': first_time,
                'final_eval': mc_first.evaluate_policy(100)
            },
            'every_visit_on_policy': {
                'agent': mc_every,
                'returns': returns_every,
                'eval_returns': eval_returns_every,
                'time': every_time,
                'final_eval': mc_every.evaluate_policy(100)
            },
            'off_policy_ordinary': {
                'agent': mc_ordinary,
                'returns': returns_ordinary,
                'eval_returns': eval_returns_ordinary,
                'time': ordinary_time,
                'final_eval': mc_ordinary.evaluate_policy(100)
            },
            'off_policy_weighted': {
                'agent': mc_weighted,
                'returns': returns_weighted,
                'eval_returns': eval_returns_weighted,
                'time': weighted_time,
                'final_eval': mc_weighted.evaluate_policy(100)
            }
        }
    
    def run_td_experiments(self, num_episodes: int = 1000) -> Dict:
        """Run TD experiments."""
        print("Running TD Experiments...")
        
        # Sarsa
        sarsa = TDOnPolicyControl(self.env, alpha=0.1, epsilon=0.1)
        start_time = time.time()
        returns_sarsa, eval_returns_sarsa = sarsa.train(num_episodes)
        sarsa_time = time.time() - start_time
        
        # Q-Learning
        q_learning = QLearning(self.env, alpha=0.1, epsilon=0.1)
        start_time = time.time()
        returns_q_learning, eval_returns_q_learning = q_learning.train(num_episodes)
        q_learning_time = time.time() - start_time
        
        # Expected Sarsa
        expected_sarsa = ExpectedSarsa(self.env, alpha=0.1, epsilon=0.1)
        start_time = time.time()
        returns_expected_sarsa, eval_returns_expected_sarsa = expected_sarsa.train(num_episodes)
        expected_sarsa_time = time.time() - start_time
        
        # TD Off-Policy (Ordinary)
        td_ordinary = TDOffPolicyControlOrdinary(self.env, alpha=0.1, epsilon=0.1)
        start_time = time.time()
        returns_td_ordinary, eval_returns_td_ordinary = td_ordinary.train(num_episodes)
        td_ordinary_time = time.time() - start_time
        
        # TD Off-Policy (Unweighted)
        td_unweighted = TDOffPolicyControl(self.env, alpha=0.1, epsilon=0.1)
        start_time = time.time()
        returns_td_unweighted, eval_returns_td_unweighted = td_unweighted.train_unweighted(num_episodes)
        td_unweighted_time = time.time() - start_time
        
        # TD Off-Policy (Weighted)
        td_weighted = TDOffPolicyControl(self.env, alpha=0.1, epsilon=0.1)
        start_time = time.time()
        returns_td_weighted, eval_returns_td_weighted = td_weighted.train_weighted(num_episodes)
        td_weighted_time = time.time() - start_time
        
        return {
            'sarsa': {
                'agent': sarsa,
                'returns': returns_sarsa,
                'eval_returns': eval_returns_sarsa,
                'time': sarsa_time,
                'final_eval': sarsa.evaluate_policy(100)
            },
            'q_learning': {
                'agent': q_learning,
                'returns': returns_q_learning,
                'eval_returns': eval_returns_q_learning,
                'time': q_learning_time,
                'final_eval': q_learning.evaluate_policy(100)
            },
            'expected_sarsa': {
                'agent': expected_sarsa,
                'returns': returns_expected_sarsa,
                'eval_returns': eval_returns_expected_sarsa,
                'time': expected_sarsa_time,
                'final_eval': expected_sarsa.evaluate_policy(100)
            },
            'td_off_policy_ordinary': {
                'agent': td_ordinary,
                'returns': returns_td_ordinary,
                'eval_returns': eval_returns_td_ordinary,
                'time': td_ordinary_time,
                'final_eval': td_ordinary.evaluate_policy(100)
            },
            'td_off_policy_unweighted': {
                'agent': td_unweighted,
                'returns': returns_td_unweighted,
                'eval_returns': eval_returns_td_unweighted,
                'time': td_unweighted_time,
                'final_eval': td_unweighted.evaluate_policy(100)
            },
            'td_off_policy_weighted': {
                'agent': td_weighted,
                'returns': returns_td_weighted,
                'eval_returns': eval_returns_td_weighted,
                'time': td_weighted_time,
                'final_eval': td_weighted.evaluate_policy(100)
            }
        }
    
    def run_all_experiments(self, num_episodes: int = 1000) -> Dict:
        """Run all experiments."""
        print("Running All Control Algorithm Experiments")
        print("=" * 60)
        
        # Run DP experiments
        dp_results = self.run_dp_experiments()
        
        # Run MC experiments
        mc_results = self.run_mc_experiments(num_episodes)
        
        # Run TD experiments
        td_results = self.run_td_experiments(num_episodes)
        
        # Combine all results
        self.results = {
            'dp': dp_results,
            'mc': mc_results,
            'td': td_results
        }
        
        return self.results
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots."""
        if not self.results:
            print("No results available. Run experiments first.")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Learning curves comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_learning_curves(ax1)
        
        # 2. Final performance comparison
        ax2 = plt.subplot(3, 3, 2)
        self._plot_final_performance(ax2)
        
        # 3. Convergence speed comparison
        ax3 = plt.subplot(3, 3, 3)
        self._plot_convergence_speed(ax3)
        
        # 4. DP value function convergence
        ax4 = plt.subplot(3, 3, 4)
        self._plot_dp_convergence(ax4)
        
        # 5. MC learning curves
        ax5 = plt.subplot(3, 3, 5)
        self._plot_mc_learning_curves(ax5)
        
        # 6. TD learning curves
        ax6 = plt.subplot(3, 3, 6)
        self._plot_td_learning_curves(ax6)
        
        # 7. Policy comparison heatmap
        ax7 = plt.subplot(3, 3, 7)
        self._plot_policy_comparison(ax7)
        
        # 8. Value function comparison
        ax8 = plt.subplot(3, 3, 8)
        self._plot_value_comparison(ax8)
        
        # 9. Algorithm summary table
        ax9 = plt.subplot(3, 3, 9)
        self._plot_algorithm_summary(ax9)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_learning_curves(self, ax):
        """Plot learning curves for all algorithms."""
        ax.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
        
        # Plot TD algorithms
        if 'td' in self.results:
            td_results = self.results['td']
            ax.plot(td_results['sarsa']['eval_returns'], 'b-', label='Sarsa', linewidth=2)
            ax.plot(td_results['q_learning']['eval_returns'], 'r-', label='Q-Learning', linewidth=2)
            ax.plot(td_results['expected_sarsa']['eval_returns'], 'g-', label='Expected Sarsa', linewidth=2)
        
        # Plot MC algorithms
        if 'mc' in self.results:
            mc_results = self.results['mc']
            ax.plot(mc_results['first_visit_on_policy']['eval_returns'], 'c-', label='MC First-Visit', linewidth=2)
            ax.plot(mc_results['every_visit_on_policy']['eval_returns'], 'm-', label='MC Every-Visit', linewidth=2)
        
        ax.set_xlabel('Evaluation Episode (×100)')
        ax.set_ylabel('Average Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_performance(self, ax):
        """Plot final performance comparison."""
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        
        algorithms = []
        performances = []
        
        # Collect final performances
        if 'dp' in self.results:
            dp_results = self.results['dp']
            algorithms.extend(['Policy Iteration', 'Value Iteration'])
            performances.extend([
                np.mean(dp_results['policy_iteration']['returns']),
                np.mean(dp_results['value_iteration']['returns'])
            ])
        
        if 'mc' in self.results:
            mc_results = self.results['mc']
            algorithms.extend(['MC First-Visit', 'MC Every-Visit', 'MC Off-Policy'])
            performances.extend([
                mc_results['first_visit_on_policy']['final_eval'],
                mc_results['every_visit_on_policy']['final_eval'],
                mc_results['off_policy_weighted']['final_eval']
            ])
        
        if 'td' in self.results:
            td_results = self.results['td']
            algorithms.extend(['Sarsa', 'Q-Learning', 'Expected Sarsa'])
            performances.extend([
                td_results['sarsa']['final_eval'],
                td_results['q_learning']['final_eval'],
                td_results['expected_sarsa']['final_eval']
            ])
        
        # Create bar plot
        bars = ax.bar(range(len(algorithms)), performances, color=['blue', 'red', 'cyan', 'magenta', 'green', 'orange', 'purple', 'brown'])
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.set_ylabel('Average Return')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, perf in zip(bars, performances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{perf:.2f}', ha='center', va='bottom')
    
    def _plot_convergence_speed(self, ax):
        """Plot convergence speed comparison."""
        ax.set_title('Convergence Speed', fontsize=14, fontweight='bold')
        
        # This would require tracking when each algorithm converges
        # For now, show training time comparison
        algorithms = []
        times = []
        
        if 'dp' in self.results:
            dp_results = self.results['dp']
            algorithms.extend(['Policy Iteration', 'Value Iteration'])
            times.extend([dp_results['policy_iteration']['time'], dp_results['value_iteration']['time']])
        
        if 'mc' in self.results:
            mc_results = self.results['mc']
            algorithms.extend(['MC First-Visit', 'MC Every-Visit'])
            times.extend([mc_results['first_visit_on_policy']['time'], mc_results['every_visit_on_policy']['time']])
        
        if 'td' in self.results:
            td_results = self.results['td']
            algorithms.extend(['Sarsa', 'Q-Learning'])
            times.extend([td_results['sarsa']['time'], td_results['q_learning']['time']])
        
        bars = ax.bar(range(len(algorithms)), times, color=['blue', 'red', 'cyan', 'magenta', 'orange', 'purple'])
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.set_ylabel('Training Time (seconds)')
        ax.grid(True, alpha=0.3)
    
    def _plot_dp_convergence(self, ax):
        """Plot DP convergence."""
        ax.set_title('DP Value Function Convergence', fontsize=14, fontweight='bold')
        
        if 'dp' in self.results:
            dp_results = self.results['dp']
            start_state_idx = self.env.get_state_index(self.env.start)
            
            pi_values = [vh[start_state_idx] for vh in dp_results['policy_iteration']['value_history']]
            vi_values = [vh[start_state_idx] for vh in dp_results['value_iteration']['value_history']]
            
            ax.plot(dp_results['policy_iteration']['iter_history'], pi_values, 'b-o', label='Policy Iteration', linewidth=2)
            ax.plot(dp_results['value_iteration']['iter_history'], vi_values, 'r-s', label='Value Iteration', linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value at Start State')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_mc_learning_curves(self, ax):
        """Plot MC learning curves."""
        ax.set_title('Monte Carlo Learning Curves', fontsize=14, fontweight='bold')
        
        if 'mc' in self.results:
            mc_results = self.results['mc']
            
            ax.plot(mc_results['first_visit_on_policy']['eval_returns'], 'b-o', label='First-Visit', linewidth=2)
            ax.plot(mc_results['every_visit_on_policy']['eval_returns'], 'r-s', label='Every-Visit', linewidth=2)
            ax.plot(mc_results['off_policy_ordinary']['eval_returns'], 'g-^', label='Off-Policy Ordinary', linewidth=2)
            ax.plot(mc_results['off_policy_weighted']['eval_returns'], 'm-d', label='Off-Policy Weighted', linewidth=2)
            
            ax.set_xlabel('Evaluation Episode (×100)')
            ax.set_ylabel('Average Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_td_learning_curves(self, ax):
        """Plot TD learning curves."""
        ax.set_title('TD Learning Curves', fontsize=14, fontweight='bold')
        
        if 'td' in self.results:
            td_results = self.results['td']
            
            ax.plot(td_results['sarsa']['eval_returns'], 'b-o', label='Sarsa', linewidth=2)
            ax.plot(td_results['q_learning']['eval_returns'], 'r-s', label='Q-Learning', linewidth=2)
            ax.plot(td_results['expected_sarsa']['eval_returns'], 'g-^', label='Expected Sarsa', linewidth=2)
            ax.plot(td_results['td_off_policy_ordinary']['eval_returns'], 'm-d', label='TD Off-Policy Ordinary', linewidth=2)
            ax.plot(td_results['td_off_policy_weighted']['eval_returns'], 'c-v', label='TD Off-Policy Weighted', linewidth=2)
            
            ax.set_xlabel('Evaluation Episode (×100)')
            ax.set_ylabel('Average Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_policy_comparison(self, ax):
        """Plot policy comparison heatmap."""
        ax.set_title('Policy Comparison', fontsize=14, fontweight='bold')
        
        # Get optimal policy from DP
        if 'dp' in self.results:
            optimal_policy = self.results['dp']['policy_iteration']['agent'].get_optimal_policy()
            policy_grid = np.argmax(optimal_policy, axis=1).reshape(self.env.height, self.env.width)
            
            im = ax.imshow(policy_grid, cmap='viridis', aspect='equal')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            # Add start and goal markers
            ax.plot(self.env.start[1], self.env.start[0], 'ws', markersize=10, markeredgecolor='black')
            ax.plot(self.env.goal[1], self.env.goal[0], 'w*', markersize=15, markeredgecolor='black')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Action (0=up, 1=down, 2=left, 3=right)')
    
    def _plot_value_comparison(self, ax):
        """Plot value function comparison."""
        ax.set_title('Value Function Comparison', fontsize=14, fontweight='bold')
        
        # Get optimal value function from DP
        if 'dp' in self.results:
            optimal_values = self.results['dp']['policy_iteration']['agent'].get_value_function()
            value_grid = optimal_values.reshape(self.env.height, self.env.width)
            
            im = ax.imshow(value_grid, cmap='viridis', aspect='equal')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            # Add start and goal markers
            ax.plot(self.env.start[1], self.env.start[0], 'ws', markersize=10, markeredgecolor='black')
            ax.plot(self.env.goal[1], self.env.goal[0], 'w*', markersize=15, markeredgecolor='black')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('State Value')
    
    def _plot_algorithm_summary(self, ax):
        """Plot algorithm summary table."""
        ax.set_title('Algorithm Summary', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        
        if 'dp' in self.results:
            dp_results = self.results['dp']
            summary_data.append(['Policy Iteration', 'DP', f"{np.mean(dp_results['policy_iteration']['returns']):.2f}", f"{dp_results['policy_iteration']['time']:.2f}s"])
            summary_data.append(['Value Iteration', 'DP', f"{np.mean(dp_results['value_iteration']['returns']):.2f}", f"{dp_results['value_iteration']['time']:.2f}s"])
        
        if 'mc' in self.results:
            mc_results = self.results['mc']
            summary_data.append(['MC First-Visit', 'MC', f"{mc_results['first_visit_on_policy']['final_eval']:.2f}", f"{mc_results['first_visit_on_policy']['time']:.2f}s"])
            summary_data.append(['MC Every-Visit', 'MC', f"{mc_results['every_visit_on_policy']['final_eval']:.2f}", f"{mc_results['every_visit_on_policy']['time']:.2f}s"])
            summary_data.append(['MC Off-Policy', 'MC', f"{mc_results['off_policy_weighted']['final_eval']:.2f}", f"{mc_results['off_policy_weighted']['time']:.2f}s"])
        
        if 'td' in self.results:
            td_results = self.results['td']
            summary_data.append(['Sarsa', 'TD', f"{td_results['sarsa']['final_eval']:.2f}", f"{td_results['sarsa']['time']:.2f}s"])
            summary_data.append(['Q-Learning', 'TD', f"{td_results['q_learning']['final_eval']:.2f}", f"{td_results['q_learning']['time']:.2f}s"])
            summary_data.append(['Expected Sarsa', 'TD', f"{td_results['expected_sarsa']['final_eval']:.2f}", f"{td_results['expected_sarsa']['time']:.2f}s"])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Algorithm', 'Type', 'Performance', 'Time'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    def generate_report(self) -> str:
        """Generate a comprehensive report."""
        if not self.results:
            return "No results available. Run experiments first."
        
        report = "WINDY GRIDWORLD CONTROL ALGORITHMS COMPARISON REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # DP Results
        if 'dp' in self.results:
            dp_results = self.results['dp']
            report += "DYNAMIC PROGRAMMING RESULTS:\n"
            report += "-" * 30 + "\n"
            report += f"Policy Iteration:\n"
            report += f"  - Final Value at Start: {dp_results['policy_iteration']['final_value']:.2f}\n"
            report += f"  - Average Return: {np.mean(dp_results['policy_iteration']['returns']):.2f} ± {np.std(dp_results['policy_iteration']['returns']):.2f}\n"
            report += f"  - Training Time: {dp_results['policy_iteration']['time']:.2f}s\n"
            report += f"  - Iterations to Converge: {len(dp_results['policy_iteration']['iter_history'])}\n\n"
            
            report += f"Value Iteration:\n"
            report += f"  - Final Value at Start: {dp_results['value_iteration']['final_value']:.2f}\n"
            report += f"  - Average Return: {np.mean(dp_results['value_iteration']['returns']):.2f} ± {np.std(dp_results['value_iteration']['returns']):.2f}\n"
            report += f"  - Training Time: {dp_results['value_iteration']['time']:.2f}s\n"
            report += f"  - Iterations to Converge: {len(dp_results['value_iteration']['iter_history'])}\n\n"
        
        # MC Results
        if 'mc' in self.results:
            mc_results = self.results['mc']
            report += "MONTE CARLO RESULTS:\n"
            report += "-" * 20 + "\n"
            for name, result in mc_results.items():
                report += f"{name.replace('_', ' ').title()}:\n"
                report += f"  - Final Evaluation: {result['final_eval']:.2f}\n"
                report += f"  - Training Time: {result['time']:.2f}s\n\n"
        
        # TD Results
        if 'td' in self.results:
            td_results = self.results['td']
            report += "TEMPORAL DIFFERENCE RESULTS:\n"
            report += "-" * 30 + "\n"
            for name, result in td_results.items():
                report += f"{name.replace('_', ' ').title()}:\n"
                report += f"  - Final Evaluation: {result['final_eval']:.2f}\n"
                report += f"  - Training Time: {result['time']:.2f}s\n\n"
        
        return report


def run_comprehensive_comparison():
    """Run comprehensive comparison of all algorithms."""
    print("WINDY GRIDWORLD COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 60)
    
    # Create environment
    env = WindyGridworld()
    
    # Create comparison tool
    comparison = AlgorithmComparison(env)
    
    # Run all experiments
    results = comparison.run_all_experiments(num_episodes=1000)
    
    # Create comparison plots
    comparison.create_comparison_plots()
    
    # Generate and print report
    report = comparison.generate_report()
    print(report)
    
    # Save report to file
    with open('algorithm_comparison_report.txt', 'w') as f:
        f.write(report)
    
    print("Report saved to 'algorithm_comparison_report.txt'")
    
    return comparison, results


if __name__ == "__main__":
    comparison, results = run_comprehensive_comparison()
