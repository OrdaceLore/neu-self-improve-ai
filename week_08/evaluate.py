"""
Evaluation script for RAGEN on WebShop and WebArena
Generates performance tables
"""
from ragen import RAGENWeb

def evaluate_webshop():
    """Evaluate on WebShop"""
    trainer = RAGENWeb(env_type='webshop')
    
    # Thorough training for evaluation
    print("Training WebShop model...")
    for i in range(50):
        trainer.train_step()
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/50 training steps")
    
    # Evaluation
    avg_reward, success_rate = trainer.evaluate(n_episodes=20)
    
    return {
        'task': 'WebShop',
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'training_steps': 50
    }

def evaluate_webarena():
    """Evaluate on WebArena"""
    trainer = RAGENWeb(env_type='webarena')
    
    # Thorough training for evaluation
    print("Training WebArena model...")
    for i in range(50):
        trainer.train_step()
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/50 training steps")
    
    # Evaluation
    avg_reward, success_rate = trainer.evaluate(n_episodes=20)
    
    return {
        'task': 'WebArena',
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'training_steps': 50
    }

def main():
    print("Performance Evaluation")
    print("=" * 60)
    
    webshop_results = evaluate_webshop()
    webarena_results = evaluate_webarena()
    
    print("\nPerformance Table:")
    print(f"{'Task':<15} {'Avg Reward':<15} {'Success Rate':<15} {'Training Steps':<15}")
    print("-" * 60)
    print(f"{webshop_results['task']:<15} {webshop_results['avg_reward']:<15.2f} "
          f"{webshop_results['success_rate']:<15.2%} {webshop_results['training_steps']:<15}")
    print(f"{webarena_results['task']:<15} {webarena_results['avg_reward']:<15.2f} "
          f"{webarena_results['success_rate']:<15.2%} {webarena_results['training_steps']:<15}")
    
    print("\nLeaderboard Comparison (Mock Data):")
    print(f"{'Method':<20} {'WebShop':<15} {'WebArena':<15}")
    print("-" * 50)
    print(f"{'Top Method':<20} {'95%':<15} {'92%':<15}")
    print(f"{'RAGEN (ours)':<20} "
          f"{webshop_results['success_rate']:<15.1%} "
          f"{webarena_results['success_rate']:<15.1%}")

if __name__ == "__main__":
    main()
