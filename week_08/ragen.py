"""
RAGEN with A*PO for WebShop and WebArena - Improved Version
Based on: "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL"

This is an improved implementation with:
- More realistic environment simulations
- Better training stability
- Comprehensive evaluation and comparison
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from policy import WebShopPolicy, WebArenaPolicy
from webshop import WebShopEnvironment
from webarena import WebArenaEnvironment
from astar_po import AStarPOOptimizer


@dataclass
class EpisodeData:
    """Store episode rollout data"""
    states: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    dones: List[bool]
    
    @property
    def total_reward(self) -> float:
        return sum(self.rewards)
    
    @property
    def length(self) -> int:
        return len(self.rewards)


class RAGENTrainer:
    """
    RAGEN Trainer for web environments using A*-PO.
    
    Key features:
    - Multi-turn rollout collection
    - A*-PO optimization with advantage weighting
    - Separate training for WebShop and WebArena
    """
    
    def __init__(
        self,
        env_type: str = "webshop",
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_rollouts_per_update: int = 16,
        max_steps_per_episode: int = 20
    ):
        self.env_type = env_type
        
        # Create environment
        if env_type == "webshop":
            self.env = WebShopEnvironment()
            self.policy = WebShopPolicy(hidden_dim)
        else:
            self.env = WebArenaEnvironment()
            self.policy = WebArenaPolicy(hidden_dim)
        
        # Create optimizer
        self.optimizer = AStarPOOptimizer(
            self.policy,
            lr=lr,
            gamma=gamma
        )
        
        self.n_rollouts_per_update = n_rollouts_per_update
        self.max_steps_per_episode = max_steps_per_episode
        
        # Training statistics
        self.episode_rewards = []
        self.success_history = []
    
    def collect_rollout(self, temperature: float = 1.0) -> EpisodeData:
        """Collect a single episode rollout"""
        state = self.env.reset()
        
        episode = EpisodeData(
            states=[], actions=[], rewards=[],
            log_probs=[], values=[], dones=[]
        )
        
        for step in range(self.max_steps_per_episode):
            # Get policy output
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                logits, value = self.policy(state_tensor)
            
            # Sample action
            probs = F.softmax(logits / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take step
            next_state, reward, done, info = self.env.step(action.item())
            
            # Store transition
            episode.states.append(state)
            episode.actions.append(action.item())
            episode.rewards.append(reward)
            episode.log_probs.append(log_prob.squeeze())
            episode.values.append(value.squeeze())
            episode.dones.append(done)
            
            if done:
                break
            
            state = next_state
        
        return episode
    
    def collect_batch(self, n_rollouts: int, temperature: float = 1.0) -> List[EpisodeData]:
        """Collect batch of rollouts"""
        episodes = []
        for _ in range(n_rollouts):
            episode = self.collect_rollout(temperature)
            episodes.append(episode)
        return episodes
    
    def train_step(self) -> Dict[str, float]:
        """Single training step with multiple rollouts"""
        # Collect rollouts
        episodes = self.collect_batch(
            self.n_rollouts_per_update,
            temperature=1.0
        )
        
        # Aggregate data
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        
        total_reward = 0
        successes = 0
        
        for episode in episodes:
            all_states.extend(episode.states)
            all_actions.extend(episode.actions)
            all_rewards.extend(episode.rewards)
            all_log_probs.extend(episode.log_probs)
            
            total_reward += episode.total_reward
            if episode.total_reward > 0.5:  # Success threshold
                successes += 1
        
        # Convert to tensors
        states = torch.stack(all_states)
        actions = torch.tensor(all_actions, dtype=torch.long)
        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        old_log_probs = torch.stack(all_log_probs).detach()
        
        # Update policy
        metrics = self.optimizer.update(
            states=states,
            actions=actions,
            rewards=rewards,
            old_log_probs=old_log_probs
        )
        
        # Record statistics
        avg_reward = total_reward / len(episodes)
        success_rate = successes / len(episodes)
        
        self.episode_rewards.append(avg_reward)
        self.success_history.append(success_rate)
        
        metrics["avg_reward"] = avg_reward
        metrics["success_rate"] = success_rate
        
        return metrics
    
    def evaluate(self, n_episodes: int = 50) -> Dict[str, float]:
        """Evaluate current policy"""
        total_reward = 0
        successes = 0
        total_steps = 0
        
        for _ in range(n_episodes):
            episode = self.collect_rollout(temperature=0.1)  # Near-greedy
            total_reward += episode.total_reward
            total_steps += episode.length
            
            if episode.total_reward > 0.5:
                successes += 1
        
        return {
            "avg_reward": total_reward / n_episodes,
            "success_rate": successes / n_episodes,
            "avg_steps": total_steps / n_episodes
        }


def train_webshop(n_steps: int = 100, verbose: bool = True) -> RAGENTrainer:
    """Train on WebShop environment"""
    trainer = RAGENTrainer(
        env_type="webshop",
        hidden_dim=64,
        lr=3e-4,
        n_rollouts_per_update=16
    )
    
    if verbose:
        print("Training on WebShop...")
        print("-" * 50)
    
    for step in range(n_steps):
        metrics = trainer.train_step()
        
        if verbose and step % 10 == 0:
            print(f"Step {step:3d} | "
                  f"Reward: {metrics['avg_reward']:.3f} | "
                  f"Success: {metrics['success_rate']:.1%} | "
                  f"Loss: {metrics['total_loss']:.4f}")
    
    return trainer


def train_webarena(n_steps: int = 100, verbose: bool = True) -> RAGENTrainer:
    """Train on WebArena environment"""
    trainer = RAGENTrainer(
        env_type="webarena",
        hidden_dim=64,
        lr=3e-4,
        n_rollouts_per_update=16
    )
    
    if verbose:
        print("Training on WebArena...")
        print("-" * 50)
    
    for step in range(n_steps):
        metrics = trainer.train_step()
        
        if verbose and step % 10 == 0:
            print(f"Step {step:3d} | "
                  f"Reward: {metrics['avg_reward']:.3f} | "
                  f"Success: {metrics['success_rate']:.1%} | "
                  f"Loss: {metrics['total_loss']:.4f}")
    
    return trainer


def analyze_failures(trainer: RAGENTrainer, n_episodes: int = 10):
    """Analyze failure cases"""
    failures = []
    
    for _ in range(n_episodes * 3):  # Collect more to find failures
        episode = trainer.collect_rollout(temperature=0.1)
        if episode.total_reward <= 0.3:  # Failure
            failures.append(episode)
            if len(failures) >= n_episodes:
                break
    
    print(f"\nAnalyzed {len(failures)} failure cases:")
    print("-" * 50)
    
    # Analyze common patterns
    short_episodes = sum(1 for e in failures if e.length < 5)
    timeout_episodes = sum(1 for e in failures if e.length >= trainer.max_steps_per_episode - 1)
    
    print(f"• Episodes too short (< 5 steps): {short_episodes}/{len(failures)}")
    print(f"• Episodes timed out: {timeout_episodes}/{len(failures)}")
    
    # Action distribution in failures
    all_actions = [a for e in failures for a in e.actions]
    if all_actions:
        action_counts = {}
        for a in all_actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        print("\nAction distribution in failures:")
        for action, count in sorted(action_counts.items()):
            pct = count / len(all_actions) * 100
            print(f"  Action {action}: {pct:.1f}%")


def main():
    """Main training and evaluation loop"""
    print("=" * 70)
    print("RAGEN with A*-PO: WebShop and WebArena Evaluation")
    print("=" * 70)
    
    # Train on WebShop
    print("\n[1] WEBSHOP TRAINING")
    print("=" * 50)
    webshop_trainer = train_webshop(n_steps=100, verbose=True)
    
    print("\nWebShop Final Evaluation:")
    webshop_results = webshop_trainer.evaluate(n_episodes=100)
    print(f"  Average Reward: {webshop_results['avg_reward']:.3f}")
    print(f"  Success Rate: {webshop_results['success_rate']:.1%}")
    print(f"  Average Steps: {webshop_results['avg_steps']:.1f}")
    
    # Analyze WebShop failures
    print("\nWebShop Failure Analysis:")
    analyze_failures(webshop_trainer)
    
    # Train on WebArena
    print("\n" + "=" * 70)
    print("[2] WEBARENA TRAINING")
    print("=" * 50)
    webarena_trainer = train_webarena(n_steps=100, verbose=True)
    
    print("\nWebArena Final Evaluation:")
    webarena_results = webarena_trainer.evaluate(n_episodes=100)
    print(f"  Average Reward: {webarena_results['avg_reward']:.3f}")
    print(f"  Success Rate: {webarena_results['success_rate']:.1%}")
    print(f"  Average Steps: {webarena_results['avg_steps']:.1f}")
    
    # Analyze WebArena failures
    print("\nWebArena Failure Analysis:")
    analyze_failures(webarena_trainer)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Environment':<15} {'Avg Reward':<15} {'Success Rate':<15} {'Avg Steps':<12}")
    print("-" * 57)
    print(f"{'WebShop':<15} {webshop_results['avg_reward']:<15.3f} "
          f"{webshop_results['success_rate']:<15.1%} {webshop_results['avg_steps']:<12.1f}")
    print(f"{'WebArena':<15} {webarena_results['avg_reward']:<15.3f} "
          f"{webarena_results['success_rate']:<15.1%} {webarena_results['avg_steps']:<12.1f}")
    
    # Leaderboard comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH LEADERBOARD")
    print("=" * 70)
    print(f"{'Method':<25} {'WebShop':<15} {'WebArena':<15}")
    print("-" * 55)
    print(f"{'SOTA (GPT-4 + tools)':<25} {'~60-70%':<15} {'~35-45%':<15}")
    print(f"{'RAGEN (paper)':<25} {'~50-60%':<15} {'~25-35%':<15}")
    print(f"{'Our A*-PO (simulation)':<25} "
          f"{webshop_results['success_rate']:<15.1%} {webarena_results['success_rate']:<15.1%}")
    
    # Why performance differs
    print("\n" + "=" * 70)
    print("WHY OUR RESULTS DIFFER FROM PAPER/LEADERBOARD")
    print("=" * 70)
    print("""
1. ENVIRONMENT SIMULATION vs REAL
   - Our implementation uses simplified simulations
   - Real WebShop/WebArena have complex HTML/DOM
   - Missing: Visual rendering, JavaScript execution, cookies
   
2. MODEL SCALE
   - Paper uses larger models (7B+ parameters)
   - We use small MLP policies (~10K parameters)
   - Missing: Pre-trained language understanding
   
3. TRAINING COMPUTE
   - Paper: 1000s of training steps, distributed
   - Ours: 100 steps, single process
   - Need ~10x more training for convergence
   
4. OBSERVATION SPACE
   - Paper: Full HTML DOM, screenshots
   - Ours: Compressed state vectors
   - Missing: Rich structural information
   
5. ACTION SPACE
   - Paper: Arbitrary text generation, clicking coordinates
   - Ours: Discrete action set
   - Missing: Fine-grained control

TO IMPROVE:
- Connect to real WebShop/WebArena servers
- Use transformer-based policies
- Increase training duration
- Add curriculum learning
- Implement replay buffers
""")
    
    return webshop_trainer, webarena_trainer


if __name__ == "__main__":
    main()

