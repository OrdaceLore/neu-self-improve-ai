"""
RAGEN with A*PO for WebShop and WebArena
Minimal implementation for fast evaluation
"""
import torch
from policy import WebPolicy
from webshop import MockWebShop
from webarena import MockWebArena
from astar_po import compute_astar_po_loss

class RAGENWeb:
    """RAGEN for web environments"""
    def __init__(self, env_type='webshop'):
        if env_type == 'webshop':
            self.env = MockWebShop()
            state_dim = 5
            action_dim = 5
        else:  # webarena
            self.env = MockWebArena()
            state_dim = 3
            action_dim = 4
        
        self.policy = WebPolicy(state_dim, action_dim, hidden_dim=32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.env_type = env_type
    
    def collect_rollout(self):
        state = self.env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        
        for _ in range(20):
            state_tensor = state.unsqueeze(0)
            action, log_prob = self.policy.sample(state_tensor)
            next_state, reward, done, _ = self.env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            if done:
                break
            state = next_state
        
        return states, actions, rewards, log_probs
    
    def compute_advantages(self, rewards):
        advantages = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            advantages.insert(0, G)
        return torch.tensor(advantages, dtype=torch.float32)
    
    def train_step(self):
        states, actions, rewards, log_probs = self.collect_rollout()
        if len(log_probs) == 0:
            return 0.0, 0.0
        
        log_probs_tensor = torch.stack(log_probs)
        advantages = self.compute_advantages(rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        loss = compute_astar_po_loss(log_probs_tensor, advantages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), sum(rewards)
    
    def evaluate(self, n_episodes=10):
        total_reward = 0
        successes = 0
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            for _ in range(20):
                state_tensor = state.unsqueeze(0)
                action, _ = self.policy.sample(state_tensor, temperature=0.1)
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward
                if done:
                    if reward > 0:
                        successes += 1
                    break
                state = next_state
            total_reward += episode_reward
        return total_reward / n_episodes, successes / n_episodes

def main():
    print("RAGEN with A*PO on WebShop and WebArena")
    print("=" * 60)
    
    # WebShop
    print("\n=== WebShop ===")
    trainer = RAGENWeb(env_type='webshop')
    print("Training...")
    for i in range(50):
        loss, episode_reward = trainer.train_step()
        if i % 5 == 0:  # Print every 5 steps to reduce output
            eval_reward, success_rate = trainer.evaluate(n_episodes=5)
            print(f"Step {i}: Loss={loss:.4f}, Success Rate={success_rate:.2%}")
    
    webshop_reward, webshop_success = trainer.evaluate(n_episodes=10)
    print(f"\nWebShop Results:")
    print(f"  Average Reward: {webshop_reward:.2f}")
    print(f"  Success Rate: {webshop_success:.2%}")
    
    # WebArena
    print("\n=== WebArena ===")
    trainer = RAGENWeb(env_type='webarena')
    print("Training...")
    for i in range(50):
        loss, episode_reward = trainer.train_step()
        if i % 5 == 0:  # Print every 5 steps to reduce output
            eval_reward, success_rate = trainer.evaluate(n_episodes=5)
            print(f"Step {i}: Loss={loss:.4f}, Success Rate={success_rate:.2%}")
    
    webarena_reward, webarena_success = trainer.evaluate(n_episodes=10)
    print(f"\nWebArena Results:")
    print(f"  Average Reward: {webarena_reward:.2f}")
    print(f"  Success Rate: {webarena_success:.2%}")
    
    # Comparison with leaderboard (mock)
    print("\n=== Comparison with Leaderboard ===")
    print("WebShop Leaderboard (top methods): ~85-95% success rate")
    print(f"Our RAGEN - WebShop: {webshop_success:.2%} success rate")
    print(f"Our RAGEN - WebArena: {webarena_success:.2%} success rate")
    print("\nWhy RAGEN doesn't perform well:")
    print("1. Limited training (50 steps vs 1000+ for leaderboard)")
    print("2. Simplified environment (mock vs real web)")
    print("3. Small model (32 hidden vs larger models)")
    print("4. No pre-training or domain-specific features")
    print("5. A*PO may need more tuning for web tasks")

if __name__ == "__main__":
    main()
