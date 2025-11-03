"""
TinyZero: Minimal implementation of DeepSeek R1 Zero with A*PO
Simplified for fast execution (seconds instead of hours)
"""
import torch
import torch.nn.functional as F
from policy import TinyPolicy
from environment import CountdownEnv, MultiplicationEnv
from astar_po import compute_astar_po_loss

class TinyZero:
    """TinyZero trainer with A*PO"""
    def __init__(self, task='countdown'):
        self.task = task
        self.env = CountdownEnv() if task == 'countdown' else MultiplicationEnv()
        self.policy = TinyPolicy(vocab_size=100, hidden_dim=32)  # Tiny model
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
    
    def collect_rollout(self, max_steps=5):
        """Collect one rollout"""
        state = self.env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        
        for _ in range(max_steps):
            # Ensure state is integer and in valid range
            state_int = state.long().clamp(0, 99)
            state_tensor = state_int.unsqueeze(0)
            action, log_prob = self.policy.sample(state_tensor, temperature=1.0)
            next_state, reward, done, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            if done:
                break
            state = next_state
        
        return states, actions, rewards, log_probs
    
    def compute_advantages(self, rewards):
        """Simple advantage: discounted rewards"""
        if len(rewards) == 0:
            return torch.tensor([], dtype=torch.float32)
        advantages = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            advantages.insert(0, G)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        # Handle case where all advantages are same (std=0)
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages
    
    def train_step(self):
        """Single training step"""
        states, actions, rewards, log_probs = self.collect_rollout()
        
        if len(log_probs) == 0:
            return 0.0
        
        log_probs_tensor = torch.stack(log_probs)
        advantages = self.compute_advantages(rewards)
        
        loss = compute_astar_po_loss(log_probs_tensor, advantages)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, n_episodes=5):
        """Quick evaluation"""
        total_reward = 0
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            for _ in range(10):
                state_int = state.long().clamp(0, 99)
                state_tensor = state_int.unsqueeze(0)
                action, _ = self.policy.sample(state_tensor, temperature=0.1)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if done:
                    break
                state = next_state
            total_reward += episode_reward
        return total_reward / n_episodes

def main():
    print("TinyZero with A*PO - Minimal Training")
    print("=" * 50)
    
    # Countdown task
    print("\nTraining on Countdown task...")
    trainer = TinyZero(task='countdown')
    for i in range(3):  # Just 3 steps for fast execution
        loss = trainer.train_step()
        if i % 1 == 0:
            eval_reward = trainer.evaluate(n_episodes=3)
            print(f"Step {i}: Loss={loss:.4f}, Eval Reward={eval_reward:.2f}")
    
    # Multiplication task
    print("\nTraining on Multiplication task...")
    trainer = TinyZero(task='multiplication')
    for i in range(3):
        loss = trainer.train_step()
        if i % 1 == 0:
            eval_reward = trainer.evaluate(n_episodes=3)
            print(f"Step {i}: Loss={loss:.4f}, Eval Reward={eval_reward:.2f}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
