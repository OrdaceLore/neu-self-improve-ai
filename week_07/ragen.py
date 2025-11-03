"""
RAGEN: Self-Evolution via Multi-Turn RL with A*PO
Minimal implementation for fast execution
"""
import torch
from policy import RagenPolicy
from frozenlake import SimpleFrozenLake
from astar_po import compute_astar_po_loss

class RAGEN:
    """RAGEN trainer with A*PO"""
    def __init__(self):
        self.env = SimpleFrozenLake()
        self.policy = RagenPolicy(state_dim=16, action_dim=4, hidden_dim=128)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.25)
        self.memory = []  # For multi-turn learning
        self.baseline = 0.0  # Baseline for A*PO weighting
    
    def collect_rollout(self, max_steps=64, temperature=1.0, epsilon=0.0):
        """Collect one episode with optional epsilon-greedy exploration"""
        state = self.env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        
        for step in range(max_steps):
            state_tensor = state.unsqueeze(0)
            
            # Epsilon-greedy: random action with probability epsilon
            if epsilon > 0 and torch.rand(1).item() < epsilon:
                action_val = torch.randint(0, 4, (1,)).item()
                action = torch.tensor(action_val)
                # Compute log_prob for the random action
                with torch.no_grad():
                    logits = self.policy.forward(state_tensor)
                    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                    log_prob = torch.log(probs[0, action_val] + 1e-8)
            else:
                action, log_prob = self.policy.sample(state_tensor, temperature=temperature)
            
            # Ensure log_prob is a scalar tensor with consistent shape
            if log_prob.dim() == 0:
                log_prob = log_prob.unsqueeze(0)
            elif log_prob.shape[0] > 1:
                log_prob = log_prob[0:1]
            
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
        """Compute advantages with discounting"""
        advantages = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            advantages.insert(0, G)
        return torch.tensor(advantages, dtype=torch.float32)
    
    def apo_weight(self, returns, beta=0.8, clip=10.0):
        """A*PO weighting: exp(advantage/beta) with baseline tracking"""
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        adv = returns_tensor - self.baseline
        # Clip to prevent numerical issues
        adv_clipped = torch.clamp(adv / beta, -clip, clip)
        weights = torch.exp(adv_clipped)
        # Update baseline slowly (like working code)
        self.baseline = 0.99 * self.baseline + 0.01 * returns
        return weights
    
    def train_step(self, n_rollouts=100):
        """Single training step with multiple rollouts - inspired by working code"""
        all_log_probs = []
        all_weights = []
        total_reward = 0.0
        positive_count = 0
        
        # Collect multiple rollouts
        for rollout_idx in range(n_rollouts):
            # More exploration to find successful paths
            epsilon = 0.5 if rollout_idx < n_rollouts // 3 else (0.3 if rollout_idx < 2 * n_rollouts // 3 else 0.1)
            states, actions, rewards, log_probs = self.collect_rollout(temperature=1.2, epsilon=epsilon)
            
            if len(log_probs) == 0:
                continue
            
            # Compute returns (sum of rewards)
            returns = sum(rewards)
            total_reward += returns
            
            if returns > 0:
                positive_count += 1
            
            # A*PO weighting: exp(advantage/beta)
            weights = self.apo_weight(returns)
            
            # Stack log_probs and apply weights
            log_probs_tensor = torch.stack(log_probs)
            
            all_log_probs.append(log_probs_tensor)
            all_weights.append(weights.repeat(len(log_probs)))
        
        if len(all_log_probs) == 0:
            return 0.0, 0.0
        
        # Concatenate all
        log_probs_all = torch.cat(all_log_probs)
        weights_all = torch.cat(all_weights)
        
        # Weighted policy loss: -weights * log_probs
        loss = -(weights_all * log_probs_all).mean()
        
        # Note: Entropy bonus would require recomputing from states which we don't store
        # Skip for now - the A*PO weighting should provide enough signal
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        avg_reward = total_reward / n_rollouts if n_rollouts > 0 else 0.0
        return loss.item(), avg_reward
    
    def evaluate(self, n_episodes=100):
        """Evaluate policy with greedy actions"""
        total_reward = 0
        successes = 0
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            for _ in range(64):  # Increased max steps
                state_tensor = state.unsqueeze(0)
                # Greedy evaluation: use argmax (temperature -> 0)
                with torch.no_grad():
                    logits = self.policy.forward(state_tensor)
                    action = torch.argmax(logits, dim=-1)
                
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward
                if done:
                    # Success = positive total reward (goal gives 1.2, holes give -1.0)
                    if episode_reward > 0:
                        successes += 1
                    break
                state = next_state
            total_reward += episode_reward
        return total_reward / n_episodes, successes / n_episodes

def main():
    print("RAGEN with A*PO on FrozenLake")
    print("=" * 50)
    
    trainer = RAGEN()
    
    print("\nTraining...")
    for i in range(5):  # Just 5 steps for fast execution
        loss, episode_reward = trainer.train_step(n_rollouts=100)
        if i % 1 == 0:
            eval_reward, success_rate = trainer.evaluate(n_episodes=100)
            print(f"Step {i}: Loss={loss:.4f}, Episode Reward={episode_reward:.2f}, "
                  f"Eval Reward={eval_reward:.2f}, Success Rate={success_rate:.2%}")
    
    print("\nFinal Evaluation:")
    eval_reward, success_rate = trainer.evaluate(n_episodes=100)
    print(f"Average Reward: {eval_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
