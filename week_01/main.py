import numpy as np
import matplotlib.pyplot as plt

# ----- Bandit Environment -----
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)  # true action values

    def step(self, action):
        return np.random.normal(self.q_true[action], 1)

    def optimal_action(self):
        return np.argmax(self.q_true)


# ----- ε-greedy Agent -----
class EGreedyAgent:
    def __init__(self, k=10, eps=0.1):
        self.k = k
        self.eps = eps
        self.q_est = np.zeros(k)
        self.action_count = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.eps:
            return np.random.randint(self.k)
        return np.argmax(self.q_est)

    def update(self, action, reward):
        self.action_count[action] += 1
        self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]


# ----- Gradient Bandit Agent -----
class GradientAgent:
    def __init__(self, k=10, alpha=0.1, use_baseline=True):
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.H = np.zeros(k)   # preferences
        self.pi = np.ones(k) / k
        self.avg_reward = 0
        self.time = 0

    def select_action(self):
        exp_H = np.exp(self.H - np.max(self.H))  # for stability
        self.pi = exp_H / np.sum(exp_H)
        return np.random.choice(self.k, p=self.pi)

    def update(self, action, reward):
        self.time += 1
        baseline = self.avg_reward if self.use_baseline else 0
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.time
        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        self.H += self.alpha * (reward - baseline) * (one_hot - self.pi)


# ----- Experiment Runner -----
def run(agent_class, agent_args, runs=800, steps=500, k=10):
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))

    for r in range(runs):
        bandit = Bandit(k)
        agent = agent_class(k, **agent_args)
        for t in range(steps):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            rewards[r, t] = reward
            if action == bandit.optimal_action():
                optimal[r, t] = 1
    return rewards.mean(axis=0), optimal.mean(axis=0)


# ----- Main -----
if __name__ == "__main__":
    runs, steps = 800, 500

    # E-greedy settings
    epsilons = [0, 0.01, 0.1]
    eg_results = []
    for eps in epsilons:
        r, o = run(EGreedyAgent, {"eps": eps}, runs, steps)
        eg_results.append((f"ε={eps}", r, o))

    # Gradient bandit settings
    grad_settings = [
        (0.1, False), (0.4, False),
        (0.1, True), (0.4, True)
    ]
    grad_results = []
    for alpha, baseline in grad_settings:
        r, o = run(GradientAgent, {"alpha": alpha, "use_baseline": baseline}, runs, steps)
        name = f"Grad α={alpha}, {'baseline' if baseline else 'no baseline'}"
        grad_results.append((name, r, o))

    # ----- Plotting -----
    plt.figure(figsize=(12, 5))

    # Avg reward
    plt.subplot(1, 2, 1)
    for label, r, _ in eg_results + grad_results:
        plt.plot(r, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()

    # % optimal action
    plt.subplot(1, 2, 2)
    for label, _, o in eg_results + grad_results:
        plt.plot(o * 100, label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()

plt.tight_layout()
plt.savefig("bandit_results.png")
print("Plot saved as bandit_results.png")
