## MCTS-UCT Minimal Project

This project provides a clean, reusable implementation of Monte Carlo Tree Search with the UCT rule, a generic two-player zero-sum game interface, and a Tic-Tac-Toe demo with a simple CLI.

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tic-Tac-Toe demo

```bash
python -m mcts.cli --simulations 200 --cpuct 1.414
```

- Human plays `X` (player +1) by default; the agent plays `O` (player -1).
- Use moves as `row col` with zero-based indexing.

### Library overview

- `mcts/game.py`: Abstract `GameState` and helpers for two-player zero-sum turn-based games.
- `mcts/mcts.py`: MCTS-UCT implementation with pluggable rollout policy.
- `mcts/tictactoe.py`: Tic-Tac-Toe environment implementing `GameState`.
- `mcts/cli.py`: Simple CLI to play against the MCTS agent.

### API sketch

```python
from mcts.mcts import MCTS
from mcts.tictactoe import TicTacToe

state = TicTacToe.initial()
agent = MCTS(num_simulations=800, cpuct=1.414)
action = agent.best_action(state)
```

- `MCTS.best_action(state)`: Runs the configured number of simulations from `state` and returns an action.
- `MCTS.search(state)`: Returns a dictionary of action -> visit count and estimated value.

### Extending to other games

Implement `GameState` with:
- `get_current_player() -> int` in {+1, -1}
- `get_legal_actions() -> list[Any]`
- `take_action(action) -> GameState`
- `is_terminal() -> bool`
- `get_reward(player: int) -> float` in {-1.0, 0.0, +1.0}

### Notes

- The implementation uses random rollouts by default. You can pass a custom `rollout_policy(state) -> action` to bias playouts.
- This structure is a good starting point to integrate learned value/policy models (e.g., for MuZero-style planning or LLM-guided expansions).

### Suggested LLM+MCTS paper and plan

- Paper: "LLM-MCTS: An Empirical Study of Monte Carlo Tree Search for Large Language Model Reasoning" (2024). Link: `https://llm-mcts.github.io/static/pdfs/paper.pdf`

Two options:
- Replicate: Implement their self-consistency rollouts with MCTS over chain-of-thought reasoning traces; evaluate on GSM8K-style math problems. Measure accuracy vs. majority-vote baselines at equal token budget.
- Apply to a realistic task: Use MCTS to plan multi-step actions for meeting scheduling emails. Nodes are partial email drafts and decisions (propose time, request info, confirm). Rollouts use an LLM to simulate recipient replies; terminal rewards score feasibility, clarity, and success. Compare to greedy CoT and beam search.

Minimal path to start:
- Implement a tree environment where actions are "next reasoning step" tokens or structured tool calls.
- Use visit-count-based action selection (UCT) and LLM-evaluated value estimates at leafs.
- Constrain token budget per simulation and total rollouts to compare fairly with baselines.
