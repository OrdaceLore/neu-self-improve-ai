from __future__ import annotations

import argparse
import sys
from typing import Optional

from .mcts import MCTS
from .tictactoe import TicTacToe


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe vs MCTS")
	parser.add_argument("--simulations", type=int, default=200)
	parser.add_argument("--cpuct", type=float, default=1.414)
	parser.add_argument("--seed", type=int, default=None)
	return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
	args = parse_args(argv)
	agent = MCTS(num_simulations=args.simulations, cpuct=args.cpuct, random_seed=args.seed)
	state = TicTacToe.initial()

	print("You are X (+1). Enter moves as 'row col' (0-based).\n")
	while True:
		print(state)
		if state.is_terminal():
			reward = state.get_reward(+1)
			if reward > 0:
				print("You win!")
			elif reward < 0:
				print("You lose.")
			else:
				print("Draw.")
			break

		if state.get_current_player() == +1:
			move_str = input("Your move (row col): ").strip()
			try:
				r, c = map(int, move_str.split())
			except Exception:
				print("Invalid input. Please enter two integers.")
				continue
			idx = r * 3 + c
			try:
				state = state.take_action(idx)
			except Exception:
				print("Illegal move. Try again.")
				continue
		else:
			stats = agent.search(state)
			if not stats:
				print("No legal moves.")
				break
			best = max(stats.items(), key=lambda kv: kv[1]["N"])[0]
			state = state.take_action(best)

	return 0


if __name__ == "__main__":
	sys.exit(main())
