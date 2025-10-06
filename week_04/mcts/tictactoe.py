from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .game import GameState


BOARD_SIZE = 3


@dataclass(frozen=True)
class TicTacToe(GameState):
	board: Tuple[int, ...]  # length 9, values in {-1, 0, +1}
	player: int  # +1 (X) or -1 (O)

	@staticmethod
	def initial() -> "TicTacToe":
		return TicTacToe(board=(0,) * (BOARD_SIZE * BOARD_SIZE), player=+1)

	def get_current_player(self) -> int:
		return self.player

	def get_legal_actions(self) -> List[int]:
		if self.is_terminal():
			return []
		return [i for i, v in enumerate(self.board) if v == 0]

	def take_action(self, action: int) -> "TicTacToe":
		if self.board[action] != 0:
			raise ValueError("Illegal move")
		new_board = list(self.board)
		new_board[action] = self.player
		next_player = -self.player
		return TicTacToe(board=tuple(new_board), player=next_player)

	def is_terminal(self) -> bool:
		winner = self._winner()
		if winner is not None:
			return True
		return all(v != 0 for v in self.board)

	def get_reward(self, player: int) -> float:
		winner = self._winner()
		if winner is None:
			return 0.0
		if winner == player:
			return 1.0
		if winner == -player:
			return -1.0
		return 0.0

	def _winner(self) -> Optional[int]:
		b = self.board
		lines = [
			# rows
			(0, 1, 2), (3, 4, 5), (6, 7, 8),
			# cols
			(0, 3, 6), (1, 4, 7), (2, 5, 8),
			# diagonals
			(0, 4, 8), (2, 4, 6),
		]
		for i, j, k in lines:
			if b[i] != 0 and b[i] == b[j] == b[k]:
				return b[i]
		return None

	def __str__(self) -> str:
		symbols = {+1: "X", -1: "O", 0: "."}
		rows = []
		for r in range(BOARD_SIZE):
			row = " ".join(symbols[self.board[r * BOARD_SIZE + c]] for c in range(BOARD_SIZE))
			rows.append(row)
		return "\n".join(rows)
