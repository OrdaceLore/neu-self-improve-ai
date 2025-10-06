from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class GameState(ABC):
	"""Abstract two-player zero-sum, turn-based game state.

	Players are represented as +1 (first) and -1 (second). Rewards are from a
	player's perspective in {-1.0, 0.0, +1.0}.
	"""

	@abstractmethod
	def get_current_player(self) -> int:
		"""Return current player in {+1, -1}."""
		raise NotImplementedError

	@abstractmethod
	def get_legal_actions(self) -> List[Any]:
		"""Return the list of legal actions from this state."""
		raise NotImplementedError

	@abstractmethod
	def take_action(self, action: Any) -> "GameState":
		"""Return the next state after applying the given action."""
		raise NotImplementedError

	@abstractmethod
	def is_terminal(self) -> bool:
		"""Return True if the state is terminal (win/loss/draw)."""
		raise NotImplementedError

	@abstractmethod
	def get_reward(self, player: int) -> float:
		"""Return terminal reward from the given player's perspective.

		Must only be called if `is_terminal()` is True. Returns one of
		{-1.0, 0.0, +1.0}.
		"""
		raise NotImplementedError
