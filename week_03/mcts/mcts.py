from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .game import GameState


RolloutPolicy = Callable[[GameState], Any]


@dataclass
class _Node:
	state: GameState
	parent: Optional["_Node"]
	action_from_parent: Optional[Any]
	player_to_move: int
	children: List["_Node"] = field(default_factory=list)
	untried_actions: List[Any] = field(default_factory=list)
	visit_count: int = 0
	total_value_from_player_to_move: float = 0.0

	def is_fully_expanded(self) -> bool:
		return len(self.untried_actions) == 0

	def is_leaf(self) -> bool:
		return len(self.children) == 0


class MCTS:
	"""MCTS with UCT selection for two-player zero-sum games.

	- Values are stored from the perspective of `player_to_move` at each node.
	- During backpropagation, values flip sign when moving to the parent (since
	  players alternate turns).
	"""

	def __init__(
		self,
		num_simulations: int = 800,
		cpuct: float = math.sqrt(2.0),
		rollout_policy: Optional[RolloutPolicy] = None,
		random_seed: Optional[int] = None,
	):
		self.num_simulations = int(num_simulations)
		self.cpuct = float(cpuct)
		self.rollout_policy = rollout_policy or self._random_rollout_policy
		if random_seed is not None:
			random.seed(random_seed)

	def best_action(self, root_state: GameState) -> Any:
		"""Run simulations and return the action with highest visit count."""
		stats = self.search(root_state)
		if not stats:
			raise ValueError("No actions available from the given state")
		return max(stats.items(), key=lambda kv: kv[1]["N"])[0]

	def search(self, root_state: GameState) -> Dict[Any, Dict[str, float]]:
		"""Run MCTS from `root_state` and return per-action statistics.

		Returns mapping: action -> {"N": visits, "Q": mean value for root player}.
		"""
		if root_state.is_terminal():
			return {}

		root_player = root_state.get_current_player()
		root = _Node(
			state=root_state,
			parent=None,
			action_from_parent=None,
			player_to_move=root_player,
			children=[],
			untried_actions=list(root_state.get_legal_actions()),
			visit_count=0,
			total_value_from_player_to_move=0.0,
		)

		for _ in range(self.num_simulations):
			leaf = self._select(root)
			value_for_leaf_player = self._expand_and_simulate(leaf)
			self._backpropagate(leaf, value_for_leaf_player)

		# Compile stats at root by child action
		stats: Dict[Any, Dict[str, float]] = {}
		for child in root.children:
			action = child.action_from_parent
			if action is None:
				continue
			# Convert child's mean value (from child's player perspective) to root player's perspective
			q_child = (
				child.total_value_from_player_to_move / child.visit_count
				if child.visit_count > 0
				else 0.0
			)
			# child.player_to_move is the player at child node; from root player's perspective:
			q_from_root = q_child if child.player_to_move == root_player else -q_child
			stats[action] = {"N": float(child.visit_count), "Q": float(q_from_root)}

		return stats

	def _select(self, node: _Node) -> _Node:
		"""Traverse down the tree using UCT until a node with untried actions or a leaf terminal is found."""
		current = node
		while not current.state.is_terminal():
			if current.untried_actions:
				return current
				# Fully expanded: select child with max UCT
			current = self._uct_select_child(current)
		return current

	def _expand_and_simulate(self, node: _Node) -> float:
		"""Expand one action if possible and then perform a rollout. Returns value for the leaf node's player_to_move."""
		if not node.state.is_terminal() and node.untried_actions:
			action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
			next_state = node.state.take_action(action)
			next_node = _Node(
				state=next_state,
				parent=node,
				action_from_parent=action,
				player_to_move=next_state.get_current_player(),
				children=[],
				untried_actions=list(next_state.get_legal_actions()),
				visit_count=0,
				total_value_from_player_to_move=0.0,
			)
			node.children.append(next_node)
			target_node = next_node
		else:
			target_node = node

		# Rollout from target_node.state until terminal
		rollout_state = target_node.state
		while not rollout_state.is_terminal():
			legal = rollout_state.get_legal_actions()
			if not legal:
				break
			a = self.rollout_policy(rollout_state)
			rollout_state = rollout_state.take_action(a)

		# Value from target node player's perspective
		value = rollout_state.get_reward(target_node.player_to_move)
		return float(value)

	def _backpropagate(self, node: _Node, value_for_node_player: float) -> None:
		"""Backpropagate value up to root.

		`value_for_node_player` is from the perspective of `node.player_to_move`.
		Each step up flips the perspective because players alternate.
		"""
		current = node
		value = value_for_node_player
		while current is not None:
			current.visit_count += 1
			current.total_value_from_player_to_move += value
			# Move to parent: flip perspective
			current = current.parent
			value = -value

	def _uct_select_child(self, node: _Node) -> _Node:
		assert node.children, "UCT selection requires children"
		log_N = math.log(max(1, node.visit_count))

		best_score = -float("inf")
		best_child: Optional[_Node] = None
		for child in node.children:
			if child.visit_count == 0:
				uct = float("inf")
			else:
				q = child.total_value_from_player_to_move / child.visit_count
				uct = q + self.cpuct * math.sqrt(log_N / child.visit_count)
			if uct > best_score:
				best_score = uct
				best_child = child
		assert best_child is not None
		return best_child

	@staticmethod
	def _random_rollout_policy(state: GameState) -> Any:
		legal = state.get_legal_actions()
		return random.choice(legal)
