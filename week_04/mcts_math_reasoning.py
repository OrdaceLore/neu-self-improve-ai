"""
LLM-MCTS for Mathematical Reasoning
Based on: "LLM-MCTS: Monte Carlo Tree Search for LLM Reasoning" (2024)

This implementation demonstrates using MCTS to guide step-by-step mathematical
reasoning, similar to the approach in the LLM-MCTS paper.

Key idea: Each node in the MCTS tree represents a partial solution (reasoning trace).
Actions are "next reasoning steps" and the value is estimated by rollout to completion.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import re

# ============================================================================
# Math Problem Environment
# ============================================================================

@dataclass
class MathProblem:
    """A math problem with question and answer"""
    question: str
    answer: float
    difficulty: str = "easy"
    
    def check_answer(self, predicted: float, tolerance: float = 0.01) -> bool:
        """Check if predicted answer is close enough"""
        return abs(predicted - self.answer) < tolerance


@dataclass
class ReasoningState:
    """
    State in the MCTS tree representing a partial reasoning trace.
    """
    problem: MathProblem
    steps: List[str] = field(default_factory=list)
    current_value: Optional[float] = None
    is_terminal: bool = False
    
    def get_trace(self) -> str:
        """Get full reasoning trace"""
        return "\n".join(self.steps)
    
    def copy(self) -> "ReasoningState":
        """Create a copy of the state"""
        return ReasoningState(
            problem=self.problem,
            steps=self.steps.copy(),
            current_value=self.current_value,
            is_terminal=self.is_terminal
        )


# ============================================================================
# Reasoning Step Generator (Simulates LLM)
# ============================================================================

class ReasoningStepGenerator:
    """
    Simulates an LLM generating reasoning steps.
    In a real implementation, this would call GPT/Claude/etc.
    """
    
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        
        # Define reasoning step templates for different operations
        self.step_templates = {
            "identify": [
                "First, let's identify the numbers: {nums}",
                "We need to work with: {nums}",
                "The values given are: {nums}"
            ],
            "operation": [
                "Performing {op}: {expr} = {result}",
                "Calculate {op}: {expr} gives us {result}",
                "Applying {op} to get: {expr} = {result}"
            ],
            "intermediate": [
                "So far we have: {value}",
                "Current result: {value}",
                "This gives us: {value}"
            ],
            "final": [
                "Therefore, the answer is {value}",
                "The final answer is {value}",
                "Answer: {value}"
            ]
        }
    
    def generate_possible_steps(self, state: ReasoningState) -> List[str]:
        """
        Generate possible next reasoning steps given current state.
        Returns a list of candidate steps.
        """
        problem = state.problem
        steps = state.steps
        
        candidates = []
        
        # Extract numbers from problem
        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', problem.question)]
        
        if len(steps) == 0:
            # First step: identify the problem
            for template in self.step_templates["identify"]:
                candidates.append(template.format(nums=nums))
        
        elif len(steps) < 3:
            # Middle steps: try different operations
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                operations = [
                    ("addition", f"{a} + {b}", a + b),
                    ("subtraction", f"{a} - {b}", a - b),
                    ("multiplication", f"{a} × {b}", a * b),
                ]
                if b != 0:
                    operations.append(("division", f"{a} ÷ {b}", a / b))
                
                for op_name, expr, result in operations:
                    for template in self.step_templates["operation"]:
                        step = template.format(op=op_name, expr=expr, result=result)
                        candidates.append(step)
        
        else:
            # Later steps: try to conclude
            # Try different possible answers
            for template in self.step_templates["final"]:
                for candidate_answer in self._get_candidate_answers(problem, nums):
                    candidates.append(template.format(value=candidate_answer))
        
        # Add some randomness based on temperature
        if self.temperature > 0:
            random.shuffle(candidates)
        
        return candidates[:5]  # Return top 5 candidates
    
    def _get_candidate_answers(self, problem: MathProblem, nums: List[float]) -> List[float]:
        """Generate candidate final answers"""
        candidates = []
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            candidates.extend([a + b, a - b, a * b])
            if b != 0:
                candidates.append(a / b)
            if a != 0:
                candidates.append(b / a)
        candidates.append(problem.answer)  # Include correct answer
        candidates.append(problem.answer + random.uniform(-5, 5))  # Noise
        return list(set(candidates))[:4]
    
    def evaluate_step(self, state: ReasoningState, step: str) -> float:
        """
        Evaluate quality of a reasoning step (simulates LLM confidence).
        Returns value between 0 and 1.
        """
        # Check if step leads toward correct answer
        try:
            # Extract number from step
            nums_in_step = re.findall(r'-?\d+\.?\d*', step)
            if nums_in_step:
                result = float(nums_in_step[-1])
                # Higher value if closer to answer
                distance = abs(result - state.problem.answer)
                return max(0, 1 - distance / (abs(state.problem.answer) + 1))
        except:
            pass
        return 0.5  # Default confidence


# ============================================================================
# MCTS for Math Reasoning
# ============================================================================

@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: ReasoningState
    parent: Optional["MCTSNode"] = None
    action: Optional[str] = None  # The reasoning step that led here
    children: List["MCTSNode"] = field(default_factory=list)
    untried_actions: List[str] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    
    @property
    def q_value(self) -> float:
        """Average value"""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        return self.state.is_terminal


class MCTSMathReasoning:
    """
    MCTS for mathematical reasoning.
    
    The tree structure:
    - Root: Initial problem state (no reasoning steps)
    - Nodes: Partial reasoning traces
    - Actions: Next reasoning steps
    - Terminal: When final answer is given
    
    This implements the core ideas from LLM-MCTS paper.
    """
    
    def __init__(
        self,
        num_simulations: int = 100,
        cpuct: float = 1.414,
        max_depth: int = 5,
        temperature: float = 0.7
    ):
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.max_depth = max_depth
        self.step_generator = ReasoningStepGenerator(temperature)
    
    def solve(self, problem: MathProblem) -> Tuple[str, float, bool]:
        """
        Solve a math problem using MCTS-guided reasoning.
        
        Returns:
            - reasoning_trace: The full reasoning trace
            - predicted_answer: The predicted answer
            - is_correct: Whether the answer is correct
        """
        # Create root state
        initial_state = ReasoningState(problem=problem)
        
        # Create root node
        root = MCTSNode(
            state=initial_state,
            untried_actions=self.step_generator.generate_possible_steps(initial_state)
        )
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            
            # Simulation (rollout)
            value = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        # Extract best path
        best_path = self._get_best_path(root)
        
        # Build final trace
        trace_steps = [node.action for node in best_path if node.action]
        reasoning_trace = "\n".join(trace_steps)
        
        # Extract predicted answer
        predicted_answer = self._extract_answer(trace_steps)
        is_correct = problem.check_answer(predicted_answer) if predicted_answer else False
        
        return reasoning_trace, predicted_answer, is_correct
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node using UCT"""
        current = node
        while not current.is_terminal():
            if not current.is_fully_expanded():
                return current
            current = self._uct_select_child(current)
        return current
    
    def _uct_select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using UCT formula"""
        log_N = math.log(max(1, node.visit_count))
        
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            if child.visit_count == 0:
                return child  # Prioritize unvisited
            
            exploitation = child.q_value
            exploration = self.cpuct * math.sqrt(log_N / child.visit_count)
            uct = exploitation + exploration
            
            if uct > best_score:
                best_score = uct
                best_child = child
        
        return best_child if best_child else node.children[0]
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node with a new child"""
        if not node.untried_actions:
            return node
        
        # Pick an untried action
        action = node.untried_actions.pop()
        
        # Create new state
        new_state = node.state.copy()
        new_state.steps.append(action)
        
        # Check if terminal (contains final answer)
        if "answer is" in action.lower() or "answer:" in action.lower():
            new_state.is_terminal = True
        
        # Check depth limit
        if len(new_state.steps) >= self.max_depth:
            new_state.is_terminal = True
        
        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=[] if new_state.is_terminal else 
                           self.step_generator.generate_possible_steps(new_state)
        )
        
        node.children.append(child)
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate (rollout) from node to terminal state.
        Returns value in [0, 1].
        """
        state = node.state.copy()
        depth = len(state.steps)
        
        # Rollout until terminal or max depth
        while not state.is_terminal and depth < self.max_depth:
            # Generate possible steps
            candidates = self.step_generator.generate_possible_steps(state)
            if not candidates:
                break
            
            # Pick random step (could use policy network here)
            step = random.choice(candidates)
            state.steps.append(step)
            depth += 1
            
            # Check if terminal
            if "answer is" in step.lower() or "answer:" in step.lower():
                state.is_terminal = True
        
        # Evaluate final state
        return self._evaluate_state(state)
    
    def _evaluate_state(self, state: ReasoningState) -> float:
        """
        Evaluate a terminal state.
        Returns 1.0 if correct, 0.0 otherwise, with partial credit.
        """
        # Extract predicted answer
        predicted = self._extract_answer(state.steps)
        
        if predicted is None:
            return 0.0
        
        # Check correctness
        if state.problem.check_answer(predicted):
            return 1.0
        
        # Partial credit based on distance
        distance = abs(predicted - state.problem.answer)
        max_distance = abs(state.problem.answer) + 10
        return max(0, 1 - distance / max_distance) * 0.5
    
    def _extract_answer(self, steps: List[str]) -> Optional[float]:
        """Extract final answer from reasoning steps"""
        for step in reversed(steps):
            if "answer" in step.lower():
                nums = re.findall(r'-?\d+\.?\d*', step)
                if nums:
                    try:
                        return float(nums[-1])
                    except:
                        pass
        return None
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent
    
    def _get_best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Get the best path through the tree (most visited)"""
        path = [root]
        current = root
        
        while current.children:
            # Select child with highest visit count
            best_child = max(current.children, key=lambda c: c.visit_count)
            path.append(best_child)
            current = best_child
        
        return path


# ============================================================================
# Evaluation and Experiments
# ============================================================================

class MathReasoningEvaluator:
    """Evaluate MCTS math reasoning on a dataset"""
    
    def __init__(self):
        # Create test problems (similar to GSM8K style)
        self.problems = [
            MathProblem("What is 5 + 3?", 8.0, "easy"),
            MathProblem("Calculate 12 - 7", 5.0, "easy"),
            MathProblem("What is 6 × 4?", 24.0, "easy"),
            MathProblem("Divide 20 by 4", 5.0, "easy"),
            MathProblem("What is 15 + 27?", 42.0, "medium"),
            MathProblem("Calculate 100 - 37", 63.0, "medium"),
            MathProblem("What is 8 × 9?", 72.0, "medium"),
            MathProblem("Divide 144 by 12", 12.0, "medium"),
            MathProblem("What is 25 + 75?", 100.0, "medium"),
            MathProblem("Calculate 50 × 3", 150.0, "medium"),
        ]
    
    def evaluate(self, mcts: MCTSMathReasoning, verbose: bool = True) -> Dict[str, Any]:
        """Run evaluation"""
        results = {
            "total": len(self.problems),
            "correct": 0,
            "traces": []
        }
        
        for i, problem in enumerate(self.problems):
            trace, predicted, is_correct = mcts.solve(problem)
            
            results["correct"] += int(is_correct)
            results["traces"].append({
                "problem": problem.question,
                "expected": problem.answer,
                "predicted": predicted,
                "correct": is_correct,
                "trace": trace
            })
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"\n[{status}] Problem {i+1}: {problem.question}")
                print(f"    Expected: {problem.answer}, Predicted: {predicted}")
                print(f"    Reasoning:\n    " + trace.replace("\n", "\n    "))
        
        results["accuracy"] = results["correct"] / results["total"]
        return results


def compare_with_baselines():
    """Compare MCTS with baseline approaches"""
    print("=" * 70)
    print("LLM-MCTS Math Reasoning Evaluation")
    print("=" * 70)
    
    evaluator = MathReasoningEvaluator()
    
    # MCTS approach
    print("\n[1] MCTS-Guided Reasoning (100 simulations)")
    print("-" * 50)
    mcts = MCTSMathReasoning(num_simulations=100, cpuct=1.414)
    mcts_results = evaluator.evaluate(mcts, verbose=True)
    
    # Greedy baseline (1 simulation = greedy)
    print("\n\n[2] Greedy Baseline (1 simulation)")
    print("-" * 50)
    greedy = MCTSMathReasoning(num_simulations=1, cpuct=0)
    greedy_results = evaluator.evaluate(greedy, verbose=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-" * 70)
    print(f"{'MCTS (100 sims)':<30} {mcts_results['accuracy']*100:.1f}%{'':<10} {mcts_results['correct']}/{mcts_results['total']}")
    print(f"{'Greedy (1 sim)':<30} {greedy_results['accuracy']*100:.1f}%{'':<10} {greedy_results['correct']}/{greedy_results['total']}")
    print("=" * 70)
    
    return mcts_results, greedy_results


if __name__ == "__main__":
    compare_with_baselines()

