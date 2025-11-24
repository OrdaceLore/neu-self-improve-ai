# MaAS Assignment: BoolQ Benchmark

This repository contains the application of **Multi-agent Architecture Search (MaAS)** to a new benchmark suite: **BoolQ (Boolean Questions)**.

## 1. Benchmark Selection
* **Benchmark:** BoolQ (Boolean Questions)
* **Leaderboard:** [BoolQ Leaderboard (SuperGLUE)](https://super.gluebenchmark.com/tasks)
* **Rationale:** This dataset tests reading comprehension with yes/no questions based on passages, which differs from the math/code-heavy benchmarks used in the original MaAS paper. It requires understanding context, inference, and logical reasoning from text passages. It is lightweight enough to allow for rapid iteration within the assignment constraints.

## 2. Adaptations & Methodology
To adhere to the assignment constraints (specifically the **3-minute training time**), the following adaptations were made to the standard MaAS codebase:

* **Dataset Subsampling:** The training set was reduced to a "micro-batch" of 10 examples to allow the architecture search to converge quickly. The test set was limited to 50 examples for rapid evaluation.
* **Search Space Constraints:** The maximum agent depth was capped at 3, and the search steps were limited to ensure the genetic algorithm (or evolution strategy) finished within the 180-second window.
* **Custom Formatter:** A `prepare_benchmark.py` script was added to convert the HuggingFace BoolQ dataset into the JSONL format expected by the MaAS agent system.

## 3. Key Findings (Summary)
The architecture search successfully identified that **Sequential Multi-Agent Systems (A->B->C)** significantly outperformed single agents on complex reading comprehension tasks (e.g., multi-hop inference or passage analysis), while single agents often failed on questions requiring deep understanding of context, logical inference, or implicit information extraction.

*(See `MaAS_Analysis_Report.md` for the detailed deep dive on Success/Failure cases)*

---

## 4. How to Reproduce

### Prerequisites
```bash
pip install -r requirements.txt
pip install datasets pandas


python prepare_benchmark.py

python run_assignment.py

python analyze_results.py