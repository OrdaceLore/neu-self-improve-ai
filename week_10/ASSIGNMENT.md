# Reading

- [Multi-agent Architecture Search via Agentic Supernet](https://www.arxiv.org/pdf/2502.04180)
- Skim: [Archon: An Architecture Search Framework for Inference-Time Techniques](https://www.arxiv.org/abs/2409.15254)
- Skim: [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al. 2024)](https://arxiv.org/abs/2408.03314)

# Assignment

Run the MaAS system on a benchmark suite other than the ones used in the paper. The github repo is https://github.com/bingreeky/MaAS.

1. Pick a benchmark suite with a leaderboard that’s been updated recently. please do not use the example, use some benchmark can be easy to train, train time should be limited to 3 min, so that other can quick see how it perform on their end
    1. Examples: 
        1. Text-to-SQL: https://bird-bench.github.io/
        2. Cybersecurity: https://github.com/jpmorganchase/CyberBench
    2. Each person needs to pick a different benchmark suite. First come first serve: put your choice in Telegram channel to claim it. Just give the link to the leaderboard.
2. If you need to significantly adapt the MaAS codebase or the benchmark suite in a specific way to get results, you’re welcome to do that. Just make note of it in your submission.
3. Find the 5 **easiest** examples in the benchmark suite in which MaAS **fails** to produce the correct answer. Do a deep dive for each example to understand why it fails.
    1. What is the root cause of the failure?
    2. Is the MaAS implementation missing an operator?
    3. Or, does it have all the operators needed for this example, but the search fails to find it?
4. Find the 5 **hardest** examples in the benchmark suite in which MaAS succeeds in producing the correct answer. Draw out the multi-agent system it finds for each example and show it side by side with the example itself.