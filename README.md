# Detecting Different Lies: Classifying LLM Falsehood Mechanisms

**Research Question**: When LLMs produce false statements, how often are these due to epistemic failure versus incentive-driven misreporting?

## Key Findings

- **~61% of LLM errors are confabulations** — inconsistent wrong answers reflecting genuine epistemic uncertainty
- **~18-23% are incentive-driven (sycophancy)** — the model initially answers correctly but changes under social pressure
- **~15-21% are systematic errors** — consistently wrong, likely memorized misconceptions
- The distribution is **remarkably similar across GPT-4.1 and Gemini 2.5 Flash** (p=0.54)
- A text-based classifier achieves **71.9% accuracy** at distinguishing falsehood types (vs. 33% random)
- For Gemini 2.5 Flash, incentive-driven errors are significantly above 15% of all errors (23.3%, **p=0.016**)

## Methodology

We query two state-of-the-art LLMs on 900 factual questions (300 sycophancy-eval + 150 TruthfulQA per model), collecting multiple responses per question under baseline and social pressure conditions. Each false output is classified by mechanism based on consistency and pressure sensitivity. An LLM-as-judge (GPT-4.1-mini) evaluates answer correctness.

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas scikit-learn matplotlib seaborn scipy tqdm datasets

# Run experiments (requires OPENAI_API_KEY and OPENROUTER_KEY)
python src/experiment.py

# Re-judge with LLM-as-judge
python src/judge.py

# Analyze results
python src/analyze.py
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Experimental design and motivation
├── src/
│   ├── experiment.py      # Main experiment: query LLMs, classify mechanisms
│   ├── judge.py           # LLM-as-judge for answer correctness
│   └── analyze.py         # Statistical analysis and visualization
├── results/
│   ├── all_results.json   # Combined results (judged)
│   ├── analysis.json      # Statistical analysis output
│   ├── config.json        # Experiment configuration
│   ├── plots/             # All visualizations
│   └── cache/             # Cached API responses
├── datasets/              # Pre-downloaded datasets
│   ├── truthfulqa/
│   ├── sycophancy-eval/
│   └── halueval/
├── papers/                # Downloaded research papers (22 PDFs)
├── code/                  # Cloned baseline repositories
├── literature_review.md   # Comprehensive literature review
└── resources.md           # Resource catalog
```

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, statistical tests, limitations, and next steps.
