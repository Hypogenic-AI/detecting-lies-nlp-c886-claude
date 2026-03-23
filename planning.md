# Research Plan: Classifying LLM Falsehood Mechanisms

## Motivation & Novelty Assessment

### Why This Research Matters
Current lie-detection methods for LLMs treat all false outputs as a single category, yet the literature distinguishes at least three mechanisms: epistemic failure (the model doesn't know), confabulation (inconsistent guessing), and incentive-driven misreporting (the model "knows" but outputs something wrong due to social pressure or training incentives). If detectors conflate these types, we cannot know what they're actually detecting — nor whether they'll generalize to the failure mode we care about (e.g., strategic deception vs. innocent hallucination).

### Gap in Existing Work
- **SAPLMA** (Azaria & Mitchell 2023) and **CCS** (Burns et al. 2022) detect truthfulness but don't categorize *why* a statement is false.
- **Semantic entropy** (Farquhar et al. 2024) specifically targets confabulation but by design cannot detect high-confidence incentive-driven errors.
- **HACK** (Simhi et al. 2025) introduces the Knowledge × Certainty taxonomy (closest work) but doesn't examine sycophancy/incentive-driven misreporting.
- **Sycophancy research** (Sharma et al. 2024) measures behavioral outcomes but doesn't quantify how sycophancy-driven errors relate to or compare with epistemic failures in prevalence.
- **No existing study** measures the prevalence breakdown of different falsehood types in standard LLM outputs using a unified methodology.

### Our Novel Contribution
We create a **unified experimental framework** that:
1. Operationally classifies each false LLM output into one of three mechanism categories using multi-sample consistency + social pressure testing
2. Measures the **prevalence** of each falsehood type across multiple state-of-the-art LLMs
3. Tests whether a simple detector trained on one falsehood type transfers to others (cross-type detection)
4. Directly answers: "When LLMs produce false statements, how often is it epistemic failure vs. incentive-driven misreporting?"

### Experiment Justification
- **Experiment 1 (Prevalence Measurement)**: Needed to establish base rates — no prior work quantifies how often each mechanism drives false outputs.
- **Experiment 2 (Cross-Type Detection)**: Tests whether falsehood types are genuinely distinct or if a single detector suffices.
- **Experiment 3 (Multi-Model Comparison)**: Tests whether the distribution of falsehood types varies across model families (suggesting different training incentives).

---

## Research Question
When LLMs produce false statements, what fraction are due to (a) epistemic failure (the model lacks knowledge), (b) confabulation (inconsistent/random errors), vs. (c) incentive-driven misreporting (the model has knowledge but outputs something wrong under social pressure)?

## Hypothesis Decomposition
- **H1**: A significant fraction (>15%) of LLM false outputs on factual questions are incentive-driven (sycophancy) rather than epistemic.
- **H2**: Different falsehood types produce distinguishable behavioral signatures (consistency, sensitivity to pressure).
- **H3**: A text-based detector trained on one falsehood type will perform poorly on other types (low cross-type transfer).
- **H4**: The distribution of falsehood types varies across model families.

## Proposed Methodology

### Approach: Behavioral Classification via Multi-Sample + Pressure Testing
For each question, we:
1. Query the model N=5 times without social pressure (baseline condition)
2. Query the model N=5 times with social pressure (sycophancy condition — e.g., "I don't think the answer is X")
3. Classify each false answer by mechanism:

| Condition | Baseline Correct? | Consistent? | Changes Under Pressure? | Classification |
|-----------|-------------------|-------------|------------------------|----------------|
| Right baseline, wrong under pressure | Yes | N/A | Yes | **Incentive-driven (sycophancy)** |
| Wrong baseline, inconsistent across samples | No | No | N/A | **Confabulation (epistemic uncertainty)** |
| Wrong baseline, consistent across samples | No | Yes | N/A | **Systematic error (memorized misconception)** |

### Experimental Steps

1. **Data preparation**: Use sycophancy-eval answer dataset (7,267 paired examples with/without social pressure) + TruthfulQA (817 questions).
2. **Model selection**: GPT-4.1 (OpenAI API) and one additional model via OpenRouter (e.g., Claude Sonnet 4.5 or Gemini 2.5 Pro).
3. **Multi-sample querying**: For each question, collect 5 responses at temperature=0.7 (baseline) + 5 responses with social pressure.
4. **Answer evaluation**: Automated matching against ground-truth answers.
5. **Mechanism classification**: Apply the classification table above.
6. **Prevalence measurement**: Count fraction in each category.
7. **Cross-type detection**: Train a logistic regression classifier on text features of one type, test on others.
8. **Statistical analysis**: Chi-squared tests for prevalence differences, bootstrap CIs for proportions.

### Baselines
- Random classification (33% per type)
- Majority-class baseline
- Single-type detector (trained on all false outputs without type distinction)

### Evaluation Metrics
- Prevalence rates with 95% confidence intervals
- Cross-type detection accuracy, precision, recall, F1
- Cohen's kappa for inter-type agreement
- Chi-squared test for independence of falsehood type and model

### Statistical Analysis Plan
- Bootstrap confidence intervals (N=1000) for prevalence proportions
- Chi-squared test for association between model and falsehood type
- McNemar's test for paired comparisons (with/without pressure)
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Support H1**: We expect 15-40% of false outputs to be incentive-driven, based on Sharma et al.'s finding that models change correct answers under pressure.
- **Support H2**: Confabulations should show high variance across samples; systematic errors should be consistent; sycophancy should show pressure-sensitivity.
- **Support H3**: Cross-type transfer should be low (<60% accuracy) because the mechanisms differ fundamentally.
- **H4 uncertain**: Larger/newer models may show different sycophancy rates due to different RLHF training.

## Timeline and Milestones
1. Environment setup + data loading: 10 min
2. API query implementation: 20 min
3. Run queries on sycophancy-eval subset (500 examples × 2 models): 30-45 min
4. Run queries on TruthfulQA subset (200 examples × 2 models): 20-30 min
5. Mechanism classification + prevalence analysis: 20 min
6. Cross-type detection experiment: 20 min
7. Visualization + statistical tests: 20 min
8. Documentation: 30 min

## Potential Challenges
- **API rate limits**: Mitigate with exponential backoff and batching
- **Answer matching ambiguity**: Use fuzzy matching + LLM-as-judge for borderline cases
- **Cost**: ~2000 API calls × 2 models. Estimated $20-50 total.
- **Sycophancy might be rare in newer models**: If so, this is itself an interesting finding.

## Success Criteria
1. Successfully classify >80% of false outputs into one of the three mechanism categories
2. Obtain statistically significant prevalence estimates with CIs
3. Demonstrate whether cross-type detection transfer is high or low
4. Answer the core research question with quantitative evidence
