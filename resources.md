# Resources Catalog

## Summary
Resources gathered for investigating how LLM-generated false statements break down into epistemic failure versus incentive-driven misreporting. The collection spans three threads: (1) internal-state probing for truthfulness, (2) hallucination taxonomy and detection, (3) sycophancy and strategic deception.

## Papers
Total papers downloaded: 22

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Internal State Knows When Lying | Azaria, Mitchell | 2023 | papers/2304.13734_internal_state_lying.pdf | SAPLMA: probes on hidden states, 71-83% acc |
| Semantic Entropy for Hallucinations | Farquhar et al. | 2024 | papers/2302.09611_semantic_entropy.pdf | Nature paper, confabulation detection |
| TruthfulQA | Lin et al. | 2021 | papers/2109.07958_truthfulqa.pdf | 817-question truthfulness benchmark |
| Deception Abilities in LLMs | Hagendorff | 2023 | papers/2307.16513_deception_abilities_llm.pdf | LLMs can induce strategic deception |
| Hallucination Survey | Huang et al. | 2023 | papers/2311.05232_hallucination_survey.pdf | Comprehensive hallucination taxonomy |
| HACK: Certainty × Knowledge | Simhi et al. | 2025 | papers/2510.24222_hack_hallucination_axes.pdf | 2×2 hallucination taxonomy, CM detection |
| HAD: Hallucination Detection | Xu et al. | 2025 | papers/2502.04427_had_hallucination_detection.pdf | 11-category taxonomy, HAD models |
| LLM Lie Detection (Diplomacy) | Banerjee et al. | 2024 | papers/2408.02431_llm_lie_detection_diplomacy.pdf | Self-generated feedback for lie detection |
| Confabulation Value | Sui et al. | 2024 | papers/2406.04175_confabulation_value.pdf | Confabulations show increased narrativity |
| LLM Misinformation Detection | Chen, Shu | 2023 | papers/2309.13788_llm_misinformation_detected.pdf | LLM misinformation harder to detect |
| LLMs Know When Hallucinate | Duan et al. | 2024 | papers/2402.02181_llm_know_hallucinate.pdf | Hidden states differ for true vs fabricated |
| MIND Hallucination Detection | Su et al. | 2024 | papers/2406.09569_mind_hallucination.pdf | Unsupervised real-time detection |
| HaluEval-Wild | Zhu et al. | 2024 | papers/2403.04307_halueval_wild.pdf | Real-world hallucination benchmark |
| Chain-of-Verification | Dhuliawala et al. | 2023 | papers/2310.18344_chain_of_verification.pdf | Self-verification reduces hallucinations |
| Understanding Sycophancy | Sharma et al. | 2024 | papers/2310.13548_sycophancy_understanding.pdf | Sycophancy from RLHF, ICLR 2024 |
| Simple Probes for LLMs | — | 2023 | papers/2308.14752_simple_probes_llm.pdf | Probing internal representations |
| Sleeper Agents | Hubinger et al. | 2024 | papers/2312.04963_sleeper_agents.pdf | Deceptive LLMs persist through safety training |
| HaluEval Benchmark | Li et al. | 2023 | papers/2305.11747_halueval.pdf | 35K sample hallucination benchmark |
| Representation Engineering | Li et al. | 2023 | papers/2310.01405_representation_engineering.pdf | RepE framework for truthfulness control |
| LLM Self-Knowledge | — | 2023 | papers/2309.15840_llm_knows_true.pdf | Do LLMs know what they know? |
| Discovering Latent Knowledge (CCS) | Burns et al. | 2022 | papers/2212.03827_discovering_latent_knowledge.pdf | Unsupervised truth extraction from internals |
| Geometry of Truth | Marks, Tegmark | 2023 | papers/2310.06824_geometry_of_truth.pdf | Linear truth directions in representation space |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA | HuggingFace | 817 questions | Truthfulness QA | datasets/truthfulqa/ | Key benchmark, 38 categories |
| HaluEval (QA) | HuggingFace | 10K samples | Hallucination detection | datasets/halueval/ | QA hallucination pairs |
| Sycophancy-eval | GitHub | 20,654 examples | Sycophancy measurement | datasets/sycophancy-eval/ | 3 sycophancy task types |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| sycophancy-eval | github.com/meg-tong/sycophancy-eval | Sycophancy evaluation suite | code/sycophancy-eval/ | Eval data + utilities |
| discovering-latent-knowledge | github.com/collin-burns/discovering_latent_knowledge | CCS unsupervised truth probing | code/discovering-latent-knowledge/ | Core method for know-say gap |
| HACK | github.com/technion-cs-nlp/HACK_... | Knowledge×Certainty hallucination taxonomy | code/hack-hallucinations/ | HK+/HK- classification + steering |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder with three queries: (1) LLM lie detection hallucination deception, (2) hallucination confabulation classification taxonomy, (3) sycophancy deceptive alignment probing
- Supplemented with known key papers from the field (CCS, Geometry of Truth, RepE, Sleeper Agents)
- Dataset search covered HuggingFace, Papers with Code, and paper-referenced datasets

### Selection Criteria
- Prioritized papers that distinguish between falsehood types or provide tools for doing so
- Focused on methods that access internal representations (probes, CCS, steering)
- Included both detection methods and behavioral evaluation frameworks
- Selected datasets that span both epistemic failure (hallucination) and incentive-driven (sycophancy) scenarios

### Challenges Encountered
- Azaria true-false dataset download failed (site unreachable)
- Paper-finder third search timed out; supplemented with known papers
- HACK paper had wrong arXiv ID in search results; corrected to 2510.24222

### Gaps and Workarounds
- No single dataset cleanly labels falsehood type (epistemic vs strategic) — this is exactly the gap the research aims to address
- Azaria dataset unavailable; can reconstruct similar data using the methodology described in the paper
- Large datasets (TriviaQA, Natural Questions) not pre-downloaded; can be loaded on-demand via HuggingFace

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **TruthfulQA** as baseline factuality benchmark
- **Sycophancy-eval** for incentive-driven misreporting scenarios
- **TriviaQA** (load on-demand) for HACK-style HK+/HK- analysis
- Construct a combined evaluation set that labels each false output by mechanism

### 2. Baseline Methods
- **CCS** (Burns et al.) — unsupervised probing for "what model believes"
- **SAPLMA** (Azaria & Mitchell) — supervised probing for truthfulness
- **Semantic entropy** (Farquhar et al.) — confabulation detection
- **Behavioral sycophancy rate** — % correct answers changed under pressure

### 3. Evaluation Metrics
- Classification accuracy on epistemic-failure vs incentive-driven subsets
- Probe accuracy differential between HK+ and HK- examples
- CM-Score (HACK) for certainty-misaligned cases
- Sycophancy rate and its correlation with internal-state probe predictions

### 4. Code to Adapt/Reuse
- **CCS** (code/discovering-latent-knowledge/) — adapt for probing during sycophancy scenarios
- **HACK** (code/hack-hallucinations/) — use HK+/HK- methodology to classify knowledge status
- **Sycophancy-eval** (code/sycophancy-eval/) — evaluation framework for incentive-driven scenarios

### 5. Proposed Experimental Pipeline
1. Select open-weight LLMs (LLaMA-2-7B, Mistral-7B) for internal state access
2. Run TruthfulQA and sycophancy-eval, collecting hidden states at each layer
3. Apply HACK methodology to classify HK+ (has knowledge) vs HK- (lacks knowledge)
4. Train CCS/linear probes on internal states
5. Compare probe predictions with actual outputs to detect know-say gaps
6. Measure prevalence of each falsehood type: (a) HK- errors = epistemic failure, (b) HK+ errors without social pressure = confabulation/retrieval failure, (c) HK+ errors with social pressure = incentive-driven misreporting
7. Evaluate whether different falsehood types produce distinguishable internal representations
