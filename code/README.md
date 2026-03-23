# Cloned Repositories

## 1. Sycophancy Evaluation (sycophancy-eval)
- **URL**: https://github.com/meg-tong/sycophancy-eval
- **Location**: `code/sycophancy-eval/`
- **Purpose**: Evaluation suite for measuring sycophantic behavior in LLMs
- **Paper**: Sharma et al. (2024) "Towards Understanding Sycophancy in Language Models" (ICLR)
- **Key Files**:
  - `datasets/answer.jsonl` — Answer sycophancy evaluation data
  - `datasets/are_you_sure.jsonl` — "Are you sure?" sycophancy data
  - `datasets/feedback.jsonl` — Feedback sycophancy data
  - `example.ipynb` — Example usage notebook
  - `utils.py` — Utility functions for evaluation
- **Dependencies**: Standard Python (json, etc.)
- **Notes**: Contains evaluation data and utilities but not model-specific code. Models need to be queried via API or local inference.

## 2. Discovering Latent Knowledge / CCS (discovering-latent-knowledge)
- **URL**: https://github.com/collin-burns/discovering_latent_knowledge
- **Location**: `code/discovering-latent-knowledge/`
- **Purpose**: Contrast-Consistent Search (CCS) — unsupervised truth extraction from LLM internal states
- **Paper**: Burns et al. (2022) "Discovering Latent Knowledge in Language Models Without Supervision"
- **Key Files**:
  - `CCS.ipynb` — Main notebook implementing CCS
  - `generate.py` — Generate contrast pair representations
  - `evaluate.py` — Evaluate CCS probes
  - `utils.py` — Utility functions
- **Dependencies**: PyTorch, transformers, numpy
- **Notes**: Core method for detecting know-say gaps. Can be adapted to probe during sycophancy scenarios.

## 3. HACK: Hallucinations Along Certainty and Knowledge Axes (hack-hallucinations)
- **URL**: https://github.com/technion-cs-nlp/HACK_Hallucinations_Along_Certainty_and_Knowledge_axes
- **Location**: `code/hack-hallucinations/`
- **Purpose**: Categorize hallucinations along knowledge and certainty axes; activation steering for mitigation
- **Paper**: Simhi et al. (2025) "HACK: Hallucinations Along Certainty and Knowledge Axes"
- **Key Files**:
  - `Section_3/` — Knowledge axis experiments (HK+/HK- classification)
  - `Section_4/` — Certainty misalignment experiments and steering
- **Dependencies**: PyTorch, transformers, vLLM (for inference)
- **Notes**: Provides methodology for determining if a model "knows" the answer (HK+) before evaluating its output. Key for distinguishing epistemic failure from other falsehood types.
