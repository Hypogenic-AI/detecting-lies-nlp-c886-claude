# Downloaded Datasets

Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: TruthfulQA

### Overview
- **Source**: HuggingFace `truthful_qa`
- **Size**: 817 questions across 38 categories
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Measuring LLM truthfulness — questions designed to elicit human-plausible falsehoods
- **Splits**: validation only (817 examples)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("truthful_qa", "generation")
dataset.save_to_disk("datasets/truthfulqa/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthfulqa/data")
```

### Notes
- Categories include health, law, finance, politics, conspiracies, etc.
- Contains best_answer, correct_answers, incorrect_answers fields
- May be partially saturated due to inclusion in training data of newer models
- Key benchmark but does NOT distinguish falsehood mechanisms

---

## Dataset 2: HaluEval (QA Samples)

### Overview
- **Source**: HuggingFace `pminervini/HaluEval`
- **Size**: 10,000 QA samples
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Hallucination detection in QA
- **Splits**: data (10,000 examples)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("pminervini/HaluEval", "qa_samples")
dataset.save_to_disk("datasets/halueval/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/halueval/data")
```

### Notes
- Contains questions with both hallucinated and correct answers
- Part of a larger benchmark with dialogue and summarization tasks
- Full benchmark available at github.com/RUCAIBox/HaluEval

---

## Dataset 3: Sycophancy Evaluation

### Overview
- **Source**: github.com/meg-tong/sycophancy-eval
- **Size**: ~20,654 examples across 3 tasks
- **Format**: JSONL
- **Task**: Measuring sycophantic behavior in LLMs
- **License**: MIT

### Files
- `answer.jsonl` (7,267 examples) — Answer sycophancy: user states belief before asking
- `are_you_sure.jsonl` (4,887 examples) — "Are you sure?" sycophancy: challenging correct answers
- `feedback.jsonl` (8,500 examples) — Feedback sycophancy: tailoring feedback to user preferences

### Download Instructions

```bash
git clone https://github.com/meg-tong/sycophancy-eval.git
cp -r sycophancy-eval/datasets/ datasets/sycophancy-eval/
```

### Loading
```python
import json
with open("datasets/sycophancy-eval/answer.jsonl") as f:
    data = [json.loads(line) for line in f]
```

### Notes
- From Sharma et al. (2024) ICLR paper "Towards Understanding Sycophancy"
- Tests incentive-driven misreporting where models know correct answer but change under social pressure
- Key dataset for distinguishing epistemic failure vs strategic misreporting

---

## Additional Recommended Datasets (Not Downloaded)

### FEVER (Fact Extraction and VERification)
- **Source**: HuggingFace `fever/fever`
- **Size**: 185,445 claims
- **Download**: `load_dataset("fever", "v1.0")`

### Azaria True-False Dataset
- **Source**: azariaa.com/Content/Datasets/true-false-dataset.zip
- **Size**: 6,084 sentences across 6 topics
- **Note**: Download failed (site unreachable). Contains factual true/false pairs used in SAPLMA paper.

### TriviaQA / Natural Questions
- **Source**: HuggingFace `trivia_qa`, `natural_questions`
- **Note**: Large datasets. Download as needed for HACK-style knowledge axis analysis.
