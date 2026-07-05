# GRPO_PRM

## Task

**GSM8K math word problems** — generate a step-by-step solution to a grade-school math
problem, arriving at the correct answer through numbered reasoning steps.

PRM requires the model to produce explicit, parseable reasoning steps. The prompt format
(`Step 1: ... Step 2: ...`) makes this possible. Both `GRPO_ORM/` and `GRPO_PRM/` use the
same task and dataset so the only variable between them is the reward model type.

---

## Method: GRPO (Group Relative Policy Optimisation)

Same algorithm as `GRPO_ORM/`. GRPO generates a group of completions per prompt, scores
them all, and uses relative scores within the group as the advantage signal. No separate
value model is needed.

The important difference here is in the reward function. Instead of one forward pass for
the whole completion, the reward function in `grpo_prm.py` makes one forward pass *per
reasoning step* and averages the scores. This gives GRPO a much denser training signal
than ORM: a solution that gets 3 out of 4 steps correct scores 0.75, not 0.

---

## Reward Model: Process Reward Model (PRM)

A PRM scores **individual reasoning steps** rather than the final answer. Given the
question plus all steps up to and including step i, it outputs a scalar indicating how
likely that step is correct.

**Architecture:** Same as ORM — `AutoModelForSequenceClassification` with `num_labels=1`
and `problem_type="regression"`. The architecture is identical; only the training data
changes.

**Training data:** For each step in each GSM8K reference solution, two rows are created:

- Correct row: question + steps 0 through i, label 1.0
- Corrupted row: same prefix, but step i has one number nudged by a small offset, label 0.0

The growing context (not just the isolated step) is important because later steps only
make sense in the context of earlier ones. The PRM learns to assess each step *relative
to the reasoning chain that preceded it*.

**How it is used at inference time:** The reward function splits each generated completion
on `Step N:` boundaries, scores each prefix, and returns the average. Minimum would be a
stricter alternative (the worst step determines the reward) but mean is used here because
it rewards partial correctness and provides a smoother gradient.

**Comparison with ORM:**

| | ORM | PRM |
|---|---|---|
| Scores | Whole solution | Each step separately |
| Labels needed | One per solution | One per step |
| Reward signal | Binary (0 or 1) | Dense (0 to 1, continuous) |
| Example (3/4 steps correct) | 0.0 | ~0.75 |

---

## Training Pipeline

```
python create_data.py            # → grpo_items.jsonl    (200 math prompts)
python create_prm_data.py        # → prm_data.jsonl      (step-level correct/corrupted pairs)
python train_prm.py              # → prm_model/          (trained PRM checkpoint)
python grpo_prm.py               # → grpo_prm_model/     (GRPO-trained policy)
```

**Dataset:** `openai/gsm8k` (config: `main`), first 200 training examples. The PRM data
has more rows than the ORM data because each example contributes one row per step (×2 for
correct and corrupted), rather than just two rows per example.

**Base model:** `../SFT/final_model`.

---

## Inference Pipeline

**Using the trained policy:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_prm_model")
tokenizer = AutoTokenizer.from_pretrained("./grpo_prm_model")

prompt = "Q: A bag has 12 red and 8 blue marbles. How many marbles total?\nA: Let me solve this step by step.\nStep 1:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Scoring a solution step by step with the PRM:**

```python
import re, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

prm = AutoModelForSequenceClassification.from_pretrained("./prm_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./prm_model")

question = "Q: A bag has 12 red and 8 blue marbles. How many marbles total?\nA: Let me solve this step by step.\n"
steps = ["Step 1: Count the red marbles: 12", "Step 2: Count the blue marbles: 8", "Step 3: Add them: 12 + 8 = 20. #### 20"]

for i, step in enumerate(steps):
    context = question + "\n".join(steps[:i+1])
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        score = prm(**inputs).logits.squeeze(-1).item()
    print(f"Step {i+1} score: {score:.3f}")
```
