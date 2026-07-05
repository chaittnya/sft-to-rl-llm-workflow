# GRPO_ORM

## Task

**GSM8K math word problems** — given a grade-school arithmetic problem, generate a
step-by-step solution and arrive at the correct numerical answer.

This directory switches away from the Alpaca instruction-following task used in `GRPO/`
because Outcome Reward Models need a verifiable ground-truth outcome. Math problems have
one — the final number — which makes it possible to label a solution as objectively
correct or incorrect without human preference judgements.

---

## Method: GRPO (Group Relative Policy Optimisation)

GRPO is a reinforcement learning algorithm for language models. For each prompt it
generates a small group of completions (4 here), scores all of them with the reward
function, and uses the *relative* scores within the group as the advantage signal. The
completion with the highest score in the group gets a positive update; the rest get
negative or zero updates proportional to how far they fell below the group average.

The key property that makes GRPO attractive here: it does not need a separate value model
(critic). Standard PPO needs a critic to estimate a baseline reward, which means training
and loading a fourth model. GRPO sidesteps this by computing the baseline from the group
of completions instead.

---

## Reward Model: Outcome Reward Model (ORM)

An ORM scores a **complete solution** in isolation and answers the binary question: is
this answer correct? It does not look at individual steps.

**Architecture:** `AutoModelForSequenceClassification` with `num_labels=1` and
`problem_type="regression"`. A single linear head on top of the transformer outputs one
scalar per input. MSELoss is used with binary 0.0/1.0 targets.

**How it differs from the Bradley-Terry model in `GRPO/`:**
- Bradley-Terry is trained on *pairs* — it learns "A is better than B" by comparing two
  responses simultaneously.
- ORM is trained on *individual responses* — it learns "this is correct" or "this is
  wrong" with no comparison needed.

**Training data:** For each GSM8K example, two rows are created — the reference solution
labelled 1.0, and the next example's solution labelled 0.0. Using a real solution from a
different problem as the "wrong" answer keeps the text fluent and forces the ORM to check
the math rather than detecting badly-formed text.

---

## Training Pipeline

Run the following scripts in order from inside the `GRPO_ORM/` directory:

```
python create_data.py            # → grpo_items.jsonl  (200 math prompts for GRPO)
python create_orm_data.py        # → orm_data.jsonl    (400 correct/wrong solution pairs)
python train_orm.py              # → orm_model/        (trained ORM checkpoint)
python grpo_orm.py               # → grpo_orm_model/   (GRPO-trained policy)
```

**Dataset:** `openai/gsm8k` (config: `main`), first 200 examples of the training split.

**Base model:** `../SFT/final_model` — the SFT checkpoint must exist before running any
script here.

---

## Inference Pipeline

**Using the trained policy to answer a math question:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_orm_model")
tokenizer = AutoTokenizer.from_pretrained("./grpo_orm_model")

prompt = "Q: There are 5 apples and 3 oranges. How many fruits are there in total?\nA: Let me solve this step by step.\nStep 1:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Using the ORM to score a candidate solution:**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

orm = AutoModelForSequenceClassification.from_pretrained("./orm_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./orm_model")

text = "Q: There are 5 apples and 3 oranges. How many fruits total?\nA: 5 + 3 = 8. #### 8"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    score = orm(**inputs).logits.squeeze(-1).item()
print(f"ORM score: {score:.3f}")  # closer to 1.0 = more likely correct
```
