# GRPO_KWISE — GRPO with Plackett-Luce (K-wise) Reward Model

## Task

**Alpaca-style instruction following** using the `yahma/alpaca-cleaned` dataset. Same
task as `GRPO/` — the only difference is the reward model training objective.

---

## Method: GRPO (Group Relative Policy Optimisation)

Same algorithm as `GRPO/`. See that directory's README for a full explanation of GRPO.

The interesting thing in this directory is not the RL algorithm but the reward model
objective. Both `GRPO/` and `GRPO_KWISE/` use GRPO, so any difference in the trained
policy comes from the reward model they were trained with.

---

## Reward Model: Plackett-Luce K-wise Ranking

`GRPO/` uses a **Bradley-Terry** (pairwise) reward model: it compares two responses at
a time and learns which one is better. `GRPO_KWISE/` uses a **Plackett-Luce** (listwise)
reward model: it ranks **k=4 responses simultaneously** and learns the full ordering.

**Why Plackett-Luce over Bradley-Terry?**

The Bradley-Terry model sees one bit of information per training example: "A is better
than B". Plackett-Luce sees one full ranking per training example: "response 1 > response
2 > response 3 > response 4". That is a richer signal — the model learns not just which
response is best but also the relative ordering among all four candidates.

**How the Plackett-Luce loss works:**

Given k responses ordered best to worst with scores s_0 > s_1 > ... > s_{k-1}, the
loss encourages this ranking by computing the probability of observing it:

```
P(ranking) = ∏_{i=0}^{k-1} exp(s_i) / Σ_{j=i}^{k-1} exp(s_j)
```

At each position i, the model asks: "given the responses not yet ranked, how likely is
response i to be the best one?" Summing the log-probabilities and negating gives the loss.
A custom `PlackettLuceTrainer` in `train_kwise_reward_model.py` implements this.

**Training data** (from `create_kwise_reward_data.py`): 4 completions per prompt, ordered
best to worst:
- Rank 0: complete reference answer
- Rank 1: truncated reference answer (first half of words)
- Rank 2: vague generic filler
- Rank 3: outright refusal

At inference time the model is used identically to the Bradley-Terry reward model — call
it on `(prompt + completion)` and get a scalar back.

---

## Training Pipeline

```
python create_data.py                # → grpo_items.jsonl         (prompts for GRPO)
python create_kwise_reward_data.py   # → kwise_reward_data.jsonl  (4-way ranked completions)
python train_kwise_reward_model.py   # → kwise_reward_model/      (Plackett-Luce RM)
python grpo_kwise.py                 # → grpo_kwise_model/        (GRPO-trained policy)
```

**Dataset:** `yahma/alpaca-cleaned`, first 200 examples.

**Base model:** `../SFT/final_model`.

---

## Inference Pipeline

**Using the trained policy:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_kwise_model")
tokenizer = AutoTokenizer.from_pretrained("./grpo_kwise_model")

prompt = "### Instruction:\nWhat is the capital of France?\n\n### Input:\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Scoring a response with the Plackett-Luce reward model:**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

rm = AutoModelForSequenceClassification.from_pretrained("./kwise_reward_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./kwise_reward_model")

responses = [
    "### Instruction:\nWhat is 2+2?\n\n### Response: The answer is 4.",
    "### Instruction:\nWhat is 2+2?\n\n### Response: I don't know.",
]
for text in responses:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        score = rm(**inputs).logits.squeeze(-1).item()
    print(f"Score: {score:.3f}  |  {text[50:80]}...")
```

The Plackett-Luce model should assign higher scores to better responses because it was
trained to rank 4 responses at a time, which provides more gradient information per
training example than the pairwise Bradley-Terry model in `GRPO/`.
