# PPO_KWISE — PPO with Plackett-Luce K-wise Reward Model

## Task

**Alpaca-style instruction following** using the `yahma/alpaca-cleaned` dataset. Same
task as `PPO/` and `GRPO_KWISE/`, which makes this directory directly comparable to both:

- vs `PPO/`: same algorithm (PPO), same task — the only variable is the reward model
  training objective (pairwise Bradley-Terry vs. k-wise Plackett-Luce).
- vs `GRPO_KWISE/`: same reward model — the only variable is the RL algorithm (GRPO vs.
  PPO, which also adds a value model).

---

## Method: PPO (Proximal Policy Optimisation)

Same four-model PPO structure as `PPO/`. See `PPO/README.md` for a full explanation of
the policy, reference, reward, and value model roles.

The short version: PPO generates completions, scores them with the frozen reward model,
and uses `actual_reward - value_estimate` (the advantage) to update the policy. A KL
penalty against the reference model prevents the policy from exploiting the reward model.

---

## Reward Model: Plackett-Luce K-wise Ranking

`PPO/` trains its reward model with **Bradley-Terry**: compare two responses, learn
which is better. `PPO_KWISE/` trains its reward model with **Plackett-Luce**: rank k=4
responses simultaneously, learn the full ordering.

**Why more ranking signal matters:**

A Bradley-Terry training example carries roughly 1 bit of information ("A > B"). A
Plackett-Luce example with k=4 carries log_2(4!) ≈ 4.6 bits ("rank 0 > rank 1 > rank 2
> rank 3"). The model sees the gradient from all four pairwise comparisons within a
single example, which makes the reward model learn faster and with less data.

**The Plackett-Luce loss:**

Given k completions ordered best→worst with scores s_0 > s_1 > ... > s_{k-1}:

```
L = -sum_{i=0}^{k-1} [ s_i - logsumexp(s_i, s_{i+1}, ..., s_{k-1}) ]
```

At each position i, item i is the "winner" of the remaining pool and must score higher
than all items not yet removed. `logsumexp` is used instead of `log(sum(exp(...)))` for
numerical stability.

**Inference — no limitation vs PPO:**

This is where PPO_KWISE differs from PPO_PRM. In PPO_PRM, the PRM needs to score steps
individually (step-splitting at inference time), but PPOTrainer only accepts a torch.nn.Module,
making that impossible. Plackett-Luce has no such limitation: at inference time the model
just takes a single (prompt + completion) text and returns one scalar — exactly the same
interface as Bradley-Terry. The k-wise structure only exists during offline reward model
training.

**Value model:** Warm-started from the k-wise reward model checkpoint in
`train_value_model.py`. The PL model already learned useful quality representations
from ranking supervision; fine-tuning those weights for regression calibrates the scoring
range before PPO starts.

---

## Training Pipeline

```
python create_data.py                # → ppo_prompts.jsonl       (200 Alpaca prompts)
python create_kwise_reward_data.py   # → kwise_reward_data.jsonl (4 ranked completions per prompt)
python train_kwise_reward_model.py   # → kwise_reward_model/     (Plackett-Luce RM)
python create_value_data.py          # → value_data.jsonl        (binary regression data)
python train_value_model.py          # → value_model/            (critic from PL checkpoint)
python ppo_kwise.py                  # → ppo_kwise_model/        (PPO-trained policy)
```

**Dataset:** `yahma/alpaca-cleaned`, first 200 examples.

**Base model:** `../SFT/final_model`. Run `SFT/lora.py` first.

---

## Inference Pipeline

**Using the trained policy:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./ppo_kwise_model")
tokenizer = AutoTokenizer.from_pretrained("./ppo_kwise_model")

prompt = "### Instruction:\nExplain the difference between supervised and unsupervised learning.\n\n### Input:\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Scoring a response with the k-wise reward model:**

The model's inference call is identical to Bradley-Terry — single text in, scalar out.
The k-wise training only changes how the weights were learned.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

rm = AutoModelForSequenceClassification.from_pretrained("./kwise_reward_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./kwise_reward_model")

responses = [
    "### Instruction:\nWhat is 2+2?\n\n### Response: The answer is 4. Addition of 2 and 2 gives 4.",
    "### Instruction:\nWhat is 2+2?\n\n### Response: I don't know, figure it out yourself.",
]
for text in responses:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        score = rm(**inputs).logits.squeeze(-1).item()
    print(f"{score:.3f}  {text[50:90]}...")
```

The k-wise model should assign clearly higher scores to the helpful response. Because it
was trained on 4 ranked candidates at once (not just one pair), it saw more contrastive
signal per example and should produce sharper score separation than the pairwise model
in `PPO/`.
