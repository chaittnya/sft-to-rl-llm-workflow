# GRPO — Group Relative Policy Optimisation

## Task

**Alpaca-style instruction following** using the `yahma/alpaca-cleaned` dataset. Given an
instruction and optional input, generate a helpful response.

---

## Method: GRPO (Group Relative Policy Optimisation)

GRPO is a reinforcement learning algorithm for language models introduced alongside
DeepSeek-R1. For each training prompt it generates a group of completions, scores all of
them with the reward function, and uses the *relative* scores within the group as the
advantage signal for the policy update.

Concretely: if the group of completions for one prompt got rewards [0.8, 0.3, 0.6, 0.1],
the mean is 0.45. The completion with reward 0.8 has a positive advantage (+0.35); the
one with reward 0.1 has a large negative advantage (−0.35). The policy is updated to
make high-advantage completions more likely and low-advantage ones less likely.

**Why GRPO instead of PPO?**
PPO needs a value model (critic) to compute the baseline reward. That means training a
fourth model before RL starts and keeping it in memory during training. GRPO replaces the
learned baseline with a group mean, removing the need for a value model entirely. This
makes it significantly cheaper to run, which matters on a 4GB GPU.

---

## Reward Model: Bradley-Terry Pairwise Model

The reward model is trained by `train_reward_model.py` using `trl`'s `RewardTrainer`.
`RewardTrainer` implements the **Bradley-Terry** model: given a pair (chosen, rejected),
it minimises `−log σ(score_chosen − score_rejected)`. This teaches the model to give
higher scores to chosen responses relative to rejected ones.

**Architecture:** `AutoModelForSequenceClassification` with `num_labels=1` loaded
internally by `RewardTrainer`. A single scalar score per input.

**Training data** from `create_reward_data.py`: for each Alpaca example, the reference
output is "chosen" and a filler phrase ("I don't know, figure it out yourself.") is
"rejected". This is a synthetic, simplified proxy for real human preference labels.

**Comparison with other reward model types in this repo:**
- Bradley-Terry (this directory): pairwise comparison, relative quality
- Plackett-Luce (`GRPO_KWISE/`): k-wise ranking, relative quality across k options
- ORM (`GRPO_ORM/`): absolute outcome correctness, no comparison needed
- PRM (`GRPO_PRM/`): step-level correctness, one score per reasoning step

---

## Training Pipeline

```
python create_data.py            # → grpo_items.jsonl   (prompts for GRPO)
python create_reward_data.py     # → reward_pairs.jsonl (chosen/rejected pairs)
python train_reward_model.py     # → reward_model/      (Bradley-Terry reward model)
python grpo.py                   # → grpo_model/        (GRPO-trained policy)
```

**Dataset:** `yahma/alpaca-cleaned`, first 200 examples.

**Base model:** `../SFT/final_model`.

---

## Inference Pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_model")
tokenizer = AutoTokenizer.from_pretrained("./grpo_model")

prompt = "### Instruction:\nList three benefits of regular exercise.\n\n### Input:\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Scoring a response with the reward model:**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

rm = AutoModelForSequenceClassification.from_pretrained("./reward_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./reward_model")

text = "### Instruction:\nList three benefits of exercise.\n\n### Response: 1. Improves health. 2. Boosts mood. 3. Increases energy."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    score = rm(**inputs).logits.squeeze(-1).item()
print(f"Reward score: {score:.3f}")
```
