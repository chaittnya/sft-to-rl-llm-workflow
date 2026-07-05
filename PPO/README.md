# PPO — Proximal Policy Optimisation

## Task

**Alpaca-style instruction following** using the `yahma/alpaca-cleaned` dataset.

---

## Method: PPO (Proximal Policy Optimisation)

PPO is the standard deep RL algorithm used for RLHF (Reinforcement Learning from Human
Feedback). It updates the policy to maximise the reward signal while staying close to the
reference policy via a KL penalty. Four models are needed simultaneously:

**Policy model** — the LLM we are training. It generates responses during rollouts and
its weights are updated at every PPO step.

**Reference model** — a frozen copy of the initial policy. After each rollout, PPO
computes KL divergence between the policy and reference distributions and adds it as a
penalty to the loss. This is critical: without the KL penalty, PPO quickly exploits the
reward model by producing responses that score well but are meaningless — a failure mode
called "reward hacking".

**Reward model** — scores each completed response with a scalar. Frozen during PPO
training. In this directory it is a Bradley-Terry pairwise model trained on preference
data.

**Value model** (critic) — a separate network that estimates the expected future reward
from the current state. PPO uses `actual_reward − value_estimate` as the advantage for
each token. A well-calibrated value model reduces variance in the gradient estimate,
which stabilises training. The value model is updated every step alongside the policy.

**Memory note:** Four models on a 4GB GPU is tight. All models are loaded in `bfloat16`
(not `float16` — bfloat16 has float32's exponent range, which prevents NaN gradients from
raw unscaled updates). `local_rollout_forward_batch_size=2` limits how many completions
are generated in parallel during rollout to avoid OOM.

---

## Reward Model: Bradley-Terry Pairwise Model

Trained by `train_reward_model.py` using `trl`'s `RewardTrainer` (Bradley-Terry loss).
The same architecture as the value model (`AutoModelForSequenceClassification`,
`num_labels=1`), but trained on pairwise (chosen, rejected) preference data.

**Value model:** Warm-started from the reward model checkpoint. The value model and reward
model share the same architecture. Initialising the value model from the reward model
checkpoint (rather than from the raw SFT base) gives it a useful starting point: it
already understands which responses are better, and only needs to calibrate the numerical
scale of its predictions.

---

## Training Pipeline

```
python create_data.py            # → ppo_prompts.jsonl  (prompts, tokenised in ppo.py)
python create_reward_data.py     # → reward_pairs.jsonl (chosen/rejected for reward model)
python train_reward_model.py     # → reward_model/      (Bradley-Terry RM, seeds value model)
python create_value_data.py      # → value_data.jsonl   (regression targets for critic)
python train_value_model.py      # → value_model/       (critic warm-started from RM)
python ppo.py                    # → ppo_model/         (PPO-trained policy)
```

**Dataset:** `yahma/alpaca-cleaned`, first 200 examples.

**Base model:** `../SFT/final_model`.

---

## Inference Pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./ppo_model")
tokenizer = AutoTokenizer.from_pretrained("./ppo_model")

prompt = "### Instruction:\nExplain the water cycle in simple terms.\n\n### Input:\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
