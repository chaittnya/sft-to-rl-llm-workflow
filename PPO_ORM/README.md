# PPO_ORM

## Task

**GSM8K math word problems** — generate a step-by-step solution to an arithmetic word
problem and produce the correct final answer.

Same task as `GRPO_ORM/`. Using the same task across GRPO_ORM, GRPO_PRM, PPO_ORM, and
PPO_PRM makes it possible to compare the four approaches on equal footing.

---

## Method: PPO (Proximal Policy Optimisation)

PPO is the classical RL algorithm for language models. It needs four models loaded
simultaneously:

1. **Policy model** — the LLM being trained. Starts from the SFT checkpoint.
2. **Reference model** — a frozen copy of the initial policy. Used to compute a KL
   penalty that prevents the policy from drifting into degenerate behaviour (e.g.
   producing nonsense that happens to score well on the reward model).
3. **Reward model** (ORM here) — scores each completed response. Frozen throughout training.
4. **Value model** (critic) — estimates the expected future reward from the current state.
   Updated every step alongside the policy. PPO uses `actual_reward - value_estimate` as
   the advantage for each completion.

The main difference from GRPO: **PPO needs a value model, GRPO does not.** GRPO computes
its baseline from a group of completions; PPO computes it with a trained neural network.
The value model adds overhead (an extra training step before PPO and an extra model in
memory) but can give more stable advantage estimates in principle.

---

## Reward Model: Outcome Reward Model (ORM)

Same ORM architecture and training as `GRPO_ORM/` — `AutoModelForSequenceClassification`
with `num_labels=1`, trained with MSELoss on binary correct/wrong solution labels.

**How PPO uses the ORM differently from GRPO:**
In `GRPO_ORM/grpo_orm.py`, the ORM is called inside a Python reward function that we
write ourselves. In PPO, `trl.experimental.ppo.PPOTrainer` accepts the ORM as a
`torch.nn.Module` and calls it internally after each rollout. The ORM fits this API
perfectly because it already produces a scalar score for any input sequence — no
extra wrapping is needed.

**Value model:** Trained in `train_value_model.py`, warm-started from the ORM checkpoint.
Loading the value model from the ORM checkpoint (rather than the raw SFT base) gives it a
head start: it already has weights that encode solution quality, so it only needs light
fine-tuning to predict expected future reward instead of outcome correctness.

---

## Training Pipeline

```
python create_data.py            # → ppo_prompts.jsonl  (200 math prompts)
python create_orm_data.py        # → orm_data.jsonl     (400 correct/wrong solution pairs)
python train_orm.py              # → orm_model/         (ORM checkpoint — also seeds value model)
python create_value_data.py      # → value_data.jsonl   (regression targets for the critic)
python train_value_model.py      # → value_model/       (critic warm-started from ORM)
python ppo_orm.py                # → ppo_orm_model/     (PPO-trained policy)
```

**Dataset:** `openai/gsm8k` (config: `main`), first 200 training examples.

**Base model:** `../SFT/final_model`. All four models in `ppo_orm.py` are initialised from
this or from checkpoints derived from it.

---

## Inference Pipeline

**Using the trained policy:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./ppo_orm_model")
tokenizer = AutoTokenizer.from_pretrained("./ppo_orm_model")

prompt = "Q: A shop sells 6 pens for $2 each. How much do all pens cost?\nA: Let me solve this step by step.\nStep 1:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Scoring a candidate response with the ORM:**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

orm = AutoModelForSequenceClassification.from_pretrained("./orm_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./orm_model")

text = "Q: A shop sells 6 pens for $2 each. How much?\nA: Step 1: 6 × 2 = 12. #### 12"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    score = orm(**inputs).logits.squeeze(-1).item()
print(f"ORM score: {score:.3f}")
```
