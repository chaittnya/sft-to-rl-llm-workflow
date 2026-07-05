# PPO_PRM

## Task

**GSM8K math word problems** — same task as `GRPO_PRM/`, `GRPO_ORM/`, and `PPO_ORM/`.
Identical prompts across all four directories so that reward model type is the only
variable between experiments.

---

## Method: PPO (Proximal Policy Optimisation)

Same four-model PPO setup as `PPO_ORM/`: policy, reference, reward (PRM here), and value
model. See `PPO_ORM/README.md` for an explanation of what each model does and why PPO
needs all four.

---

## Reward Model: Process Reward Model (PRM)

The PRM scores individual reasoning steps. It was trained on growing step prefixes
(question + steps 0..i) with binary correct/corrupted labels, using the same
architecture as the ORM (`AutoModelForSequenceClassification`, `num_labels=1`,
MSELoss).

**Important limitation of PPO + PRM:**

In `GRPO_PRM/`, the reward is a Python function. We split the generated completion into
steps, run the PRM once per step, and average the scores. This is the proper way to use a
PRM — it gives a dense, step-level signal.

In PPO, `trl`'s `PPOTrainer` requires the reward model to be a `torch.nn.Module`. It
calls `reward_model.forward(input_ids)` on the full `(prompt + completion)` sequence after
each rollout. There is no way to inject step-splitting logic into that call without
modifying trl internals.

So in PPO_PRM the PRM ends up scoring the full sequence, not individual steps. This loses
the step-level granularity. What remains useful: the PRM was trained to detect errors in
step contexts, so its representations are more sensitive to local reasoning mistakes than
an ORM trained on whole-solution labels. It is a weaker signal than GRPO_PRM but a
stronger signal than PPO_ORM on step-structured text.

This trade-off is one reason to prefer GRPO over PPO when using a PRM.

**Value model:** Warm-started from the PRM checkpoint in `train_value_model.py`. Because
the PRM was trained at the step level, the value model inherits representations that are
tuned to judge step-by-step reasoning quality — a better initialisation for the critic
than starting from an ORM (which only learned whole-solution quality) or from the raw SFT
base.

---

## Training Pipeline

```
python create_data.py            # → ppo_prompts.jsonl  (200 math prompts)
python create_prm_data.py        # → prm_data.jsonl     (step-level correct/corrupted pairs)
python train_prm.py              # → prm_model/         (PRM — also seeds value model)
python create_value_data.py      # → value_data.jsonl   (binary regression targets)
python train_value_model.py      # → value_model/       (critic warm-started from PRM)
python ppo_prm.py                # → ppo_prm_model/     (PPO-trained policy)
```

**Dataset:** `openai/gsm8k` (config: `main`), first 200 training examples.

**Base model:** `../SFT/final_model`.

---

## Inference Pipeline

**Using the trained policy:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./ppo_prm_model")
tokenizer = AutoTokenizer.from_pretrained("./ppo_prm_model")

prompt = "Q: Sarah has 15 stickers and gives 7 to her friend. How many does she have left?\nA: Let me solve this step by step.\nStep 1:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Scoring a full solution with the PRM (as PPO uses it internally):**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

prm = AutoModelForSequenceClassification.from_pretrained("./prm_model", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("./prm_model")

text = "Q: Sarah has 15 stickers and gives 7 away. How many left?\nA: Step 1: 15 - 7 = 8. #### 8"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    score = prm(**inputs).logits.squeeze(-1).item()
print(f"PRM score (full sequence): {score:.3f}")
```

**Scoring step by step (how GRPO_PRM uses it — not available in PPO's training loop):**

```python
import re
steps_text = "Step 1: 15 - 7 = 8\nStep 2: So Sarah has 8 stickers. #### 8"
question_prefix = "Q: Sarah has 15 stickers and gives 7 away. How many left?\nA: Let me solve this step by step.\n"
steps = [s.strip() for s in re.split(r'(?=Step \d+:)', steps_text) if s.strip()]

for i, step in enumerate(steps):
    context = question_prefix + "\n".join(steps[:i+1])
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        score = prm(**inputs).logits.squeeze(-1).item()
    print(f"Step {i+1} score: {score:.3f}")
```
