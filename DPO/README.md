# DPO — Direct Preference Optimisation

## Task

**Alpaca-style instruction following** using the `yahma/alpaca-cleaned` dataset.

DPO optimises the policy using preference pairs — examples where one response is labelled
as "chosen" (preferred) and another is labelled as "rejected" (not preferred). The goal
is to make the model more likely to produce chosen-style responses and less likely to
produce rejected-style ones.

---

## Method: DPO (Direct Preference Optimisation)

DPO is an RL-free alternative to PPO for preference alignment. Standard RLHF with PPO
has three stages: train a reward model on preferences, then use PPO with that reward
model to update the policy. DPO collapses the reward model training and RL steps into one
by deriving a closed-form update directly from the preference data.

The insight behind DPO: given a preference dataset, you can write down the optimal policy
in closed form as a function of the reference policy and the preference ratios. Rearranging
that expression gives a loss you can optimise directly on (chosen, rejected) pairs without
ever training a reward model separately.

In practice this means DPO is simpler and cheaper than PPO (no reward model, no value
model, no rollout generation during training), at the cost of being limited to offline
preferences. You cannot use DPO to explore and discover new good responses; you can only
learn from the preferences you already have.

The `trl` library's `DPOTrainer` handles all the details. It loads the reference model
automatically from the same checkpoint as the policy.

---

## Reward Model

None. DPO does not train a reward model. The preference signal is encoded directly in the
loss function via the Bradley-Terry model of human preferences.

The `create_data.py` script generates synthetic preference pairs where the reference
Alpaca output is the "chosen" response and a clearly unhelpful filler ("I don't know,
figure it out yourself.") is the "rejected" response. In a real DPO experiment, these
would be actual human preference labels.

---

## Training Pipeline

```
python create_data.py    # → dpo_pairs.jsonl   (chosen/rejected pairs from Alpaca)
python dpo.py            # → dpo_model/        (DPO-trained policy)
```

**Dataset:** `yahma/alpaca-cleaned`, first 200 examples.

**Base model:** `../SFT/final_model` — run `SFT/lora.py` first.

**Column format:** `DPOTrainer` expects three columns: `prompt`, `chosen`, and `rejected`.
`chosen` and `rejected` contain only the response text (not the full prompt + response),
because DPOTrainer prepends the prompt internally.

---

## Inference Pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./dpo_model")
tokenizer = AutoTokenizer.from_pretrained("./dpo_model")

prompt = "### Instruction:\nWrite a short poem about autumn.\n\n### Input:\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The DPO-trained model should produce more helpful responses than the raw SFT model and
be less likely to produce the kind of dismissive, unhelpful responses it was trained to
avoid.
