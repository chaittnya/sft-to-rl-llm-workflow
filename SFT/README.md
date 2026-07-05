# SFT — Supervised Fine-Tuning

## Task

**Alpaca-style instruction following** — given an instruction and optional input context,
generate a helpful response. The dataset is `yahma/alpaca-cleaned`, a cleaned version of
Stanford Alpaca's 52K instruction-following examples.

SFT is the first stage in all of the RL pipelines in this repository. Every other
directory (`PPO/`, `DPO/`, `GRPO/`, etc.) starts from the checkpoint trained here.

---

## Method: Supervised Fine-Tuning (SFT)

SFT is the simplest form of language model training for downstream tasks. We take a
pre-trained base model and continue training it on (prompt, response) pairs from a
curated dataset. The model learns to predict the response tokens given the prompt tokens,
with standard cross-entropy loss.

No reward signal, no RL, no preference comparisons — just next-token prediction on
high-quality examples. This is why SFT is run first: it teaches the model the format and
style of helpful responses before any reward-based fine-tuning begins.

**Two scripts are provided:**

- `sft.py` — standard full fine-tuning. All model parameters are updated.
- `lora.py` — LoRA fine-tuning. Only a small set of low-rank adapter parameters are
  trained, which uses far less memory and is much faster. The run_pipeline script uses
  `lora.py` by default for this reason.

**dtype choice:** Both scripts load the model with `torch_dtype=torch.float32` even
though the base model supports bfloat16. This is because `fp16=True` in
`TrainingArguments` uses PyTorch's GradScaler, which requires float16 master weights (not
bfloat16). Loading in float32 and letting the GradScaler handle the fp16 conversion
avoids a dtype mismatch crash.

---

## Reward Model

None. SFT does not use a reward model. The training signal is cross-entropy loss on the
reference responses in the dataset.

---

## Training Pipeline

```
python sft.py        # full fine-tune  → sft_model/
# or
python lora.py       # LoRA fine-tune  → final_model/   (used by all downstream scripts)
```

**Dataset:** `yahma/alpaca-cleaned`, first 200 examples of the training split.

**Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (135M parameter decoder-only model).

After training, the checkpoint in `final_model/` is the starting point for all PPO, DPO,
GRPO, and reward model training in sibling directories.

---

## Inference Pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# If using the LoRA checkpoint, load the base model and merge the adapter.
base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = PeftModel.from_pretrained(base, "./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model")

prompt = "### Instruction:\nExplain what photosynthesis is.\n\n### Input:\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

You can also test the raw base model (`lora_testing.py`) to compare outputs before and
after fine-tuning.
