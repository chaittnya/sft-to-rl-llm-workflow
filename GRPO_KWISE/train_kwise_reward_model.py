import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ==========================================================
# PLACKETT-LUCE (K-WISE) REWARD MODEL TRAINING
# ==========================================================
# Bradley-Terry (the standard RewardTrainer) only ever compares 2 items at
# once. Plackett-Luce generalises this to a full ranking over k items.
#
# Given k completions for a prompt ranked from best (index 0) to worst
# (index k-1), the Plackett-Luce log-likelihood of observing that ranking is:
#
#   log P(ranking) = sum_{i=0}^{k-1} [ s_i - logsumexp(s_i, s_{i+1}, ..., s_{k-1}) ]
#
# Intuitively: at each position i we "remove" the winner from the pool and ask
# how likely the model is to pick it over the remaining candidates. Summing
# over all positions gives the full ranking probability.
#
# The trained model is still AutoModelForSequenceClassification with
# num_labels=1 — it outputs a single scalar score per (prompt + completion).
# Only the training objective changes; inference is identical to the pairwise
# reward model, so grpo_kwise.py can swap this checkpoint in directly.

BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./kwise_reward_model"
DATA_PATH = "./kwise_reward_data.jsonl"

# Must match the K used in create_kwise_reward_data.py.
K = 4


# -------------------------------------------------------
# Custom data collator
# -------------------------------------------------------
# Each dataset row has text_0 ... text_{K-1} (strings, ordered best->worst).
# The collator tokenizes all K texts and stacks them so each batch element is
# a tensor of shape [K, seq_len] instead of the usual [seq_len].
# The trainer then sees inputs of shape [batch_size, K, seq_len].
class KWiseCollator:
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_mask = []

        for feature in features:
            texts = [feature[f"text_{i}"] for i in range(K)]
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_input_ids.append(tokenized["input_ids"])       # [K, seq_len]
            batch_attention_mask.append(tokenized["attention_mask"])

        return {
            # [batch_size, K, seq_len]
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
        }


# -------------------------------------------------------
# Custom Trainer with Plackett-Luce loss
# -------------------------------------------------------
class PlackettLuceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch_size, k, seq_len = inputs["input_ids"].shape

        # Flatten [batch_size, K, seq_len] -> [batch_size*K, seq_len] so the
        # forward pass runs in a single GPU kernel instead of K separate ones.
        flat_ids = inputs["input_ids"].view(batch_size * k, seq_len)
        flat_mask = inputs["attention_mask"].view(batch_size * k, seq_len)

        # scores: [batch_size * K]
        scores = model(input_ids=flat_ids, attention_mask=flat_mask).logits.squeeze(-1)

        # Reshape back to [batch_size, K].
        # Rows are ordered best (index 0) -> worst (index K-1), matching the
        # order we created in create_kwise_reward_data.py.
        scores = scores.view(batch_size, k)

        # Plackett-Luce negative log-likelihood.
        # At each position i, the winner is item i, competing against items
        # i, i+1, ..., K-1 (the ones not yet "removed").
        # Loss contribution at i: s_i - logsumexp(s_i ... s_{K-1})
        #   -> we want s_i to be large relative to the remaining pool.
        loss = torch.zeros(batch_size, device=scores.device)
        for i in range(k):
            # scores[:, i:] is the pool of remaining candidates at step i.
            loss = loss - (scores[:, i] - torch.logsumexp(scores[:, i:], dim=-1))

        # Average over the batch.
        loss = loss.mean()

        return (loss, scores) if return_outputs else loss


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=1,
        device_map="auto",
    )
    # Zero the scoring-head bias so all completions start from the same score.
    if hasattr(model, "score") and hasattr(model.score, "bias") and model.score.bias is not None:
        torch.nn.init.zeros_(model.score.bias)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,  # each row has K=4 texts; effective GPU batch = 2*4 = 8
        learning_rate=1e-5,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,    # keep text_0...text_{K-1} columns
    )

    trainer = PlackettLuceTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=KWiseCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"K-wise reward model saved to {OUTPUT_DIR}")
