import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# This script trains the Plackett-Luce k-wise reward model.
#
# Why Plackett-Luce instead of Bradley-Terry?
# The standard reward model in PPO/ uses Bradley-Terry: it sees a (chosen,
# rejected) pair and minimises -log σ(score_chosen - score_rejected). That is
# one comparison per training example. Plackett-Luce generalises this: it sees
# a fully ranked list of k completions and learns the model that is most
# consistent with that entire ordering. For k=4 and a random ordering, the
# information content per example is log_2(4!) ≈ 4.6 bits compared to 1 bit
# for a single pairwise comparison. More signal per example = faster and more
# stable reward model learning.
#
# The Plackett-Luce loss:
# Given k completions ordered best (index 0) to worst (index k-1) with model
# scores s_0, s_1, ..., s_{k-1}, the loss is the negative log-likelihood of
# the observed ranking under the Plackett-Luce model:
#
#   L = -sum_{i=0}^{k-1} [ s_i - logsumexp(s_i, s_{i+1}, ..., s_{k-1}) ]
#
# At each position i the model "peels off" the winner (item i) from the
# remaining pool and computes how likely that was. Summing over all positions
# gives the total probability of the observed ordering. Negating gives the loss
# to minimise.
#
# After training, the model is used exactly like any other reward model: call it
# on a single (prompt + completion) text and read the scalar logit. The k-wise
# training happens offline; inference is identical to Bradley-Terry.
#
# This is an important difference from PRM in PPO_PRM/: PRM genuinely needs
# step-by-step scoring at inference time (which PPOTrainer cannot do), so
# PPO_PRM loses some of PRM's expressiveness. Plackett-Luce has no such
# limitation — its inference call is just a single forward pass.

BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./kwise_reward_model"
DATA_PATH = "./kwise_reward_data.jsonl"
K = 4  # must match create_kwise_reward_data.py


class KWiseCollator:
    # The dataset has K texts per row stored as text_0, text_1, ..., text_{K-1}.
    # Standard DataCollatorWithPadding only handles a single text per row.
    # This collator tokenizes all K texts and stacks them into a 3D tensor
    # [batch_size, K, seq_len] so that PlackettLuceTrainer can flatten and score
    # all K candidates in one forward pass.
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_mask = []

        for feature in features:
            # features is a list of dicts, one per dataset row.
            # Each dict has keys text_0 ... text_{K-1}.
            texts = [feature[f"text_{i}"] for i in range(K)]

            # Tokenize all K texts at once. padding="max_length" ensures every
            # sequence in the K-tuple has the same length so they can be stacked
            # into a regular tensor.
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_input_ids.append(tokenized["input_ids"])        # [K, seq_len]
            batch_attention_mask.append(tokenized["attention_mask"])

        return {
            "input_ids":      torch.stack(batch_input_ids),       # [batch_size, K, seq_len]
            "attention_mask": torch.stack(batch_attention_mask),
        }


class PlackettLuceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch_size, k, seq_len = inputs["input_ids"].shape

        # Flatten to [batch_size * K, seq_len] so we can score all K
        # completions for all batch items in a single model forward call.
        # This is more efficient than k separate forward passes because modern
        # GPUs are optimised for large batch operations.
        flat_ids  = inputs["input_ids"].view(batch_size * k, seq_len)
        flat_mask = inputs["attention_mask"].view(batch_size * k, seq_len)

        # scores: [batch_size * K, 1] → squeeze to [batch_size * K]
        scores = model(input_ids=flat_ids, attention_mask=flat_mask).logits.squeeze(-1)

        # Reshape back to [batch_size, K] with best completion at index 0.
        scores = scores.view(batch_size, k)

        # Plackett-Luce loss: iterate over positions 0 to K-1. At position i,
        # item i is the "winner" of the remaining pool scores[:, i:].
        # The loss penalises how much lower s_i is compared to the logsumexp
        # of all remaining candidates — equivalently, it rewards making s_i
        # the largest in its pool.
        loss = torch.zeros(batch_size, device=scores.device)
        for i in range(k):
            # torch.logsumexp is numerically stable and equivalent to
            # log(sum(exp(scores[:, i:]))) but without the overflow risk.
            loss = loss - (scores[:, i] - torch.logsumexp(scores[:, i:], dim=-1))

        return (loss.mean(), scores) if return_outputs else loss.mean()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=1,
        device_map="auto",
    )

    # Zero the scoring-head bias so all completions start at the same score
    # before training. If the bias is non-zero, the model begins with a
    # systematic preference for shorter or longer sequences (depending on sign)
    # which slows convergence.
    if hasattr(model, "score") and hasattr(model.score, "bias") and model.score.bias is not None:
        torch.nn.init.zeros_(model.score.bias)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        # Each dataset row contains K=4 texts. The effective number of forward
        # passes per batch step is per_device_train_batch_size * K = 2 * 4 = 8.
        # Keeping the nominal batch size at 2 prevents OOM on 4GB VRAM.
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        logging_steps=10,
        report_to="none",
        # remove_unused_columns=False is required here. By default Trainer drops
        # any column not in the model's forward() signature. Our dataset has
        # text_0...text_{K-1} columns that the model's forward() does not know
        # about — they are consumed by the custom collator instead. If Trainer
        # drops them, the collator receives empty dicts.
        remove_unused_columns=False,
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
