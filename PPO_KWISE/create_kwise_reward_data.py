from datasets import load_dataset
import json


# This script creates the k-wise ranking data for the Plackett-Luce reward model.
#
# Same logic as GRPO_KWISE/create_kwise_reward_data.py — see that file for the
# full explanation of what Plackett-Luce is and why it is more informative than
# pairwise Bradley-Terry training. The short version: instead of telling the
# model "A is better than B" (one bit of signal), we tell it the full ranking
# "rank 0 > rank 1 > rank 2 > rank 3" (log_2(4!) ≈ 4.6 bits per example).
#
# One thing worth clarifying for the PPO context: the k-wise training happens
# entirely offline, before PPO starts. At inference time — when PPO's rollout
# loop calls the reward model — the model just takes a single (prompt + completion)
# text and returns one scalar score, exactly like the Bradley-Terry model does.
# The k-wise structure only exists during training to create a richer gradient
# signal. This is different from PRM in PPO_PRM, where the step-splitting logic
# genuinely cannot be expressed in PPOTrainer's reward model API. With
# Plackett-Luce there is no such limitation.

DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./kwise_reward_data.jsonl"
K = 4


def build_kwise(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    full_answer = example["output"].strip()

    # Four completions in descending quality order. The Plackett-Luce loss
    # treats the index as the rank, so text_0 must be the best, text_3 the worst.
    completions = [
        # rank 0 — full reference answer
        f"{prompt} {full_answer}",

        # rank 1 — first half of words only; starts well but trails off
        f"{prompt} {' '.join(full_answer.split()[:max(1, len(full_answer.split()) // 2)])}...",

        # rank 2 — vague and content-free, but at least attempts to engage
        f"{prompt} That's a good question. There are many ways to think about this.",

        # rank 3 — outright unhelpful refusal
        f"{prompt} I don't know, figure it out yourself.",
    ]

    # Keys text_0...text_{K-1} let the KWiseCollator in train_kwise_reward_model.py
    # look up each completion by rank index.
    return {f"text_{i}": completions[i] for i in range(K)}


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            entry = build_kwise(example)
            writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} k-wise ranking examples (k={K}) to {OUTPUT_PATH}")
