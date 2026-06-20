from datasets import load_dataset
import json


# ==========================================================
# DATA CREATION FOR THE GRPO REWARD MODEL
# ==========================================================
# GRPO needs a reward function that can score a generated completion. Up to
# now grpo.py used a toy length-based heuristic. This script instead builds
# pairwise data (one "chosen" response, one "rejected" response per prompt)
# so we can train an actual reward model on it, the same way PPO/create_reward_data.py
# does for the PPO folder.

# Which dataset to pull from the Hugging Face Hub.
DATA_SOURCE = "yahma/alpaca-cleaned"

# How much of the dataset to use.
# Possible values:
# - "train" for everything
# - "train[:200]" for just the first 200 rows (used here to keep things fast)
# - "train[:10%]" for a percentage slice
SPLIT = "train[:200]"

# Where the generated preference pairs will be written.
OUTPUT_PATH = "./reward_pairs.jsonl"


# Build one preference pair for a given dataset row.
# "chosen" and "rejected" are the column names trl's RewardTrainer expects.
# In a real project these would be two different model outputs that a human
# (or a stronger model) compared and picked a winner for. Here we fake that
# judgement by writing the real Alpaca answer as "chosen" and a clearly
# unhelpful filler as "rejected", so the reward model has an easy, obvious
# signal to learn from.
def build_pair(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    return {
        "prompt": prompt,
        "chosen": f"{prompt} {example['output']}",
        "rejected": f"{prompt} I don't know, figure it out yourself.",
    }


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            pair_entry = build_pair(example)
            writer.write(json.dumps(pair_entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} reward model pairs to {OUTPUT_PATH}")
