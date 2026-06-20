from datasets import load_dataset
import json


# ==========================================================
# DATA CREATION FOR THE REWARD MODEL
# ==========================================================
# Before we can run PPO, the policy needs a reward model that actually knows
# how to score a response. A reward model is trained on pairs of responses
# where one is marked "chosen" (better) and the other "rejected" (worse).
# This script builds that pairwise data. It is the same idea as DPO/create_data.py,
# just kept as its own file so the PPO folder is self-contained.

# Which dataset to pull from the Hugging Face Hub.
# Possible values:
# - any dataset repo id on the hub, e.g. "yahma/alpaca-cleaned"
# - a local path if you have your own dataset saved on disk
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
# judgement by writing a clearly more helpful sentence as "chosen" and a
# clearly weaker one as "rejected", so the reward model has an easy, obvious
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
