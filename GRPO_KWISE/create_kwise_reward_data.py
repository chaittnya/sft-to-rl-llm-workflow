from datasets import load_dataset
import json


# ==========================================================
# K-WISE (PLACKETT-LUCE) REWARD DATA CREATION
# ==========================================================
# The standard Bradley-Terry reward model only compares 2 completions at a time
# (chosen vs rejected). Plackett-Luce extends this to k completions at once,
# which gives the model more signal per prompt: it learns to rank all k
# responses instead of just separating one pair.
#
# This script generates 4 completions per prompt, ordered best -> worst:
#   rank 0 (best)   - the real Alpaca reference answer
#   rank 1          - a shorter, less complete version of the real answer
#   rank 2          - a vague, generic filler
#   rank 3 (worst)  - an outright refusal / unhelpful response
#
# train_kwise_reward_model.py reads this file and treats the order as a
# complete ranking for the Plackett-Luce loss.

DATA_SOURCE = "yahma/alpaca-cleaned"

# How much data to use.
# Options: "train" for everything, "train[:200]" for the first 200 rows.
SPLIT = "train[:200]"

OUTPUT_PATH = "./kwise_reward_data.jsonl"

# Number of candidates per prompt.
K = 4


def build_kwise(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    full_answer = example["output"].strip()

    # Create k completions of descending quality.
    # We concatenate each completion onto the prompt so the model sees the
    # full context (same as what the reward model will see at GRPO inference time).
    completions = [
        # rank 0 - complete, correct answer
        f"{prompt} {full_answer}",

        # rank 1 - truncated answer; still useful but noticeably incomplete.
        # We take the first half of words from the real output.
        f"{prompt} {' '.join(full_answer.split()[:max(1, len(full_answer.split()) // 2)])}...",

        # rank 2 - very generic; says something but nothing specific
        f"{prompt} That's a good question. There are many ways to think about this.",

        # rank 3 - outright unhelpful refusal
        f"{prompt} I don't know, figure it out yourself.",
    ]

    # Store as text_0 ... text_{K-1} so the custom collator in
    # train_kwise_reward_model.py can look them up by index.
    entry = {f"text_{i}": completions[i] for i in range(K)}
    return entry


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            entry = build_kwise(example)
            writer.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} k-wise ranking examples (k={K}) to {OUTPUT_PATH}")
