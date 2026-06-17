from datasets import load_dataset
import json


# Data creation for PPO
# This script creates a small set of prompts in the format expected by the PPO
# training code. It is intended for prototyping rather than full-scale
# production training.
DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./ppo_prompts.jsonl"

# Construct a prompt from a dataset example. This prompt style mirrors the SFT
# input format so that the policy receives familiar inputs during RL tuning.
def build_prompt(example):
    return {
        "query": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            "### Response:"
        )
    }

if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            prompt_entry = build_prompt(example)
            writer.write(json.dumps(prompt_entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} PPO prompts to {OUTPUT_PATH}")
