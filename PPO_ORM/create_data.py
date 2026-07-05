from datasets import load_dataset
import json


# This script creates the prompt dataset for PPO with an Outcome Reward Model.
#
# Same task and format as GRPO_ORM/create_data.py — GSM8K math problems with
# numbered steps. The only difference is the output filename: PPO/ convention
# uses ppo_prompts.jsonl. The prompts themselves are identical to GRPO_ORM so
# the two algorithms are trained on exactly the same questions and can be
# compared fairly.
#
# PPOTrainer does not use raw text prompts directly. In ppo_orm.py I tokenise
# this file and replace the "prompt" text column with an "input_ids" column
# before passing it to the trainer. That tokenisation step is why this script
# only needs to produce clean prompt strings.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./ppo_prompts.jsonl"


def build_prompt(example):
    return {
        "prompt": (
            f"Q: {example['question']}\n"
            "A: Let me solve this step by step.\n"
            "Step 1:"
        )
    }


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, CONFIG, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            item = build_prompt(example)
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} math prompts to {OUTPUT_PATH}")
