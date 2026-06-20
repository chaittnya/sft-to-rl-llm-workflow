from datasets import load_dataset
import json


# Data creation for GRPO
# Create a small set of prompts for reward-guided policy optimization. GRPO
# generates and scores completions itself during training, so this script only
# needs to supply prompts; the reward function lives in grpo.py.
DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./grpo_items.jsonl"

# Build a prompt entry. The "prompt" column name matches what trl's
# GRPOTrainer expects.
def build_item(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    return {"prompt": prompt}

if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            item = build_item(example)
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} GRPO items to {OUTPUT_PATH}")
