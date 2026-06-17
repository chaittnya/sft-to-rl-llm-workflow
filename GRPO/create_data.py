from datasets import load_dataset
import json


# Data creation for GRPO
# Create a small set of prompt-reward examples to illustrate how reward-guided
# policy optimization can be launched from a supervised base model.
DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./grpo_items.jsonl"

# A toy reward function used only for illustrative purposes. A real reward
# function would be learned from human judgments or a dedicated reward model.
def compute_reward(text):
    return min(len(text.split()) / 20.0, 1.0)

# Build a prompt entry that includes a placeholder response and its reward.
def build_item(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    placeholder_response = "This is a sample response used for reward estimation."
    return {
        "query": prompt,
        "sample_response": placeholder_response,
        "reward": compute_reward(placeholder_response),
    }

if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            item = build_item(example)
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} GRPO items to {OUTPUT_PATH}")
