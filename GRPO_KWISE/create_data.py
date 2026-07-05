from datasets import load_dataset
import json


# Data creation for GRPO (k-wise variant)
# Same prompt format as GRPO/create_data.py. GRPO generates and scores
# completions itself during training, so only prompts are needed here.
DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./grpo_items.jsonl"


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
