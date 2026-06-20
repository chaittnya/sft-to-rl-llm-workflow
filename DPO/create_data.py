from datasets import load_dataset
import json


# Data creation for DPO
# Generate synthetic preference pairs for DPO training. In a rigorous study, the
# preferred and dispreferred responses would be annotated by humans, but this
# script is intended for a minimal reproducible example.
DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./dpo_pairs.jsonl"

# Build a pair of responses for each prompt. The preferred response is written to
# be more helpful, while the dispreferred response is intentionally weaker.
# Column names follow the standard preference format expected by trl's
# DPOTrainer: "prompt", "chosen", "rejected".
def build_pair(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    return {
        "prompt": prompt,
        "chosen": " This response is more helpful and complete.",
        "rejected": " This response is shorter and less detailed.",
    }

if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            pair_entry = build_pair(example)
            writer.write(json.dumps(pair_entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} DPO pairs to {OUTPUT_PATH}")
