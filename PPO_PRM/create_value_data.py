from datasets import load_dataset
import json


# Value model training data for PPO_PRM.
#
# Same format as PPO_ORM/create_value_data.py: binary 0/1 labels on full
# solutions. See that file for the full explanation of why PPO needs a value
# model separately from the reward model.
#
# One question worth thinking about: the PRM was trained on step-level data,
# so should the value model data also be step-level? In principle, yes — the
# value at step i should estimate the expected cumulative reward from step i
# onwards, not just the final outcome. In practice, for this small demonstration
# the binary full-solution labels work well enough as a starting point. The
# value model is warm-started from the PRM checkpoint anyway, so it already
# has step-aware representations before seeing this data. The full-solution
# labels then fine-tune it to predict end-of-episode outcomes, which is what
# PPO's advantage computation actually needs.
#
# A more principled approach would label each step prefix with the mean PRM
# score of all remaining steps — that would give the value model proper
# per-step targets. I leave that as a possible extension.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./value_data.jsonl"


def build_examples(examples_list):
    rows = []
    n = len(examples_list)
    for i, example in enumerate(examples_list):
        prefix = (
            f"Q: {example['question']}\n"
            "A: Let me solve this step by step.\n"
        )
        correct = prefix + example["answer"]
        wrong = prefix + examples_list[(i + 1) % n]["answer"]

        rows.append({"text": correct, "label": 1.0})
        rows.append({"text": wrong,   "label": 0.0})
    return rows


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, CONFIG, split=SPLIT)
    examples_list = list(dataset)
    rows = build_examples(examples_list)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} value model examples to {OUTPUT_PATH}")
