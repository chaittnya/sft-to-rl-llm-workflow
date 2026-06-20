from datasets import load_dataset
import json


# ==========================================================
# DATA CREATION FOR THE VALUE MODEL
# ==========================================================
# The value model has a different job than the reward model. The reward
# model only needs to rank two responses against each other (chosen vs
# rejected). The value model needs to output an actual number for a single
# response on its own, since PPO uses that number as a baseline to work out
# how much better or worse a response was than expected.
# Because of that, the value model is trained with a regression target
# (a plain number) instead of a pairwise comparison, which is why this data
# looks a bit different from create_reward_data.py even though it comes from
# the same underlying responses.

# Which dataset to pull from the Hugging Face Hub.
DATA_SOURCE = "yahma/alpaca-cleaned"

# How much of the dataset to use.
# Possible values:
# - "train" for everything
# - "train[:200]" for just the first 200 rows (used here to keep things fast)
SPLIT = "train[:200]"

# Where the generated regression examples will be written.
OUTPUT_PATH = "./value_data.jsonl"


# Build two regression examples from one dataset row: a "good" response that
# should get a high value, and a "bad" response that should get a low value.
# The label is a plain float target, not a class index, because the value
# model is trained as a regressor (num_labels=1, problem_type="regression").
def build_examples(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    good_response = f"{prompt} {example['output']}"
    bad_response = f"{prompt} I don't know, figure it out yourself."
    return [
        {"text": good_response, "label": 1.0},
        {"text": bad_response, "label": 0.0},
    ]


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            for row in build_examples(example):
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} value model examples to {OUTPUT_PATH}")
