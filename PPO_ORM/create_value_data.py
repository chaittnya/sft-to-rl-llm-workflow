from datasets import load_dataset
import json


# This script creates training data for the value model (PPO's critic).
#
# PPO needs two separate scorer models that have similar architectures but
# very different roles:
#
# The reward model (ORM) is frozen once training starts. Its job is to give
# the policy an honest signal about whether a completed response was good or
# bad. It is never updated during PPO.
#
# The value model is updated at every PPO step. Its job is to predict, before
# seeing the full completion, how good the eventual reward will be. PPO uses
# the difference between the actual reward and the value model's prediction as
# the "advantage" — the measure of whether a completion was better or worse
# than expected. If the advantage is positive, the policy is pushed to produce
# more responses like this one. If it is negative, fewer.
#
# Because the value model has to track the policy's changing behavior, it
# cannot just be a copy of the reward model — it needs to stay trainable.
# But it makes sense to initialise it from the ORM checkpoint rather than
# from the raw SFT base, because the ORM already has representations that
# correspond to "how likely is this text to lead to a correct answer", which
# is close to what the value model needs to predict.
#
# The training data format here is the same as the ORM data: (text, 0.0 or 1.0).
# The value model essentially starts with the same binary supervision as the ORM
# and then gets fine-tuned further by PPO during RL training.

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
        # Same circular shift as the ORM data: fluent wrong answer, not garbage.
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
