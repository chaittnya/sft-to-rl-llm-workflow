from datasets import load_dataset
import json


# Value model training data for PPO_KWISE.
#
# PPO needs a value model (critic) separate from the reward model, for the
# same reason as in PPO/, PPO_ORM/, and PPO_PRM/: the critic predicts expected
# future reward before the full completion is available, and PPO uses the gap
# between prediction and actual reward (the advantage) to update the policy.
#
# The k-wise reward model was trained with the Plackett-Luce ranking loss on
# 4 candidates at a time. It cannot be used directly as the critic because:
#   1. It was trained with a batch-based ranking loss that needs K texts at once,
#      so scoring a single text was not part of its original training objective.
#      (In practice it can score single texts — and ppo_kwise.py uses it that way
#      as the frozen reward model — but its calibration for that task is weaker.)
#   2. The critic needs to be updatable during PPO training, so it must be a
#      separate object from the frozen reward model.
#
# I warm-start the value model from the k-wise reward model checkpoint. This
# gives it useful initialisation: the PL model already learned to distinguish
# good from bad responses (just in a ranking sense), and the value model
# fine-tunes those representations for single-response regression.
#
# The training data is the same binary format as PPO/create_value_data.py:
# good response → label 1.0, bad response → label 0.0.

DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./value_data.jsonl"


def build_examples(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    good_response = f"{prompt} {example['output']}"
    bad_response  = f"{prompt} I don't know, figure it out yourself."
    return [
        {"text": good_response, "label": 1.0},
        {"text": bad_response,  "label": 0.0},
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
