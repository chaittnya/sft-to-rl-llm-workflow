from datasets import load_dataset
import json


# This script builds the prompt file for GRPO with a Process Reward Model.
#
# I use GSM8K math problems here for the same reason as GRPO_ORM: there has to
# be an objectively correct answer for reward labelling to work. But PRM also
# needs one more thing — the model's output has to be split into discrete
# steps that can each be judged. Open-ended instruction following (Alpaca)
# does not have natural step boundaries. Math does: each arithmetic operation
# is a step, and we can check whether that step's numbers are consistent with
# what came before.
#
# The prompt format is important here. I tell the model to number its steps
# ("Step 1:", "Step 2:", etc.) because grpo_prm.py later uses that exact
# pattern to split the generated text. If the model just writes a paragraph
# with no step markers, the split finds nothing and the whole completion gets
# a reward of 0.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./grpo_items.jsonl"


def build_prompt(example):
    # "Step 1:" at the end of the prompt nudges the model to continue in the
    # numbered-step format we need. Without it, the model sometimes writes a
    # prose answer instead.
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
