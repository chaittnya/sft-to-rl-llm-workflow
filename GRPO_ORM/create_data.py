from datasets import load_dataset
import json


# This script builds the prompt file GRPO needs as input.
#
# I switched from the Alpaca instruction-following task (used in GRPO/) to
# GSM8K math word problems here. The reason is that an Outcome Reward Model
# only makes sense when there is an objectively correct answer to check
# against. With Alpaca, "is this a good answer?" depends entirely on human
# preference — there is no ground truth. With math, the answer is a number
# and it is either right or wrong, so ORM has something concrete to learn.
#
# GSM8K ("grade school math 8k") was released by OpenAI and contains around
# 8,500 math word problems with step-by-step solutions. Each example has a
# "question" field and an "answer" field. The answer is a full solution ending
# with "#### <number>" on the last line.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"   # there is also a "socratic" config; "main" is the standard one
SPLIT = "train[:200]"
OUTPUT_PATH = "./grpo_items.jsonl"


def build_prompt(example):
    # I ask the model to solve step by step and number each step. Strictly
    # speaking, GRPO_ORM only looks at the final answer, so the numbered
    # format is not required here. But it makes this prompt identical to the
    # one in GRPO_PRM/, which makes the two experiments easy to compare — same
    # prompts, different reward signals.
    #
    # "prompt" is the exact column name GRPOTrainer looks for. Any other name
    # and it either crashes or silently ignores the data.
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
