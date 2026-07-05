from datasets import load_dataset
import json


# This script creates the training data for the Outcome Reward Model.
#
# What is an ORM?
# An Outcome Reward Model scores a complete solution and gives a single
# yes-or-no verdict: was the final answer correct? It does not care about
# the reasoning steps — it only looks at the end result. This is different
# from the Bradley-Terry reward model in GRPO/ which always compares two
# responses to each other. ORM judges one response independently.
#
# Why math?
# For ORM to work, "correct" has to be a binary fact, not a preference.
# Alpaca responses have no ground truth — a response can be more or less
# helpful but there is no objectively right answer. GSM8K math problems do
# have an objectively right answer (the number after "####"), which is why
# this task makes ORM training possible.
#
# How I construct the wrong answers:
# My first instinct was to write obviously bad answers like "I don't know",
# but that makes the ORM's job too easy — it could just learn to detect
# short or badly-formed text rather than mathematical correctness. Instead
# I use the reference solution from the *next* example as the wrong answer.
# That solution is well-written and has a real step-by-step structure, it
# just solves a different problem, so the numbers are wrong for this question.
# The ORM has to actually understand the math to tell the difference.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./orm_data.jsonl"


def build_orm_examples(examples_list):
    rows = []
    n = len(examples_list)
    for i, example in enumerate(examples_list):
        # This is the prefix both the correct and wrong solution will share.
        # I use the same "Q: ... A: Let me solve this step by step." format as
        # the prompts in create_data.py so the reward model sees text in the
        # same form as the completions it will score during GRPO training.
        prefix = (
            f"Q: {example['question']}\n"
            "A: Let me solve this step by step.\n"
        )

        correct_solution = example["answer"]

        # (i + 1) % n wraps around so the last example uses the first example's
        # answer. This circular shift gives every example a unique wrong answer
        # without any randomness that would change the dataset between runs.
        wrong_solution = examples_list[(i + 1) % n]["answer"]

        rows.append({"text": prefix + correct_solution, "label": 1.0})
        rows.append({"text": prefix + wrong_solution,   "label": 0.0})
    return rows


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, CONFIG, split=SPLIT)
    # I convert to a list so I can index it by position for the circular shift.
    # The HuggingFace Dataset object does not support this kind of indexing.
    examples_list = list(dataset)
    rows = build_orm_examples(examples_list)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} ORM training examples to {OUTPUT_PATH}")
