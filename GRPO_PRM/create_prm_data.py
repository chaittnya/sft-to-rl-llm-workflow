from datasets import load_dataset
import json
import re
import random


# This script creates training data for the Process Reward Model.
#
# What is a PRM?
# Where ORM gives one score at the end of a solution, a Process Reward Model
# scores each individual reasoning step. The key insight behind PRM is that
# if you can tell the model "step 3 was wrong" instead of just "this whole
# solution was wrong", the learning signal is much denser and more informative.
# For a 5-step solution where only step 3 is wrong, ORM gives a reward of 0
# for the whole thing. PRM can give rewards of [1, 1, 0, ?, ?], which tells
# the policy exactly where the reasoning broke down.
#
# Why GSM8K again?
# Same reason as ORM: we need objectively correct steps. Each line in a GSM8K
# answer is an arithmetic operation with a specific numerical result, so we
# can tell whether a step is wrong by checking its numbers. That would not be
# possible with open-ended text like Alpaca responses.
#
# How I get step-level labels without human annotators:
# Real PRM datasets (like PRM800K from OpenAI) have human annotators label
# each step. I cannot afford that here, so I use a synthetic proxy: the real
# steps from GSM8K's reference answers are labelled correct (1.0), and
# corrupted versions of those steps — where one number is nudged by a small
# offset — are labelled incorrect (0.0). The corruption is small enough that
# the text still reads like a plausible math step; the PRM has to actually
# check the arithmetic to find the error.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./prm_data.jsonl"

random.seed(42)


def extract_steps(answer_text):
    # GSM8K answers look like:
    #   "Step 1: blah blah\nStep 2: blah\n#### 42"
    # or sometimes just newline-separated sentences with "#### 42" at the end.
    # I split on newlines, strip blank lines, and drop the "#### N" line because
    # that is just the final answer, not a reasoning step.
    lines = [line.strip() for line in answer_text.split("\n") if line.strip()]
    return [line for line in lines if not line.startswith("####")]


def corrupt_step(step):
    # Find every integer in this step and replace one of them with a slightly
    # different number. The offset is small (2-7) so the corrupted step still
    # looks realistic — it is just arithmetically inconsistent with the
    # question. This is harder to detect than replacing a number with "999" or
    # deleting words entirely.
    #
    # I use re.findall to get all digit sequences, pick one at random, compute
    # a replacement, then replace only the first occurrence in the string.
    # Replacing only the first occurrence matters because if the same number
    # appears twice in a step (e.g. "5 * 5 = 25"), changing both copies would
    # actually make the step internally consistent with the wrong number.
    numbers = re.findall(r'\d+', step)
    if not numbers:
        # Some steps have no numbers (e.g. "Let's call the unknown x.").
        # I append a wrong claim rather than leaving the step unchanged,
        # otherwise the PRM gets a "correct" and an identical "incorrect" row
        # for that step, which is useless.
        return step + " This is incorrect."
    target = random.choice(numbers)
    offset = random.choice([-3, -2, 2, 3, 5, 7])
    replacement = str(max(0, int(target) + offset))  # max(0,...) avoids negative numbers
    return step.replace(target, replacement, 1)


def build_prm_examples(example):
    question_prefix = (
        f"Q: {example['question']}\n"
        "A: Let me solve this step by step.\n"
    )
    steps = extract_steps(example["answer"])

    rows = []
    for i, step in enumerate(steps):
        # The PRM is trained on growing contexts, not individual steps in
        # isolation. At position i, the input is the question plus every step
        # up to and including step i. This is important because steps do not
        # always make sense on their own — step 3 might say "so we subtract 12"
        # which only makes sense after reading steps 1 and 2.
        correct_context = question_prefix + "\n".join(steps[: i + 1])

        # For the corrupted version, I keep all the previous steps intact and
        # only corrupt step i. This way, the previous context is the same for
        # both the correct and corrupted row, and the PRM has to learn to detect
        # the error in the current step specifically.
        corrupted_context = question_prefix + "\n".join(
            steps[:i] + [corrupt_step(step)]
        )

        rows.append({"text": correct_context,   "label": 1.0})
        rows.append({"text": corrupted_context, "label": 0.0})
    return rows


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, CONFIG, split=SPLIT)
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            for row in build_prm_examples(example):
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"Saved {count} PRM step-level examples to {OUTPUT_PATH}")
