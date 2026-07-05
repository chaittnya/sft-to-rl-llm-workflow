from datasets import load_dataset
import json
import re
import random


# Training data for the Process Reward Model used in PPO_PRM.
#
# Identical logic to GRPO_PRM/create_prm_data.py — see that file for a full
# explanation of what PRM is, why we use GSM8K, and how the step corruption
# works. I kept this file as a separate copy (rather than importing from
# GRPO_PRM) so each directory in this repo can run independently without
# needing files from other directories.
#
# One thing specific to the PPO use case: even though PPOTrainer cannot do
# per-step scoring at inference time (it calls the reward model on the full
# sequence), the PRM is still worth training on step-level data. The reason
# is that step-level training teaches the model to be sensitive to local
# errors in reasoning — a sensitivity it retains even when it later scores
# a full sequence. Think of it as giving the model a very detailed
# understanding of what "correct math steps" look like, which then informs
# its overall judgement of the full solution.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./prm_data.jsonl"

random.seed(42)


def extract_steps(answer_text):
    # GSM8K solutions end with "#### <number>". I drop that line because it is
    # just the final answer box, not a reasoning step I want the PRM to score.
    lines = [line.strip() for line in answer_text.split("\n") if line.strip()]
    return [line for line in lines if not line.startswith("####")]


def corrupt_step(step):
    # Nudge one number in the step by a small offset. The corrupted step still
    # looks like a real math step — the text structure is intact, just the
    # arithmetic is wrong. This forces the PRM to check the numbers rather than
    # relying on surface features like sentence length or word choice.
    numbers = re.findall(r'\d+', step)
    if not numbers:
        return step + " This is incorrect."
    target = random.choice(numbers)
    offset = random.choice([-3, -2, 2, 3, 5, 7])
    replacement = str(max(0, int(target) + offset))
    return step.replace(target, replacement, 1)


def build_prm_examples(example):
    question_prefix = (
        f"Q: {example['question']}\n"
        "A: Let me solve this step by step.\n"
    )
    steps = extract_steps(example["answer"])

    rows = []
    for i, step in enumerate(steps):
        # Build the growing context up to step i. The correct and corrupted
        # rows share the same history (steps 0 to i-1) and differ only at
        # step i, which isolates the error signal to the current step.
        correct_context = question_prefix + "\n".join(steps[: i + 1])
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
