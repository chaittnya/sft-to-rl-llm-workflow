from datasets import load_dataset
import json


# Training data for the Outcome Reward Model used in PPO_ORM.
#
# Identical logic to GRPO_ORM/create_orm_data.py — the ORM architecture and
# training objective do not change depending on whether the model ends up
# plugged into GRPO or PPO. The difference is in how the trained checkpoint
# gets used:
#
# In GRPO_ORM, the ORM is wrapped in a Python reward function that GRPO calls
# directly. In PPO_ORM, the ORM is passed to PPOTrainer as a torch.nn.Module
# and PPO calls its forward method internally. Either way, the model receives
# (question + solution) text and returns a scalar score.
#
# This checkpoint also warm-starts the value model. PPO needs two separate
# scorer models — a frozen reward model and a trainable critic (value model).
# Loading the value model from the ORM checkpoint rather than from the raw SFT
# base gives it a useful starting point: it already knows what a correct math
# solution looks like before its own regression fine-tuning begins.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./orm_data.jsonl"


def build_orm_examples(examples_list):
    rows = []
    n = len(examples_list)
    for i, example in enumerate(examples_list):
        prefix = (
            f"Q: {example['question']}\n"
            "A: Let me solve this step by step.\n"
        )
        correct_solution = example["answer"]
        # Circular shift: use the next example's full solution as the wrong
        # answer. It is grammatically correct and well-structured, so the ORM
        # cannot cheat by detecting bad text — it must learn the math.
        wrong_solution = examples_list[(i + 1) % n]["answer"]

        rows.append({"text": prefix + correct_solution, "label": 1.0})
        rows.append({"text": prefix + wrong_solution,   "label": 0.0})
    return rows


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, CONFIG, split=SPLIT)
    examples_list = list(dataset)
    rows = build_orm_examples(examples_list)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} ORM training examples to {OUTPUT_PATH}")
