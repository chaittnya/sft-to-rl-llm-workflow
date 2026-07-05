from datasets import load_dataset
import json


# Prompt dataset for PPO with a Plackett-Luce k-wise reward model.
#
# Same Alpaca instruction-following task as PPO/ and GRPO_KWISE/. The
# Plackett-Luce reward model was trained on Alpaca data (4 ranked completions
# per prompt), so the policy should also be trained and evaluated on the same
# kind of prompts.
#
# The output file is named ppo_prompts.jsonl to stay consistent with PPO/
# conventions. ppo_kwise.py will tokenise these prompts before passing them
# to PPOTrainer, because PPOTrainer expects an "input_ids" column rather than
# raw text.

DATA_SOURCE = "yahma/alpaca-cleaned"
SPLIT = "train[:200]"
OUTPUT_PATH = "./ppo_prompts.jsonl"


def build_prompt(example):
    # Same Alpaca prompt format used in PPO/ and GRPO_KWISE/.
    # "prompt" is the column name PPOTrainer's tokenisation step looks for.
    return {
        "prompt": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            "### Response:"
        )
    }


if __name__ == "__main__":
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            item = build_prompt(example)
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} prompts to {OUTPUT_PATH}")
