from datasets import load_dataset
import json


# Prompt dataset for PPO with a Process Reward Model.
#
# Same as PPO_ORM/create_data.py and both GRPO variants — GSM8K math problems
# formatted with numbered steps. Keeping the prompt format identical across
# all four directories (GRPO_ORM, GRPO_PRM, PPO_ORM, PPO_PRM) means any
# difference in training outcomes comes from the reward model type, not from
# the questions the policy was trained on.
#
# One thing worth noting: ppo_prm.py cannot do per-step PRM scoring inside
# PPOTrainer's reward call (PPO's API only accepts a torch Module, not a
# Python function). But we still use the numbered-step format here so the
# prompts stay consistent with GRPO_PRM. The PRM model itself was trained on
# step-level data so it still produces more meaningful scores on step-structured
# text even when it scores the full sequence at once.

DATA_SOURCE = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train[:200]"
OUTPUT_PATH = "./ppo_prompts.jsonl"


def build_prompt(example):
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
