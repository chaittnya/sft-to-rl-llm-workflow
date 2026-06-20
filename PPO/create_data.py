from datasets import load_dataset
import json


# ==========================================================
# DATA CREATION FOR PPO
# ==========================================================
# This script builds a small file of prompts that the PPO script (ppo.py) can
# load later. We are not training anything here, we are just preparing the
# input data. Think of this as "step 0" before the actual RL training starts.


# Which dataset to pull from the Hugging Face Hub.
# Possible values:
# - any dataset repo id on the hub, e.g. "yahma/alpaca-cleaned"
# - a local path if you have your own dataset saved on disk
# We use "yahma/alpaca-cleaned" because it is the same dataset used for SFT,
# so the policy model already knows how to handle this style of prompt.
DATA_SOURCE = "yahma/alpaca-cleaned"

# Which part of the dataset to load, and how much of it.
# Possible values:
# - "train" loads the whole training split
# - "train[:200]" loads only the first 200 examples (what we use here)
# - "train[:10%]" loads the first 10% of examples
# - "train[200:400]" loads a slice from index 200 to 400
# - "test" or "validation" for other splits, if the dataset has them
# We only take a small slice (200 rows) because this is meant to be a quick,
# lightweight example, not a full production run.
SPLIT = "train[:200]"

# Where the generated prompts will be written.
# Possible values:
# - any valid file path ending in ".jsonl" (one JSON object per line)
# - could also be ".json" if you wanted one big list instead, but then the
# writing code below would need to change
OUTPUT_PATH = "./ppo_prompts.jsonl"


# Turn one row of the raw dataset into the prompt format PPO expects.
# We reuse the same "### Instruction / ### Input / ### Response" layout that
# was used for SFT, so the model is not confused by a brand-new prompt style.
# The key must be called "prompt" because that is the column name trl's
# PPOTrainer looks for when it reads the dataset.
def build_prompt(example):
    return {
        "prompt": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            "### Response:"
        )
    }


if __name__ == "__main__":

    # Downloads (or loads from cache) the dataset slice described above.
    dataset = load_dataset(DATA_SOURCE, split=SPLIT)

    # Open the output file for writing.
    # Mode options for open():
    # "w" overwrites the file if it already exists (what we use here)
    # "a" appends to the end instead of overwriting
    # "x" creates a new file, but fails if it already exists
    # encoding="utf-8" is used so that any non-English characters in the
    # dataset (accents, other alphabets, etc.) are saved correctly.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
        for example in dataset:
            prompt_entry = build_prompt(example)

            # json.dumps turns our Python dictionary into a JSON string.
            # ensure_ascii option:
            # True escapes non-English characters into \uXXXX codes
            # False keeps non-English characters as-is, which is easier to
            # read if you open the file yourself, and why we use False here
            # We add "\n" after each entry so every line in the file is its
            # own separate JSON object (this is what ".jsonl" means).
            writer.write(json.dumps(prompt_entry, ensure_ascii=False) + "\n")

    # Quick sanity check printed to the console so we know the script worked
    # and how many prompts ended up in the file.
    print(f"Saved {len(dataset)} PPO prompts to {OUTPUT_PATH}")
