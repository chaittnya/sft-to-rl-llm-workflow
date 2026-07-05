import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This trains the Process Reward Model on the step-level data from create_prm_data.py.
#
# The architecture is identical to the ORM: AutoModelForSequenceClassification
# with num_labels=1 and problem_type="regression". The difference is entirely
# in the training data. Each row in prm_data.jsonl is the reasoning context up
# to one step (question + all steps so far), labelled 1.0 if that last step is
# correct and 0.0 if it is corrupted. The model learns to be sensitive to
# step-level errors inside a growing reasoning chain.
#
# Why does the architecture stay the same even though PRM and ORM have such
# different objectives?
# Both models ultimately need to answer "how good is this text?" and return a
# scalar. The difference is in what "this text" contains: for ORM it is the
# full solution, for PRM it is the context up to one specific step. The scoring
# head itself does not need to know which one it is — the information is all
# in the input. So we can train both with exactly the same architecture and
# training loop, swapping only the dataset.
#
# How grpo_prm.py uses this model:
# During GRPO training, the reward function in grpo_prm.py splits each
# generated completion into steps, then calls this model once per step prefix,
# and averages all the step scores into a single reward value. A completion
# that gets the first three steps right and the fourth wrong will score around
# 0.75 rather than 0. This gradient is more informative than the hard 0/1
# signal ORM returns for the same solution.

BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./prm_model"
DATA_PATH = "./prm_data.jsonl"


def tokenize(example, tokenizer):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        # max_length=256 might truncate very long step contexts. In a real
        # experiment I would increase this, but 256 keeps training fast on
        # this GPU for a demonstration.
    )
    tokenized["labels"] = float(example["label"])
    return tokenized


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=1,
        problem_type="regression",
        device_map="auto",
    )

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        remove_unused_columns=True,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"PRM saved to {OUTPUT_DIR}")
