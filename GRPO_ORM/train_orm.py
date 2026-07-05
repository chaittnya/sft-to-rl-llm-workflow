import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This script trains the Outcome Reward Model on the data from create_orm_data.py.
#
# Architecture choice:
# I use AutoModelForSequenceClassification with num_labels=1. This adds a
# single linear "scoring head" on top of the base transformer. The head takes
# the last hidden state from the [CLS]-equivalent token and projects it down
# to one number. That number is the model's score for the input text.
#
# Why num_labels=1 and not num_labels=2?
# With num_labels=2 the model would do binary classification (correct vs
# incorrect) and use CrossEntropyLoss. That would work, but I wanted a scalar
# score in [0, 1] so the reward function in grpo_orm.py can return a
# continuous value instead of a hard 0 or 1. Using num_labels=1 with
# problem_type="regression" gets exactly that: the Trainer uses MSELoss, and
# the output is a single unconstrained float that we treat as a quality score.
# Values close to 1.0 mean "looks correct", values close to 0.0 mean "looks wrong".
#
# The problem_type parameter is easy to miss — without it, HuggingFace will
# try to infer the loss type from the label dtype and shape, and it sometimes
# gets it wrong. Explicitly setting it to "regression" ensures MSELoss is used.

BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./orm_model"
DATA_PATH = "./orm_data.jsonl"


def tokenize(example, tokenizer):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
    )
    # Trainer looks for a key called "labels" (plural) in each batch. The value
    # must be a float (not an int) for MSELoss to work correctly. If you pass
    # an int here, PyTorch will complain about a type mismatch between the model
    # output (float32) and the target (int64).
    tokenized["labels"] = float(example["label"])
    return tokenized


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # SmolLM2 does not set a pad token by default. DataCollatorWithPadding
    # needs one to know which token to use when padding shorter sequences in
    # a batch to the same length. Using eos_token as the pad token is the
    # standard workaround for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=1,
        problem_type="regression",
        device_map="auto",
    )

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    # remove_unused_columns=True (the default) inside Trainer would drop any
    # column not in the model's forward() signature. The dataset has a "text"
    # column that we no longer need after tokenisation, but the map call below
    # already handles that because we only keep the tokenizer's output keys and
    # "labels". The remove_unused_columns default is fine here.
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
        # DataCollatorWithPadding pads each batch's input_ids and
        # attention_mask to the length of the longest sequence in that batch,
        # which is much more efficient than padding everything to the global
        # max_length during tokenisation.
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    # save_model() saves the model weights but not the tokenizer. Saving the
    # tokenizer separately means we can load the ORM checkpoint later with
    # just its own directory path, without needing the original SFT path.
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"ORM saved to {OUTPUT_DIR}")
