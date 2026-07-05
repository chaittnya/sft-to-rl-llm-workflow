from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This fine-tunes the value model (PPO's critic) starting from the k-wise reward
# model checkpoint.
#
# The value model and the k-wise reward model share the same architecture:
# AutoModelForSequenceClassification with num_labels=1 and problem_type="regression".
# The key difference is their role during PPO:
#
#   k-wise reward model — frozen during PPO, provides the ground-truth reward
#     signal for each completed response.
#
#   value model — updated every PPO step, predicts the expected future reward
#     from the current state. PPO computes (actual_reward - value_estimate) as
#     the advantage, so the critic needs to track the evolving policy.
#
# Why warm-start the value model from the k-wise reward checkpoint?
# The PL model already learned to distinguish good from bad responses, even
# though it was trained with a ranking loss rather than a regression loss. Those
# representations transfer well to the critic's job. Starting from a random
# initialisation or from the raw SFT base would take many more PPO steps for
# the critic to become useful, which destabilises early training.
#
# This fine-tuning step adapts the PL model to single-text regression. The
# value_data.jsonl labels (1.0 for good, 0.0 for bad) act as calibration
# targets: they pull the model's scoring range toward [0, 1] and anchor it to
# a sensible baseline before PPO starts shifting it further.

KWISE_REWARD_MODEL_PATH = "./kwise_reward_model"
OUTPUT_DIR = "./value_model"
DATA_PATH = "./value_data.jsonl"


def tokenize(example, tokenizer):
    tokenized = tokenizer(example["text"], truncation=True, max_length=256)
    # "labels" (plural) is the key Trainer uses for the regression target.
    # It must be a float for MSELoss to apply; an int here would cause a
    # dtype mismatch error.
    tokenized["labels"] = float(example["label"])
    return tokenized


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(KWISE_REWARD_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        KWISE_REWARD_MODEL_PATH,
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
    print(f"Value model saved to {OUTPUT_DIR}")
