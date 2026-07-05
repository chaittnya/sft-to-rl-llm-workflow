from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This fine-tunes the value model starting from the PRM checkpoint.
#
# The reasoning is the same as PPO_ORM/train_value_model.py: warm-starting
# from a trained reward model gives the value model (PPO's critic) a much
# better initialisation than starting from the raw SFT base.
#
# For PPO_PRM the warm-start point is the PRM rather than the ORM. This is
# actually a better fit for the value function in a process-reward setting.
# The PRM was trained to judge step-level quality, which is close to what
# the value function is supposed to do: at any point in the generation, the
# value model should estimate how much reward the remaining steps will
# contribute. Starting from a model that already understands step quality
# makes that easier than starting from an ORM that only understands final
# outcomes.
#
# The training data (value_data.jsonl) uses full-solution binary labels, not
# step-level labels. This is a simplification — ideally the value model would
# be trained with per-step value targets. But for the purposes of this
# demonstration, binary full-solution supervision is enough to get a critic
# that is somewhat calibrated before PPO begins updating it.

PRM_MODEL_PATH = "./prm_model"
OUTPUT_DIR = "./value_model"
DATA_PATH = "./value_data.jsonl"


def tokenize(example, tokenizer):
    tokenized = tokenizer(example["text"], truncation=True, max_length=256)
    tokenized["labels"] = float(example["label"])
    return tokenized


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load from the PRM checkpoint. The value model inherits the PRM's
    # step-level representations and then is adjusted by the regression
    # training below to predict full-solution expected rewards.
    model = AutoModelForSequenceClassification.from_pretrained(
        PRM_MODEL_PATH,
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
