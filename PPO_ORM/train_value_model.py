from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This script fine-tunes the value model (PPO's critic) starting from the ORM checkpoint.
#
# Why load from the ORM instead of from the SFT base?
# The value model and the reward model have the same architecture
# (AutoModelForSequenceClassification with num_labels=1). The reward model
# is already trained to associate solution text with quality scores. If I
# initialise the value model from the raw SFT base, it starts with random
# scoring weights and needs to learn from scratch what a good solution looks
# like. If I initialise it from the ORM checkpoint, it starts with weights
# that already capture that intuition, and the fine-tuning just needs to
# adjust the output scale and bias to better match the value targets.
# In practice this leads to a more stable critic at the start of PPO training.
#
# I want to be clear about what "warm-starting" means here: I am loading the
# ORM's weights into the value model's architecture. Both are independent
# torch.nn.Module objects during PPO — the ORM is frozen and used as the
# reward signal, the value model is updated at every PPO step. They share no
# parameters after this point; the warm start just gives the value model a
# better initialisation.

ORM_MODEL_PATH = "./orm_model"
OUTPUT_DIR = "./value_model"
DATA_PATH = "./value_data.jsonl"


def tokenize(example, tokenizer):
    tokenized = tokenizer(example["text"], truncation=True, max_length=256)
    tokenized["labels"] = float(example["label"])
    return tokenized


if __name__ == "__main__":
    # Load tokenizer from the ORM directory so we get the same vocabulary and
    # special token settings that the ORM was trained with.
    tokenizer = AutoTokenizer.from_pretrained(ORM_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model weights from the ORM checkpoint. After this line the value
    # model has the ORM's weights, but it is a completely separate object that
    # will be updated independently during this fine-tuning run.
    model = AutoModelForSequenceClassification.from_pretrained(
        ORM_MODEL_PATH,
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
