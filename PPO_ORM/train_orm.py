import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This trains the Outcome Reward Model that PPO_ORM will use as its reward signal.
#
# The architecture and training procedure are identical to GRPO_ORM/train_orm.py.
# AutoModelForSequenceClassification with num_labels=1 and problem_type="regression"
# gives a single scalar output per input text, trained with MSELoss on binary 0/1
# targets. See GRPO_ORM/train_orm.py for a detailed explanation of why we use
# regression instead of binary classification here.
#
# This checkpoint has two jobs in the PPO pipeline:
#
# First, it becomes the frozen reward_model that PPOTrainer uses to score each
# generated completion during rollouts. PPOTrainer calls reward_model.forward()
# internally, so from PPO's perspective the ORM is just a torch.nn.Module that
# takes token IDs and returns a logit. ORM fits this API perfectly because it
# already produces a scalar score for any input sequence.
#
# Second, it is the starting point for train_value_model.py. The value model
# (the PPO critic) and the reward model have the same architecture. Instead of
# initialising the value model from scratch or from the raw SFT base, I load
# the ORM checkpoint first. This gives the value model a head start: it already
# has weights that encode "what does a correct math solution look like", which
# is exactly what the critic needs to predict future rewards from.

BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./orm_model"
DATA_PATH = "./orm_data.jsonl"


def tokenize(example, tokenizer):
    tokenized = tokenizer(example["text"], truncation=True, max_length=256)
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
    print(f"ORM saved to {OUTPUT_DIR}")
