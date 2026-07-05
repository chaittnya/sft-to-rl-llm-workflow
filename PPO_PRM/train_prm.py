import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# This trains the Process Reward Model for the PPO_PRM pipeline.
#
# The training code is identical to GRPO_PRM/train_prm.py. I want to call out
# one thing that is specific to the PPO use case, because it is a bit subtle.
#
# In GRPO_PRM, the reward function is a Python function that we write ourselves.
# We can call the PRM however we like — split the completion into steps, run
# the model N times, average the scores. That is the "correct" way to use a PRM.
#
# In PPO (trl's experimental PPOTrainer), the reward model must be a
# torch.nn.Module passed to the trainer constructor. PPOTrainer calls it
# internally on the full (prompt + response) sequence, with no way to inject
# the step-splitting loop. So for PPO, the PRM ends up scoring the full
# sequence, not individual steps.
#
# This feels like a limitation, but it is not as bad as it sounds. The PRM was
# trained on growing step-by-step contexts, so its representations are tuned
# to judge reasoning quality step by step. When it sees a full solution (which
# is really just all steps concatenated), it draws on those same representations.
# It is weaker than per-step scoring, but it is still better than using an ORM
# that was never exposed to step-level reasoning signals during training.
#
# The checkpoint produced here also warm-starts train_value_model.py, following
# the same logic as PPO_ORM: the value model inherits the PRM's understanding
# of what correct reasoning looks like before its own fine-tuning.

BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./prm_model"
DATA_PATH = "./prm_data.jsonl"


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
    print(f"PRM saved to {OUTPUT_DIR}")
