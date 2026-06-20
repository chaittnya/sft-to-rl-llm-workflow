from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ==========================================================
# VALUE MODEL TRAINING
# ==========================================================
# This trains the critic that PPO needs. The value model looks at a single
# response and predicts how good it is, as a plain number. PPO compares this
# prediction against the actual reward it observes to figure out the
# "advantage" of each action, which is what drives the policy update.
#
# We start from the reward model checkpoint rather than the plain SFT model.
# Both models share the same scalar-output architecture, so this gives the
# value head a reasonable starting point instead of random weights. We then
# fine-tune it here on its own regression data (see create_value_data.py) so
# it learns the value model's specific job before PPO starts using it.
REWARD_MODEL_PATH = "./reward_model"
OUTPUT_DIR = "./value_model"
DATA_PATH = "./value_data.jsonl"

# Load the regression examples produced by create_value_data.py.
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token


# Turn the raw text into token ids, and rename "label" to "labels" because
# that is the field name AutoModelForSequenceClassification looks for when
# it computes the loss.
def tokenize(example):
    tokenized = tokenizer(example["text"], truncation=True, max_length=256)
    tokenized["labels"] = float(example["label"])
    return tokenized


dataset = dataset.map(tokenize, remove_columns=["text", "label"])

# num_labels=1 plus problem_type="regression" tells the model to predict a
# single continuous number and train with mean squared error, instead of
# treating this as a classification problem. device_map="auto" loads it
# straight onto the GPU (NVIDIA RTX 2050) if one is available.
model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, problem_type="regression", device_map="auto"
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Possible values: 1 for a quick pass, 2-3 if you want a closer fit on
    # this small dataset.
    num_train_epochs=1,

    per_device_train_batch_size=4,

    # Common values: 1e-5 to 1e-4 for this kind of small fine-tune.
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
