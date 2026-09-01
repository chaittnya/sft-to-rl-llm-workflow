from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


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
