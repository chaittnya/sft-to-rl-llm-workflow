from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# Dataset loading
# The cleaned Alpaca dataset is used here because it is a standard benchmark
# for instruction-following models. For this demonstration, we intentionally
# limit the sample size to keep the workflow lightweight.

dataset = load_dataset("yahma/alpaca-cleaned")
train_dataset = dataset["train"].select(range(1000))
val_dataset = dataset["train"].select(range(1000, 1200))


# Tokenizer setup
# The tokenizer must be loaded from the same base model to ensure that token
# IDs are consistent. We map the pad token to the EOS token because this is a
# standard choice for causal language model fine-tuning.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# Example formatting
# Convert each dataset example into a single prompt string using an instruction-
# input-response structure. This mirrors the common SFT format used in recent
# research experiments.
def format_example(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    return {"text": text}

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)


# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)


# Training arguments
args = TrainingArguments(

    # Directory where checkpoints, logs, and the final model will be written.
    # Possible values: any valid folder path.
    # Example: "./sft_model", "/content/output", "runs/exp1"
    output_dir="./sft_model",

    # Number of complete passes over the training dataset.
    # Possible values:
    # - 1: one full pass, useful for quick experiments
    # - 2, 3, 5: common for fine-tuning
    # - larger values: more training, but higher risk of overfitting
    # Alternative to epochs:
    # - max_steps: train for a fixed number of optimizer updates instead
    num_train_epochs=3,

    # Number of training samples processed on each GPU/device before one backward pass.
    # Smaller values use less VRAM, larger values are more stable but need more memory.
    # Common values: 1, 2, 4, 8, 16
    per_device_train_batch_size=4,

    # Number of samples processed per GPU/device during evaluation.
    # Since no gradients are computed in eval, this can often be larger than train batch size.
    # Common values: 4, 8, 16, 32
    per_device_eval_batch_size=4,

    # Step size for parameter updates.
    # Controls how aggressively the model weights move in the direction of the gradient.
    # Common values in SFT:
    # - 1e-6, 5e-6: very small, cautious updates
    # - 1e-5, 2e-5: common starting points
    # - 5e-5, 1e-4: faster learning, but can destabilize training
    learning_rate=2e-5,

    # L2 regularization on weights.
    # Helps reduce overfitting by discouraging overly large parameter values.
    # Common values:
    # - 0.0: no weight decay
    # - 0.01: very common for transformer fine-tuning
    # - 0.1: stronger regularization
    weight_decay=0.01,

    # How often to print training logs.
    # Measured in optimizer update steps.
    # Common values: 1, 10, 50, 100
    # Smaller values give more frequent feedback, but noisier logs.
    logging_steps=10,

    # When to run evaluation on the validation set.
    # Possible values:
    # - "no": never evaluate during training
    # - "steps": evaluate every eval_steps updates
    # - "epoch": evaluate once after each epoch
    # "steps" is useful for long runs; "epoch" is simpler for small datasets.
    eval_strategy="steps",

    # Number of update steps between evaluations.
    # Only used when eval_strategy="steps".
    # Common values: 50, 100, 500, 1000
    eval_steps=50,

    # Number of update steps between saving checkpoints.
    # Possible values depend on save_strategy.
    # Usually combined with save_strategy="steps" or "epoch".
    save_steps=50,

    # Maximum number of checkpoints to keep on disk.
    # Older checkpoints are deleted once this limit is exceeded.
    # Common values:
    # - 1: keep only latest/best checkpoint
    # - 2 or 3: keep a few recent ones
    # Useful when disk space is limited.
    save_total_limit=2,

    # Use FP16 mixed precision training.
    # Benefits:
    # - lower VRAM usage
    # - faster training on supported NVIDIA GPUs
    # Possible values:
    # - True: enable 16-bit floating point training
    # - False: disable it and use full precision
    # Alternative:
    # - bf16=True for BF16-capable hardware like A100/H100
    fp16=True,

    # Where to send training metrics.
    # Possible values:
    # - "none": disable reporting
    # - "tensorboard": write logs for TensorBoard
    # - "wandb": send logs to Weights & Biases
    # - "mlflow": send logs to MLflow
    # - list of reporters in some setups, e.g. ["tensorboard", "wandb"]
    report_to="none"
)

trainer = SFTTrainer(
    model=model,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    args=args
)

trainer.train()

trainer.save_model("./final_model")