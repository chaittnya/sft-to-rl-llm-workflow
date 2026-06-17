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


'''
AI Generated Explanation:

args = TrainingArguments(

    # ==========================================================
    # OUTPUT / CHECKPOINT DIRECTORY
    # ==========================================================

    # Directory where everything is saved:
    # - checkpoints
    # - logs
    # - trainer state
    # - final model
    #
    # Possible values:
    # Any valid folder path
    #
    # Examples:
    # "./sft_model"
    # "./outputs/run1"
    # "/content/checkpoints"
    output_dir="./sft_model",

    # ==========================================================
    # TRAINING DURATION
    # ==========================================================

    # Number of complete passes through the training dataset.
    #
    # Example:
    # Dataset = 1000 samples
    # Epochs = 3
    #
    # Model sees:
    # 3000 samples total
    #
    # Common values:
    # 1      -> quick experiment
    # 2-5    -> standard SFT
    # 10+    -> often overfitting
    #
    # Alternative:
    # max_steps=1000
    #
    # If max_steps is specified,
    # training stops after exactly that many updates.
    num_train_epochs=3,

    # ==========================================================
    # BATCH SIZE
    # ==========================================================

    # Number of samples processed on each device
    # before one forward/backward pass.
    #
    # Larger batch:
    # + smoother gradients
    # - more VRAM
    #
    # Smaller batch:
    # + less VRAM
    # - noisier gradients
    #
    # Common values:
    # 1,2,4,8,16,32
    per_device_train_batch_size=4,

    # Evaluation batch size.
    #
    # Since no gradients are computed,
    # this can often be larger than training batch size.
    #
    # Common values:
    # 4,8,16,32,64
    per_device_eval_batch_size=8,

    # ==========================================================
    # GRADIENT ACCUMULATION
    # ==========================================================

    # Number of batches whose gradients are accumulated
    # before performing an optimizer step.
    #
    # Effective Batch Size:
    #
    # effective_batch_size =
    # per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    #
    # Example:
    #
    # batch_size = 4
    # grad_accum = 4
    #
    # effective_batch_size = 16
    #
    # Useful when GPU memory is limited.
    #
    # Common values:
    # 1   -> no accumulation
    # 2
    # 4
    # 8
    # 16
    gradient_accumulation_steps=4,

    # ==========================================================
    # LEARNING RATE
    # ==========================================================

    # Step size used during gradient descent.
    #
    # Update rule:
    #
    # theta = theta - lr * gradient
    #
    # Too large:
    # - unstable training
    # - catastrophic forgetting
    #
    # Too small:
    # - slow convergence
    #
    # Common SFT values:
    # 1e-6
    # 5e-6
    # 1e-5
    # 2e-5
    # 5e-5
    # 1e-4
    learning_rate=2e-5,

    # ==========================================================
    # LEARNING RATE WARMUP
    # ==========================================================

    # Fraction of training used for warmup.
    #
    # Learning rate gradually increases:
    #
    # 0 --> learning_rate
    #
    # during first warmup_ratio fraction
    # of training.
    #
    # Helps prevent unstable updates
    # at beginning of training.
    #
    # Common values:
    # 0.0
    # 0.01
    # 0.03
    # 0.05
    # 0.1
    warmup_ratio=0.05,

    # ==========================================================
    # LR SCHEDULER
    # ==========================================================

    # Determines how learning rate changes
    # during training.
    #
    # Possible options:
    #
    # "constant"
    #   LR never changes.
    #
    # "linear"
    #   LR linearly decays to zero.
    #
    # "cosine"
    #   Cosine decay.
    #   Very popular for LLM training.
    #
    # "cosine_with_restarts"
    #   Cosine decay with periodic resets.
    #
    # "polynomial"
    #   Polynomial decay.
    #
    # "constant_with_warmup"
    #   Warmup then constant LR.
    #
    # "inverse_sqrt"
    #   Used in some transformer training setups.
    #
    # Most common:
    # "linear"
    # "cosine"
    lr_scheduler_type="cosine",

    # ==========================================================
    # REGULARIZATION
    # ==========================================================

    # L2 regularization coefficient.
    #
    # Loss becomes:
    #
    # total_loss =
    # task_loss + λ||θ||²
    #
    # Helps reduce overfitting.
    #
    # Common values:
    # 0.0
    # 0.001
    # 0.01
    # 0.1
    #
    # Most common for transformers:
    # 0.01
    weight_decay=0.01,

    # ==========================================================
    # EVALUATION STRATEGY
    # ==========================================================

    # Controls when validation runs.
    #
    # Possible options:
    #
    # "no"
    #   Never evaluate.
    #
    # "steps"
    #   Evaluate every eval_steps.
    #
    # "epoch"
    #   Evaluate after each epoch.
    #
    # Common:
    # "steps" for large datasets.
    # "epoch" for small datasets.
    eval_strategy="steps",

    # Number of update steps between evaluations.
    #
    # Only used when:
    # eval_strategy="steps"
    #
    # Common values:
    # 50
    # 100
    # 500
    # 1000
    eval_steps=50,

    # ==========================================================
    # CHECKPOINT SAVING
    # ==========================================================

    # Number of update steps between checkpoint saves.
    #
    # Only relevant when save_strategy="steps"
    #
    # Common values:
    # 50
    # 100
    # 500
    # 1000
    save_steps=50,

    # Maximum number of checkpoints retained.
    #
    # Example:
    #
    # checkpoint-100
    # checkpoint-200
    # checkpoint-300
    #
    # save_total_limit=2
    #
    # checkpoint-100 deleted automatically.
    #
    # Common values:
    # 1
    # 2
    # 3
    # 5
    save_total_limit=2,

    # ==========================================================
    # BEST MODEL SELECTION
    # ==========================================================

    # At training end:
    #
    # False:
    #   Return final checkpoint.
    #
    # True:
    #   Return best checkpoint according
    #   to metric_for_best_model.
    #
    # Usually recommended.
    load_best_model_at_end=True,

    # Metric used to determine best checkpoint.
    #
    # Common options:
    #
    # "eval_loss"
    #
    # Custom metrics:
    # "accuracy"
    # "f1"
    # "bleu"
    # "rouge"
    #
    # Must exist in evaluation output.
    metric_for_best_model="eval_loss",

    # Determines whether larger or smaller metric values
    # are considered better.
    #
    # False:
    # Smaller is better.
    #
    # Examples:
    # eval_loss
    # perplexity
    #
    # True:
    # Larger is better.
    #
    # Examples:
    # accuracy
    # f1
    # rouge
    # bleu
    greater_is_better=False,

    # ==========================================================
    # MIXED PRECISION
    # ==========================================================

    # Use FP16 training.
    #
    # Benefits:
    # - Faster training
    # - Less VRAM
    #
    # Options:
    #
    # fp16=True
    #   Use float16
    #
    # fp16=False
    #   Disable float16
    #
    # Alternative:
    #
    # bf16=True
    #   Use bfloat16
    #
    # BF16 is usually preferred on:
    # A100
    # H100
    # RTX 4090
    fp16=True,

    # ==========================================================
    # LOGGING BACKEND
    # ==========================================================

    # Controls where metrics are reported.
    #
    # Possible values:
    #
    # "none"
    #   No reporting.
    #
    # "tensorboard"
    #   TensorBoard logs.
    #
    # "wandb"
    #   Weights & Biases.
    #
    # "mlflow"
    #   MLflow tracking.
    #
    # "comet_ml"
    #   Comet logging.
    #
    # Multiple values possible:
    #
    # report_to=["tensorboard","wandb"]
    #
    # Very common:
    # "none"
    # "wandb"
    report_to="none"
)
'''