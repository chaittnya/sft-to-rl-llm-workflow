from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# DATASET
dataset = load_dataset("yahma/alpaca-cleaned")

train_dataset = dataset["train"].select(range(1000))
val_dataset = dataset["train"].select(range(1000,1200))

def format_example(example):
    text = f"""
### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}
"""
    return {"text": text}

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)


# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# MODEL
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# LORA CONFIG
peft_config = LoraConfig(

    # LoRA rank: the bottleneck dimension of the low-rank update.
    # This controls how much extra capacity LoRA gets.
    #
    # Formula:
    #   ΔW = B @ A
    # where A has shape (r, in_features) and B has shape (out_features, r)
    #
    # Common options:
    #   r=4   -> very lightweight, lowest memory
    #   r=8   -> small, often enough for simple adaptation
    #   r=16  -> common default for SFT
    #   r=32  -> higher capacity, more VRAM
    #   r=64  -> strong capacity, but heavier
    r=16,

    # LoRA scaling factor.
    # The effective LoRA contribution is usually scaled by alpha / r.
    #
    # Intuition:
    #   larger alpha => stronger effect of LoRA updates
    #   smaller alpha => weaker effect
    #
    # Common options:
    #   8, 16, 32, 64, 128
    #
    # Common heuristic:
    #   alpha = 2 * r
    # Example:
    #   r=16 -> alpha=32
    lora_alpha=32,

    # Dropout applied on the LoRA branch during training.
    # Helps reduce overfitting, especially on small datasets.
    #
    # Common options:
    #   0.0  -> no dropout, maximum capacity
    #   0.05 -> mild regularization, very common
    #   0.1  -> stronger regularization
    #
    # Typical use:
    #   small dataset -> 0.05 or 0.1
    #   large dataset -> 0.0 or 0.05
    lora_dropout=0.05,

    # Whether biases are trainable.
    #
    # Options:
    #   "none"      -> do not train any bias terms
    #   "all"       -> train all bias parameters in the model
    #   "lora_only" -> train only bias terms associated with LoRA-adapted layers
    #
    # Most common choice:
    #   "none"
    #
    # Why:
    #   keeps training parameter count low and is usually enough for SFT
    bias="none",

    # Task type tells PEFT what kind of model/training setup this is.
    #
    # Common options:
    #   "CAUSAL_LM"      -> decoder-only language models like Mistral, Llama, SmolLM
    #   "SEQ_2_SEQ_LM"   -> encoder-decoder models like T5, FLAN-T5, BART
    #   "SEQ_CLS"        -> sequence classification
    #   "TOKEN_CLS"      -> token classification / NER
    #
    # For instruction fine-tuning of an LLM, use:
    #   "CAUSAL_LM"
    task_type="CAUSAL_LM",

    # Which internal linear modules should receive LoRA adapters.
    #
    # In transformer attention, these usually correspond to:
    #   q_proj -> query projection
    #   k_proj -> key projection
    #   v_proj -> value projection
    #   o_proj -> output projection
    #
    # Common choices:
    #   ["q_proj", "v_proj"]
    #       -> cheaper, often enough for many tasks
    #
    #   ["q_proj", "k_proj", "v_proj", "o_proj"]
    #       -> standard/common, more expressive
    #
    #   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #       -> broader coverage, higher capacity, more memory
    #
    # Note:
    #   exact module names depend on the model architecture
    #   some models may use different names
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)


# TRAINING ARGUMENTS


args = TrainingArguments(

    output_dir="./smollm_lora",

    num_train_epochs=3,

    learning_rate=2e-4,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    eval_strategy="steps",

    eval_steps=50,

    save_steps=50,

    logging_steps=10,

    save_total_limit=2,

    fp16=True,

    report_to="none"
)


# TRAINER


trainer = SFTTrainer(

    model=model,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    peft_config=peft_config,

    args=args
)

trainer.train()

trainer.save_model("./final_lora")