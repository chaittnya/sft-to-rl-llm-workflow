from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer


# Configuration
# We initialize PPO from the supervised fine-tuned checkpoint. This is a common
# approach in reward learning research, because it gives the policy a useful
# starting point before RL-based improvements are applied.
# REWARD_MODEL_PATH and VALUE_MODEL_PATH point at the checkpoints produced by
# train_reward_model.py and train_value_model.py. Run create_reward_data.py,
# train_reward_model.py, create_value_data.py and train_value_model.py before
# this script, otherwise these paths will not exist yet.
BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./reward_model"
VALUE_MODEL_PATH = "./value_model"
OUTPUT_DIR = "./ppo_model"
DATA_PATH = "./ppo_prompts.jsonl"

# Load the prompts produced by create_data.py.
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


# PPOTrainer expects a tokenized "input_ids" column rather than raw prompt text.
def tokenize(example):
    return {"input_ids": tokenizer(example["prompt"])["input_ids"]}


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Policy and reference policy are loaded from the same base checkpoint. The
# reference model is used by PPO to estimate the KL penalty.
# device_map="auto" loads each model straight onto the GPU (NVIDIA RTX 2050)
# if one is available, rather than loading four separate copies into system
# RAM first.
policy_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto"
)
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto"
)

# The reward model is loaded from the checkpoint trained by
# train_reward_model.py, so it has actually learned to score responses
# instead of giving out a random untrained score.
#
# The value model is loaded from the checkpoint trained by
# train_value_model.py. It was itself warm-started from the reward model and
# then fine-tuned on its own regression data, so it already has a reasonable
# sense of response quality before PPO starts updating it further as the
# critic.
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto"
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    VALUE_MODEL_PATH, num_labels=1, device_map="auto"
)

args = PPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    total_episodes=len(dataset),
    learning_rate=3e-6,
    report_to="none",
)

trainer = PPOTrainer(
    args=args,
    processing_class=tokenizer,
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
