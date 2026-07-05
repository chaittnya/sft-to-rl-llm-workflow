import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Import order matters here: importing GRPOConfig before GRPOTrainer segfaults
# on this machine's torch/trl combination. Keep Trainer before Config.
from trl import GRPOTrainer, GRPOConfig


# Configuration
# Same setup as GRPO/grpo.py, but the reward model checkpoint is the one
# trained with the Plackett-Luce k-wise ranking loss
# (train_kwise_reward_model.py) instead of the standard pairwise
# Bradley-Terry loss (GRPO/train_reward_model.py).
# At inference time both models are used identically:
#   score(prompt + completion) -> scalar
# Only the training objective differs.
BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./kwise_reward_model"
OUTPUT_DIR = "./grpo_kwise_model"
DATA_PATH = "./grpo_items.jsonl"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Load the k-wise reward model once so we are not reloading it on every call.
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto"
)
reward_model.eval()


def reward_func(prompts, completions, **kwargs):
    # Score each (prompt + completion) pair with the k-wise reward model.
    # The model was trained to give higher scores to better completions, so
    # we can use its raw logit as the reward signal directly.
    texts = [p + c for p, c in zip(prompts, completions)]
    inputs = reward_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(reward_model.device)

    with torch.no_grad():
        scores = reward_model(**inputs).logits.squeeze(-1)

    return scores.tolist()


args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    num_generations=4,
    learning_rate=1e-5,
    logging_steps=10,
    report_to="none",
    model_init_kwargs={"device_map": "auto"},
)

trainer = GRPOTrainer(
    model=BASE_MODEL_PATH,
    reward_funcs=reward_func,
    args=args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
