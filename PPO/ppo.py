from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer


# Configuration
# We initialize PPO from the supervised fine-tuned checkpoint. This is a common
# approach in reward learning research, because it gives the policy a useful
# starting point before RL-based improvements are applied.
BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./ppo_model"

# Load a compact RL dataset for illustrative purposes.
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:200]")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Load the policy and reference policy from the same base checkpoint.
# The reference model is used by PPO to estimate the KL penalty.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto"
)

# Build a prompt for each example. This follows the same prompt structure as the
# initial SFT dataset, so the policy is not surprised by the input format.
def build_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )

prompts = [build_prompt(example) for example in dataset]

# Reward function used here is intentionally simplistic. In a genuine
# experiment you would replace this with a learned reward model or human
# preference signal.
def compute_reward(response_text):
    return min(len(response_text.split()) / 20.0, 1.0)

trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=prompts,
    output_dir=OUTPUT_DIR,
)

# Training loop hyperparameters.
PPO_BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
NUM_EPOCHS = 1

for epoch in range(NUM_EPOCHS):
    for start in range(0, len(prompts), PPO_BATCH_SIZE):
        batch_prompts = prompts[start : start + PPO_BATCH_SIZE]
        responses = []

        for query in batch_prompts:
            response = trainer.generate(
                query,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(response[0], skip_special_tokens=True)
            responses.append(text)

        rewards = [compute_reward(r) for r in responses]
        trainer.step(batch_prompts, responses, rewards)

trainer.save_model(OUTPUT_DIR)
