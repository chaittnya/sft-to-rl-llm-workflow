from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer


# Configuration
# The GRPO script uses the SFT checkpoint as the initial policy. This setup is
# aligned with the idea of bootstrapping reinforcement learning from supervised
# pretraining, which is a common strategy in modern language model research.
BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./grpo_model"

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:200]")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto"
)

# Define a toy reward metric. This is purely illustrative; a real reward model
# would be learned separately and would supply scores for candidate outputs.
def compute_reward(response_text):
    return min(len(response_text.split()) / 20.0, 1.0)

# Build prompts in the same structured format as the SFT dataset.
def build_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )

prompts = [build_prompt(example) for example in dataset]

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    output_dir=OUTPUT_DIR,
)

# Training settings below are intentionally modest for demonstration. A real
# research experiment would tune batch size, generation length, and learning
# rate on a validation objective.
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
NUM_EPOCHS = 1

for epoch in range(NUM_EPOCHS):
    for start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[start : start + BATCH_SIZE]
        responses = []

        for prompt in batch_prompts:
            output_ids = trainer.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            responses.append(text)

        rewards = [compute_reward(r) for r in responses]
        trainer.step(batch_prompts, responses, rewards)

trainer.save_model(OUTPUT_DIR)
