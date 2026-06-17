from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer


# Configuration
# This script initializes DPO from the SFT checkpoint, which is a sensible
# choice for subsequent preference optimization. The model already knows how
# to follow instructions, and DPO refines the preference-shaped scoring.
BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./dpo_model"

# Use a small subset of the Alpaca dataset for brevity.
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:200]")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto"
)

# DPO operates on pairs of responses: one preferred and one dispreferred.
# In a real experiment these would come from human annotation.
def make_pair(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:"
    )
    preferred = f"{prompt} This response is more helpful and complete."
    dispreferred = f"{prompt} This response is shorter and less detailed."
    return {
        "query": prompt,
        "preferred": preferred,
        "dispreferred": dispreferred,
    }

pairs = [make_pair(ex) for ex in dataset]

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    output_dir=OUTPUT_DIR,
)

# Training loop with a simple batch-based update scheme. In practice, the batch
# size and number of epochs would be tuned according to validation preference
# accuracy or reward ranking performance.
BATCH_SIZE = 4
NUM_EPOCHS = 1

for epoch in range(NUM_EPOCHS):
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]
        queries = [item["query"] for item in batch]
        preferred_responses = [item["preferred"] for item in batch]
        dispreferred_responses = [item["dispreferred"] for item in batch]

        trainer.step(
            queries=queries,
            preferred_responses=preferred_responses,
            dispreferred_responses=dispreferred_responses,
        )

trainer.save_model(OUTPUT_DIR)
