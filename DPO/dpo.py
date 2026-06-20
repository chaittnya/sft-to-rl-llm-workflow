from datasets import load_dataset
from trl import DPOConfig, DPOTrainer


# Configuration
# We initialize DPO from the supervised fine-tuned checkpoint. This is a
# sensible choice for preference optimization, since the model already knows
# how to follow instructions and DPO refines the preference-shaped scoring.
BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./dpo_model"
DATA_PATH = "./dpo_pairs.jsonl"

# Load the preference pairs produced by create_data.py. The dataset already
# has the "prompt"/"chosen"/"rejected" columns that DPOTrainer expects.
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    logging_steps=10,
    report_to="none",

    # DPOTrainer loads the model internally since we passed it a string path
    # above instead of a model object. model_init_kwargs lets us control that
    # internal from_pretrained() call, so device_map="auto" still puts the
    # model straight onto the GPU (NVIDIA RTX 2050) if one is available.
    model_init_kwargs={"device_map": "auto"},
)

trainer = DPOTrainer(
    model=BASE_MODEL_PATH,
    args=args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
