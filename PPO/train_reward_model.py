from datasets import load_dataset
from trl import RewardConfig, RewardTrainer


# ==========================================================
# REWARD MODEL TRAINING
# ==========================================================
# PPO needs a reward model that can actually look at a response and give it a
# score. Without this, PPO has nothing real to optimize against. This script
# trains that reward model on the chosen/rejected pairs from
# create_reward_data.py, before we ever touch ppo.py.
#
# We start the reward model from the same SFT checkpoint as the policy. The
# reward model is loaded as AutoModelForSequenceClassification under the
# hood (with num_labels=1), so it outputs a single score instead of a
# distribution over the vocabulary.
BASE_MODEL_PATH = "../SFT/final_model"
OUTPUT_DIR = "./reward_model"
DATA_PATH = "./reward_pairs.jsonl"

# Load the chosen/rejected pairs produced by create_reward_data.py.
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

args = RewardConfig(
    output_dir=OUTPUT_DIR,

    # RewardTrainer loads the model internally since we passed it a string
    # path above instead of a model object. model_init_kwargs lets us control
    # that internal from_pretrained() call, so device_map="auto" still puts
    # the model straight onto the GPU (NVIDIA RTX 2050) if one is available.
    model_init_kwargs={"device_map": "auto"},

    # Number of complete passes over the preference data.
    # Possible values: 1 for a quick pass, 2-3 if the dataset is small and you
    # want the reward model to fit it more closely.
    num_train_epochs=1,

    # Batch size per device. Smaller values use less memory but train slower.
    per_device_train_batch_size=4,

    # Learning rate for the reward head and the underlying model weights.
    # Common values: 1e-5 to 1e-4 for this kind of small fine-tune.
    learning_rate=1e-5,

    logging_steps=10,
    report_to="none",
)

trainer = RewardTrainer(
    model=BASE_MODEL_PATH,
    args=args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
