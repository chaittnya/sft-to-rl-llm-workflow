import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# Configuration
# The GRPO script uses the SFT checkpoint as the initial policy. This setup is
# aligned with the idea of bootstrapping reinforcement learning from supervised
# pretraining, which is a common strategy in modern language model research.
# REWARD_MODEL_PATH points at the checkpoint produced by train_reward_model.py.
# Run create_reward_data.py and train_reward_model.py before this script,
# otherwise this path will not exist yet.
BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./reward_model"
OUTPUT_DIR = "./grpo_model"
DATA_PATH = "./grpo_items.jsonl"

# Load the prompts produced by create_data.py.
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Load the trained reward model once, outside the reward function, so we are
# not reloading it from disk on every single call during training.
# device_map="auto" puts it on the GPU (NVIDIA RTX 2050) if one is available.
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto"
)
reward_model.eval()


# Score each generated completion with the trained reward model instead of a
# toy length heuristic. GRPOTrainer calls this with the prompts and the
# completions it just generated, and expects a list of floats back, one
# score per completion.
def reward_func(prompts, completions, **kwargs):
    texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
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

    # GRPOTrainer loads the model internally since we passed it a string path
    # above instead of a model object. model_init_kwargs lets us control that
    # internal from_pretrained() call, so device_map="auto" still puts the
    # model straight onto the GPU (NVIDIA RTX 2050) if one is available.
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
