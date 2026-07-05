import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer


# PPO with a Process Reward Model.
#
# This is structurally identical to PPO_ORM/ppo_orm.py. The four-model setup,
# the bfloat16 choice, the small batch sizes — all the same. The only change
# is that the reward_model and value_model checkpoints come from the PRM
# pipeline instead of the ORM pipeline.
#
# There is one important design limitation to be aware of with PPO + PRM.
#
# In GRPO_PRM/grpo_prm.py the reward function is a Python function. We split
# the generated completion into numbered steps, score each step prefix with
# the PRM, and average those scores. This is the "correct" PRM usage because
# it gives the policy a dense signal: step 3 was wrong → step 3 gets a low
# score, steps 1-2 were correct → they get high scores.
#
# In PPO, trl's PPOTrainer does not accept a Python reward function. It only
# accepts a torch.nn.Module. Internally, after generating each completion, it
# calls reward_model.forward(input_ids) on the full (prompt + completion)
# token sequence and uses the scalar output as the reward. There is no hook
# to inject step-splitting logic between the generation and the reward call.
#
# So for PPO_PRM, the PRM scores the full sequence rather than individual steps.
# This is a real limitation: the step-level granularity the PRM was trained for
# cannot be expressed in PPO's training loop. What we still get is a reward
# model whose representations were shaped by step-level training, which means
# it is more sensitive to local reasoning errors than an ORM would be. It is
# a halfway-house between ORM and a properly-integrated PRM, but it is the
# best PPO's API can do without modifying trl internals.
#
# If you want to compare the two fairly: train both GRPO_PRM and PPO_PRM on
# the same prompts and inspect which policy produces more consistently correct
# reasoning steps. The GRPO version should have an advantage because of the
# richer reward signal.

BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./prm_model"
VALUE_MODEL_PATH = "./value_model"
OUTPUT_DIR = "./ppo_prm_model"
DATA_PATH = "./ppo_prompts.jsonl"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    return {"input_ids": tokenizer(example["prompt"])["input_ids"]}


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

policy_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
# The PRM checkpoint loaded here was trained to score step-level contexts.
# PPOTrainer will call it on full sequences, but the step-aware representations
# still make it a more informative reward signal than an ORM would be in this
# position — at least in theory.
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    VALUE_MODEL_PATH, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)

args = PPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    local_rollout_forward_batch_size=2,
    response_length=64,
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
