import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
# PPOTrainer is in trl.experimental.ppo, not the top-level trl namespace.
# This placement means the API may change in future trl versions, but it is
# the correct location for trl 1.6.0. It also requires torch>=2.6 because
# trl imports torch.distributed.fsdp.FSDPModule at the module level.
from trl.experimental.ppo import PPOConfig, PPOTrainer


# PPO with an Outcome Reward Model.
#
# PPO needs four models simultaneously. On a GPU with 4GB of VRAM this is
# tight, but manageable with bfloat16 and small batch sizes:
#
# policy_model: the LLM we are training. It starts from the SFT checkpoint
#   and gets updated at every PPO step to produce better completions.
#
# ref_model: a frozen copy of the initial policy. PPO uses it to compute a
#   KL-divergence penalty that prevents the policy from drifting too far from
#   its starting point. Without this, RL would quickly push the policy into
#   degenerate behaviour (e.g. producing very short responses that happen to
#   score well on the reward model but are useless in practice).
#
# reward_model: the ORM trained in train_orm.py. It scores each completed
#   response with a scalar and is frozen throughout PPO. PPOTrainer calls it
#   internally by running reward_model.forward() on the full tokenised
#   (prompt + completion) sequence.
#
# value_model: the critic, trained in train_value_model.py. It estimates the
#   expected future reward from the current state, and gets updated every step
#   alongside the policy. PPO uses (actual_reward - value_estimate) as the
#   advantage. A well-calibrated value model reduces variance in the advantage
#   signal, which makes policy updates more stable.
#
# This is the same structure as PPO/ppo.py. The only change is the reward
# and value model checkpoints pointing to ORM-based paths instead of
# Bradley-Terry-based ones.

BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./orm_model"
VALUE_MODEL_PATH = "./value_model"
OUTPUT_DIR = "./ppo_orm_model"
DATA_PATH = "./ppo_prompts.jsonl"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    # PPOTrainer expects the dataset to have an "input_ids" column containing
    # the tokenised prompt (not raw text). The raw "prompt" column is removed
    # after this map call because remove_columns=dataset.column_names drops
    # everything that was there before.
    return {"input_ids": tokenizer(example["prompt"])["input_ids"]}


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# All four models in bfloat16. Two reasons: memory (bfloat16 halves the VRAM
# footprint vs float32) and numerical stability (bfloat16 has the same
# exponent range as float32, so raw unscaled gradient updates do not overflow
# to NaN the way float16 does). float16 would cut memory just as much but
# causes NaN gradients in PPO almost immediately on this setup.
policy_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    VALUE_MODEL_PATH, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)

args = PPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,

    # local_rollout_forward_batch_size controls how many completions are
    # generated in one no-grad forward pass during rollout. The default is 64,
    # which easily OOMs on a 4GB GPU with four models loaded. Setting it to 2
    # (same as the train batch size) keeps memory flat.
    local_rollout_forward_batch_size=2,

    # Math solutions are longer than the Alpaca-style answers in PPO/. 32
    # tokens was fine for "### Response: The answer is X." but math steps need
    # more room. 64 still fits in memory and gives the policy enough space to
    # write at least two or three arithmetic steps.
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
