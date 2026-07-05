import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
# PPOTrainer is in trl.experimental.ppo, not the top-level trl namespace.
# This requires torch>=2.6 due to a trl dependency on FSDPModule.
from trl.experimental.ppo import PPOConfig, PPOTrainer


# PPO with a Plackett-Luce k-wise reward model.
#
# This is structurally identical to PPO/ppo.py — four models, bfloat16 to fit
# on 4GB VRAM, small batch sizes for the rollout. The only change is that the
# reward_model checkpoint comes from the Plackett-Luce training in
# train_kwise_reward_model.py rather than from Bradley-Terry RewardTrainer.
#
# One important thing to understand about how PL interacts with PPO:
#
# During training (train_kwise_reward_model.py), the PL model sees k=4
# completions at once and is optimised with the Plackett-Luce ranking loss.
# This is the "k-wise" part of the name.
#
# During PPO rollouts (this script), PPOTrainer calls reward_model.forward()
# on a single (prompt + completion) sequence and expects a scalar back. The PL
# model does this fine — it was trained to output a single logit per input, so
# the interface is identical to the Bradley-Terry reward model. There is no
# step-splitting or multi-text inference required, unlike PRM in PPO_PRM/.
#
# So PPO_KWISE and PPO/ are comparable in a clean way: same algorithm, same
# task, same inference API for the reward model. The only difference is the
# training objective of the reward model (pairwise vs. k-wise ranking).
# Any improvement in the policy trained by PPO_KWISE over PPO/ is attributable
# directly to the richer reward model training signal.

BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./kwise_reward_model"
VALUE_MODEL_PATH = "./value_model"
OUTPUT_DIR = "./ppo_kwise_model"
DATA_PATH = "./ppo_prompts.jsonl"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    # PPOTrainer expects the dataset to have an "input_ids" column with the
    # tokenised prompt. The raw "prompt" text column is no longer needed after
    # this, so remove_columns drops it.
    return {"input_ids": tokenizer(example["prompt"])["input_ids"]}


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# bfloat16 for all four models. The reason for bfloat16 over float16 is
# numerical stability: PPO computes raw (unscaled) gradient updates, and
# float16's narrow exponent range causes them to overflow to NaN almost
# immediately on this setup. bfloat16 has the same exponent range as float32,
# so it handles the dynamic range without a separate GradScaler.
policy_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

# The k-wise reward model is loaded with num_labels=1 even though it was trained
# without problem_type="regression" (PlackettLuceTrainer uses a custom loss,
# not MSELoss). num_labels=1 just tells AutoModelForSequenceClassification how
# many output neurons the scoring head has — the head shape was fixed at
# training time, so this must match what was used then.
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    VALUE_MODEL_PATH, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)

args = PPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,

    # Defaults to 64, which generates 64 completions in a single no-grad
    # forward pass during rollout. 64 is too large for 4GB VRAM with four
    # models loaded. Setting it equal to the train batch size (2) keeps the
    # memory footprint flat between rollout and update phases.
    local_rollout_forward_batch_size=2,

    response_length=32,
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
