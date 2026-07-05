import re
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Import order matters: GRPOConfig before GRPOTrainer segfaults on torch 2.6.
from trl import GRPOTrainer, GRPOConfig


# GRPO with a Process Reward Model as the reward signal.
#
# The key difference from grpo_orm.py is in how the reward function works.
# ORM gives one score for the whole completion. PRM gives one score per step,
# and I average those scores to get the final reward for the completion.
#
# Why does averaging step scores give better training signal than one final score?
# Consider a 4-step solution where steps 1, 2, 3 are correct but step 4 is wrong.
# ORM gives this a reward of 0 (wrong final answer). The policy gets no signal
# about which parts were good. PRM gives step scores of roughly [1, 1, 1, 0]
# and the average reward is 0.75. The policy gets a positive signal for the
# first three steps and a negative signal specifically for step 4. This is
# much more informative for learning.
#
# GRPO is a particularly good fit for PRM because GRPO accepts a Python
# function as the reward. That means I can write arbitrary Python code inside
# the reward function — split the text, loop over steps, call the model N
# times, aggregate. PPO's API does not allow this (see PPO_PRM/ppo_prm.py for
# how PPO handles it differently and what gets lost in the process).

BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./prm_model"
OUTPUT_DIR = "./grpo_prm_model"
DATA_PATH = "./grpo_items.jsonl"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto"
)
reward_model.eval()


def split_steps(text):
    # Split the generated text on "Step N:" boundaries. re.split with a
    # lookahead (?=...) keeps the delimiter at the start of each part rather
    # than discarding it, so each chunk still has its "Step N:" header.
    # For example "Step 1: foo\nStep 2: bar" splits into
    # ["Step 1: foo\n", "Step 2: bar"].
    parts = re.split(r"(?=Step \d+:)", text)
    return [p.strip() for p in parts if p.strip()]


def score_context(context):
    # Tokenise and score a single context string with the PRM. Returns a plain
    # Python float so it can be collected into a list later.
    inputs = reward_tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(reward_model.device)
    with torch.no_grad():
        score = reward_model(**inputs).logits.squeeze(-1)
    return score.item()


def reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        steps = split_steps(completion)

        if not steps:
            # The model did not produce any numbered steps — maybe it wrote a
            # paragraph instead. Give a low reward so the policy learns to use
            # the numbered format we need.
            rewards.append(0.0)
            continue

        step_scores = []
        for i, step in enumerate(steps):
            # Build the growing context: question + all steps up to and
            # including step i. This is exactly the format the PRM was trained
            # on in create_prm_data.py. Giving it the same format at inference
            # time is important — if the prompt format changes, the scores
            # become unreliable.
            context = prompt + "\n".join(steps[: i + 1])
            step_scores.append(score_context(context))

        # Average across all step scores to get one reward per completion.
        # Min would be a stricter alternative: the reward equals the worst step.
        # That would encourage the policy to never make a single mistake, but
        # it also means one bad step tanks an otherwise good solution completely.
        # Mean is a softer signal that rewards partial correctness.
        rewards.append(sum(step_scores) / len(step_scores))

    return rewards


args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    num_generations=4,
    per_device_train_batch_size=4,
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
