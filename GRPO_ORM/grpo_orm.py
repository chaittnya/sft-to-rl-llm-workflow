import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Import order matters: GRPOConfig before GRPOTrainer segfaults on torch 2.6
# with this version of trl. Trainer before Config is fine.
from trl import GRPOTrainer, GRPOConfig


# This script runs GRPO with an Outcome Reward Model as the reward signal.
#
# Quick recap of GRPO:
# GRPO (Group Relative Policy Optimisation) is a variant of PPO that removes
# the need for a separate value model (critic). Instead of estimating a
# baseline value for each state, it generates a group of completions for each
# prompt, computes the reward for each one, and uses the *relative* rewards
# within the group as the advantage signal. The completion with the highest
# reward in the group gets a positive advantage; the rest get negative ones.
# This sidesteps the whole "train a separate value model" step that PPO needs.
#
# How the ORM fits in:
# The reward function below receives the prompt and the generated completion,
# concatenates them, and runs the ORM on the combined text. The ORM returns a
# scalar score — high if the solution looks correct, low if it does not. GRPO
# then uses those scores (relative to the group mean) to update the policy.
#
# One thing I noticed: the ORM gives the same kind of score as a Bradley-Terry
# reward model (both return a scalar per completion), so plugging the ORM into
# grpo.py is structurally identical to plugging in the BT model. The difference
# is in what each model learned: BT learned "this is better than that" from
# pairwise comparisons; ORM learned "this is correct or not" from absolute labels.

BASE_MODEL_PATH = "../SFT/final_model"
REWARD_MODEL_PATH = "./orm_model"
OUTPUT_DIR = "./grpo_orm_model"
DATA_PATH = "./grpo_items.jsonl"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Load the ORM once here at module level, outside the reward function.
# If I loaded it inside reward_func it would reload the weights from disk on
# every single call during training, which would be extremely slow.
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH, num_labels=1, device_map="auto"
)
# eval() turns off dropout and batch normalisation. This matters because we
# are not updating the reward model — it is frozen — and we want deterministic
# scores at inference time.
reward_model.eval()


def reward_func(prompts, completions, **kwargs):
    # GRPOTrainer calls this function after generating a batch of completions.
    # It passes the original prompts and the corresponding generated text.
    # We must return a list of floats, one per (prompt, completion) pair.
    #
    # The ORM was trained on (question + full solution) texts, so we
    # concatenate prompt and completion before tokenising. This matches the
    # format the model saw during training.
    texts = [p + c for p, c in zip(prompts, completions)]
    inputs = reward_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(reward_model.device)

    with torch.no_grad():
        # .logits is a tensor of shape [batch_size, 1]. squeeze(-1) collapses
        # the last dimension to get a 1D tensor of shape [batch_size].
        scores = reward_model(**inputs).logits.squeeze(-1)

    return scores.tolist()


args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,

    # Each prompt is used to generate num_generations completions. GRPO then
    # ranks those completions by their reward and uses the relative ranking as
    # the advantage signal. More generations = better advantage estimates but
    # more GPU memory used per step.
    num_generations=4,

    per_device_train_batch_size=4,
    learning_rate=1e-5,
    logging_steps=10,
    report_to="none",

    # model_init_kwargs is passed to from_pretrained() when GRPOTrainer loads
    # the policy model internally. device_map="auto" makes it use the GPU.
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
