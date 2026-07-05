# LLM Fine-Tuning and RL Project

This repository contains scripts for training and experimenting with large language models using supervised fine-tuning and reinforcement learning methods.

## What is included

- `SFT/`
  - `sft.py` — a supervised fine-tuning script that trains a causal language model on a small subset of the `yahma/alpaca-cleaned` dataset.
  - The script loads the tokenizer and model from a pretrained checkpoint, formats examples into an instruction-input-response prompt, and saves the resulting fine-tuned model to `./final_model`.

- `PPO/`
  - `create_data.py` — generates prompt data for PPO training in JSONL format.
  - `create_reward_data.py` — generates synthetic chosen/rejected pairs used to train a reward model.
  - `train_reward_model.py` — trains a reward model on those pairs and saves it to `./reward_model`.
  - `create_value_data.py` — generates regression examples (response text plus a target score) used to train a value model.
  - `train_value_model.py` — fine-tunes a value model, warm-started from the reward model checkpoint, and saves it to `./value_model`.
  - `ppo.py` — a PPO script that uses the SFT-finetuned model as the policy, the trained reward model checkpoint to score responses, and the trained value model checkpoint as the critic.
  - The PPO code is intended to show how a policy and reference model can be initialized from the same supervised checkpoint, and how the reward model and value model are each trained for their own distinct job (pairwise ranking vs. single-response regression) before PPO uses them.

- `DPO/`
  - `create_data.py` — creates synthetic preference pairs for DPO training.
  - `dpo.py` — a DPO script that starts from the SFT-finetuned model and optimizes it using paired preferred/dispreferred examples.
  - This directory is designed to demonstrate how preference-based fine-tuning can be set up from a pretrained policy.

- `GRPO/`
  - `create_data.py` — generates example prompts for GRPO.
  - `create_reward_data.py` — generates synthetic chosen/rejected pairs used to train a reward model.
  - `train_reward_model.py` — trains a pairwise (Bradley-Terry) reward model on those pairs and saves it to `./reward_model`.
  - `grpo.py` — a GRPO script that uses the SFT checkpoint as the base model and scores each generated completion with the trained reward model instead of a toy heuristic.
  - The GRPO example shows reward-guided policy updates driven by an actual trained reward model rather than a hand-written reward formula.

- `GRPO_KWISE/`
  - `create_data.py` — generates example prompts (same format as `GRPO/create_data.py`).
  - `create_kwise_reward_data.py` — generates k-wise ranking data: 4 completions per prompt ordered from best to worst (full answer → truncated answer → generic filler → refusal).
  - `train_kwise_reward_model.py` — trains a reward model using the **Plackett-Luce** listwise ranking loss instead of the standard pairwise Bradley-Terry loss, and saves it to `./kwise_reward_model`.
  - `grpo_kwise.py` — a GRPO script identical in structure to `GRPO/grpo.py` but using the k-wise reward model checkpoint as the reward signal.
  - This directory demonstrates how replacing the pairwise reward objective with a listwise one affects the reward model's training signal; the GRPO training loop itself is unchanged.

- `PPO_KWISE/`
  - `create_data.py` — generates Alpaca prompts (same format as `PPO/create_data.py`).
  - `create_kwise_reward_data.py` — generates the same 4-way ranked completion data as `GRPO_KWISE/`.
  - `train_kwise_reward_model.py` — trains a **Plackett-Luce** reward model. Contains the custom `KWiseCollator` (batches K texts per row into a 3D tensor) and `PlackettLuceTrainer` (implements the PL ranking loss).
  - `create_value_data.py` — generates binary regression data to warm-start the PPO critic.
  - `train_value_model.py` — fine-tunes a value model initialised from the k-wise reward model checkpoint.
  - `ppo_kwise.py` — PPO with the Plackett-Luce reward model. Unlike `PPO_PRM/`, there is no inference-time limitation: the PL model scores a single (prompt + completion) text with one forward pass, which is exactly what PPOTrainer's reward model API expects.
  - Direct comparisons: `PPO_KWISE/` vs `PPO/` isolates the reward model training objective; `PPO_KWISE/` vs `GRPO_KWISE/` isolates the RL algorithm.

- `GRPO_ORM/`
  - Task: **GSM8K math word problems** (switched from Alpaca because ORM needs a verifiable ground-truth outcome).
  - `create_data.py` — generates math prompts formatted to elicit numbered step-by-step solutions.
  - `create_orm_data.py` — generates binary-labelled training data: `(question + correct solution, 1.0)` and `(question + wrong solution, 0.0)`.
  - `train_orm.py` — trains an **Outcome Reward Model** that scores a complete solution in isolation (MSELoss regression, `num_labels=1`). Contrast with Bradley-Terry, which always compares two responses.
  - `grpo_orm.py` — GRPO with the ORM as the reward function; the full `(prompt + completion)` is scored in one forward pass.

- `GRPO_PRM/`
  - Task: **GSM8K math word problems** (PRM needs explicit, parseable reasoning steps).
  - `create_data.py` — same math prompt format as `GRPO_ORM/`.
  - `create_prm_data.py` — generates step-level training data: for each step in a reference solution, one correct row (growing context up to that step, label 1.0) and one corrupted row (a number in the step is nudged, label 0.0).
  - `train_prm.py` — trains a **Process Reward Model** that scores each reasoning step (same architecture as ORM, different training data).
  - `grpo_prm.py` — GRPO whose reward function splits each generated completion on `Step N:` boundaries, scores each step prefix with the PRM, and averages the step scores. This gives denser feedback than ORM: a completion that gets 3 out of 4 steps right scores ~0.75 rather than a binary 0.

- `PPO_ORM/`
  - Task: **GSM8K math word problems**.
  - `create_data.py`, `create_orm_data.py`, `train_orm.py` — identical logic to `GRPO_ORM/`.
  - `create_value_data.py` — generates binary regression data for the value model (same format as the ORM data).
  - `train_value_model.py` — fine-tunes a value model warm-started from the ORM checkpoint, so the critic inherits the ORM's understanding of outcome quality before PPO begins updating it.
  - `ppo_orm.py` — PPO with the ORM as the reward model. Because `trl.experimental.ppo.PPOTrainer` calls the reward model on the full `(prompt + completion)` sequence internally, ORM integrates into the PPO loop without any changes to the training code.

- `PPO_PRM/`
  - Task: **GSM8K math word problems**.
  - `create_data.py`, `create_prm_data.py`, `train_prm.py` — identical logic to `GRPO_PRM/`.
  - `create_value_data.py`, `train_value_model.py` — value model warm-started from the PRM checkpoint.
  - `ppo_prm.py` — PPO with the PRM as the reward model. Unlike `GRPO_PRM/`, the step-splitting aggregation cannot be done inside `PPOTrainer`'s reward call (it requires a `torch.nn.Module`, not a Python function), so the PRM scores the full sequence. The script comments explain this trade-off explicitly. This makes `PPO_PRM` vs `GRPO_PRM` a useful comparison of how the reward interface design of each algorithm constrains what reward models can do.

## How it was designed

The scripts are written in a style that is intended to resemble a postgraduate student report:
- explanatory comments are added throughout the code,
- comments give context for choices such as dataset selection, tokenizer settings, and training hyperparameters,
- the RL scripts all reuse the SFT-finetuned model as the starting point.

## Notes

- The repository is not a polished production training pipeline; it is a teaching-oriented collection of examples.
- The reward signal used in DPO is still a synthetic/toy placeholder. PPO and GRPO both train an actual reward model on synthetic preference data first; PPO additionally trains a value model used as its critic.
- `GRPO_KWISE` uses the same GRPO algorithm as `GRPO` but trains its reward model with a Plackett-Luce (k-wise) loss instead of the Bradley-Terry pairwise loss. The two directories are otherwise identical in structure so the difference in reward model training is easy to compare.
- `GRPO_ORM`, `GRPO_PRM`, `PPO_ORM`, and `PPO_PRM` switch to GSM8K math word problems because ORM and PRM require a verifiable outcome (the final numerical answer) and explicit reasoning steps respectively. These concepts do not apply cleanly to the open-ended Alpaca instruction-following task.
- A key contrast between `GRPO_PRM` and `PPO_PRM`: GRPO accepts a Python reward function, so the PRM can score each step of a generated solution and average those scores (dense feedback). PPO's `trl` API accepts only a `torch.nn.Module` as the reward model, so the PRM is called on the full sequence — it still provides a meaningful signal but loses step-level granularity. This design difference is documented in `PPO_PRM/ppo_prm.py`.
- The datasets are intentionally kept small to make the examples easier to inspect and to reduce runtime when testing locally.

## Usage

1. Run `SFT/sft.py` first to produce a base model checkpoint in `SFT/final_model`.
2. Use the `create_data.py` script in each RL folder to generate the input files.
3. For PPO specifically, also run `PPO/create_reward_data.py` then `PPO/train_reward_model.py` to produce a reward model checkpoint, and `PPO/create_value_data.py` then `PPO/train_value_model.py` to produce a value model checkpoint, before running `ppo.py`.
4. For GRPO specifically, also run `GRPO/create_reward_data.py` then `GRPO/train_reward_model.py` to produce a reward model checkpoint before running `grpo.py`.
5. For the k-wise GRPO variant, run `GRPO_KWISE/create_data.py`, then `GRPO_KWISE/create_kwise_reward_data.py`, then `GRPO_KWISE/train_kwise_reward_model.py`, and finally `GRPO_KWISE/grpo_kwise.py`.
6. For `GRPO_ORM`: run `create_data.py` → `create_orm_data.py` → `train_orm.py` → `grpo_orm.py`.
7. For `GRPO_PRM`: run `create_data.py` → `create_prm_data.py` → `train_prm.py` → `grpo_prm.py`.
8. For `PPO_ORM`: run `create_data.py` → `create_orm_data.py` → `train_orm.py` → `create_value_data.py` → `train_value_model.py` → `ppo_orm.py`.
9. For `PPO_PRM`: run `create_data.py` → `create_prm_data.py` → `train_prm.py` → `create_value_data.py` → `train_value_model.py` → `ppo_prm.py`.
10. For `PPO_KWISE`: run `create_data.py` → `create_kwise_reward_data.py` → `train_kwise_reward_model.py` → `create_value_data.py` → `train_value_model.py` → `ppo_kwise.py`.

This README is intended as an overview and should help someone understand the general purpose and structure of the code.
