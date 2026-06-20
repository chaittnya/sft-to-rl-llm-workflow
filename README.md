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
  - `train_reward_model.py` — trains a reward model on those pairs and saves it to `./reward_model`.
  - `grpo.py` — a GRPO script that uses the SFT checkpoint as the base model and scores each generated completion with the trained reward model instead of a toy heuristic.
  - The GRPO example shows reward-guided policy updates driven by an actual trained reward model rather than a hand-written reward formula.

## How it was designed

The scripts are written in a style that is intended to resemble a postgraduate student report:
- explanatory comments are added throughout the code,
- comments give context for choices such as dataset selection, tokenizer settings, and training hyperparameters,
- the RL scripts all reuse the SFT-finetuned model as the starting point.

## Notes

- The repository is not a polished production training pipeline; it is a teaching-oriented collection of examples.
- The reward signal used in DPO is still a synthetic/toy placeholder. PPO and GRPO both train an actual reward model on synthetic preference data first; PPO additionally trains a value model used as its critic.
- The datasets are intentionally kept small to make the examples easier to inspect and to reduce runtime when testing locally.

## Usage

1. Run `SFT/sft.py` first to produce a base model checkpoint in `SFT/final_model`.
2. Use the `create_data.py` script in each RL folder to generate the input files.
3. For PPO specifically, also run `PPO/create_reward_data.py` then `PPO/train_reward_model.py` to produce a reward model checkpoint, and `PPO/create_value_data.py` then `PPO/train_value_model.py` to produce a value model checkpoint, before running `ppo.py`.
4. For GRPO specifically, also run `GRPO/create_reward_data.py` then `GRPO/train_reward_model.py` to produce a reward model checkpoint before running `grpo.py`.
5. Run the RL script in `PPO/`, `DPO/`, or `GRPO/` to continue training from the supervised checkpoint.

This README is intended as an overview and should help someone understand the general purpose and structure of the code.
