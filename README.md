# LLM Fine-Tuning and RL Project

This repository contains scripts for training and experimenting with large language models using supervised fine-tuning and reinforcement learning methods.

## What is included

- `SFT/`
  - `sft.py` — a supervised fine-tuning script that trains a causal language model on a small subset of the `yahma/alpaca-cleaned` dataset.
  - The script loads the tokenizer and model from a pretrained checkpoint, formats examples into an instruction-input-response prompt, and saves the resulting fine-tuned model to `./final_model`.

- `PPO/`
  - `create_data.py` — generates prompt data for PPO training in JSONL format.
  - The PPO code is intended to show how a policy and reference model can be initialized from the same supervised checkpoint and then updated based on a toy reward function.
  - `ppo.py` — a simple PPO script that uses the SFT-finetuned model as the base policy.

- `DPO/`
  - `create_data.py` — creates synthetic preference pairs for DPO training.
  - This directory is designed to demonstrate how preference-based fine-tuning can be set up from a pretrained policy.
  - `dpo.py` — a DPO script that starts from the SFT-finetuned model and optimizes it using paired preferred/dispreferred examples.

- `GRPO/`
  - `create_data.py` — generates example prompt-reward pairs for GRPO.
  - The GRPO example is intentionally lightweight and uses a toy reward metric for demonstration.
  - `grpo.py` — a GRPO script that also uses the SFT checkpoint as the base model and performs reward-guided policy updates.

## How it was designed

The scripts are written in a style that is intended to resemble a postgraduate student report:
- explanatory comments are added throughout the code,
- comments give context for choices such as dataset selection, tokenizer settings, and training hyperparameters,
- the RL scripts all reuse the SFT-finetuned model as the starting point.

## Notes

- The repository is not a polished production training pipeline; it is a teaching-oriented collection of examples.
- The RL reward functions in the current scripts are placeholders and should be replaced with real reward models or preference data for serious experiments.
- The datasets are intentionally kept small to make the examples easier to inspect and to reduce runtime when testing locally.

## Usage

1. Run `SFT/sft.py` first to produce a base model checkpoint in `SFT/final_model`.
2. Use the `create_data.py` script in each RL folder to generate the input files.
3. Run the RL script in `PPO/`, `DPO/`, or `GRPO/` to continue training from the supervised checkpoint.

This README is intended as an overview and should help someone understand the general purpose and structure of the code.
