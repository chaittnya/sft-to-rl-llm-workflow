<#
.SYNOPSIS
    Runs the SFT -> PPO / DPO / GRPO scripts in this repo in the correct order.

.DESCRIPTION
    Each script in this repo assumes it is run from inside its own folder
    (they use relative paths like "../SFT/final_model" or "./reward_model").
    This script takes care of that, and runs every stage in the order the
    README describes:
    
    powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1


      1. SFT/lora.py                     -> SFT/final_model
      2. PPO/create_data.py              -> PPO/ppo_prompts.jsonl
         PPO/create_reward_data.py       -> PPO/reward_pairs.jsonl
         PPO/train_reward_model.py       -> PPO/reward_model
         PPO/create_value_data.py        -> PPO/value_data.jsonl
         PPO/train_value_model.py        -> PPO/value_model
         PPO/ppo.py                      -> PPO/ppo_model
      3. DPO/create_data.py              -> DPO/dpo_pairs.jsonl
         DPO/dpo.py                      -> DPO/dpo_model
      4. GRPO/create_data.py             -> GRPO/grpo_items.jsonl
         GRPO/create_reward_data.py      -> GRPO/reward_pairs.jsonl
         GRPO/train_reward_model.py      -> GRPO/reward_model
         GRPO/grpo.py                    -> GRPO/grpo_model

    Use the -Skip* switches below to skip a stage, e.g. if you already have a
    checkpoint from a previous run and only want to re-run PPO.

.PARAMETER PythonPath
    Path to the python.exe that has datasets/transformers/peft/trl/accelerate
    installed. Defaults to the project's conda env.

.PARAMETER SkipSFT
    Skip supervised fine-tuning. Only use this if SFT/final_model already exists.

.PARAMETER SkipPPO
    Skip the whole PPO stage (data creation, reward model, value model, PPO training).

.PARAMETER SkipDPO
    Skip the DPO stage.

.PARAMETER SkipGRPO
    Skip the GRPO stage.

.EXAMPLE
    .\run_pipeline.ps1
    Runs every stage from scratch.

.EXAMPLE
    .\run_pipeline.ps1 -SkipSFT -SkipDPO -SkipGRPO
    Re-runs only the PPO stage, assuming SFT/final_model already exists.
#>

param(
    [string]$PythonPath = "C:\Users\user\.conda\envs\py312\python.exe",
    [switch]$SkipSFT,
    [switch]$SkipPPO,
    [switch]$SkipDPO,
    [switch]$SkipGRPO
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

# Runs one script with the given folder as the working directory, since the
# scripts in this repo rely on relative paths to find checkpoints and data.
function Invoke-Stage {
    param(
        [string]$Folder,
        [string]$Script
    )

    Write-Host ""
    Write-Host "==> $Folder\$Script" -ForegroundColor Cyan

    Push-Location (Join-Path $RepoRoot $Folder)
    try {
        & $PythonPath $Script
        if ($LASTEXITCODE -ne 0) {
            throw "$Folder\$Script exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipSFT) {
    Invoke-Stage "SFT" "lora.py"
}
else {
    Write-Host "Skipping SFT stage." -ForegroundColor Yellow
}

if (-not $SkipPPO) {
    Invoke-Stage "PPO" "create_data.py"
    Invoke-Stage "PPO" "create_reward_data.py"
    Invoke-Stage "PPO" "train_reward_model.py"
    Invoke-Stage "PPO" "create_value_data.py"
    Invoke-Stage "PPO" "train_value_model.py"
    Invoke-Stage "PPO" "ppo.py"
}
else {
    Write-Host "Skipping PPO stage." -ForegroundColor Yellow
}

if (-not $SkipDPO) {
    Invoke-Stage "DPO" "create_data.py"
    Invoke-Stage "DPO" "dpo.py"
}
else {
    Write-Host "Skipping DPO stage." -ForegroundColor Yellow
}

if (-not $SkipGRPO) {
    Invoke-Stage "GRPO" "create_data.py"
    Invoke-Stage "GRPO" "create_reward_data.py"
    Invoke-Stage "GRPO" "train_reward_model.py"
    Invoke-Stage "GRPO" "grpo.py"
}
else {
    Write-Host "Skipping GRPO stage." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "All requested stages completed successfully." -ForegroundColor Green
