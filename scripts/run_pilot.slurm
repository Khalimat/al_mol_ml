#!/bin/bash
# Slurm batch script for scripts/run_pilot.py -- the generalization-study
# pilot matrix (docs/generalization_study_design.md). Runs everything
# serially in one job/one GPU; see the note at the bottom for splitting it
# into a job array once you're past the pilot and running the full
# 10-iteration study.
#
# The lines below marked "<-- adjust" are cluster-specific and almost
# certainly need editing for your Slurm setup (partition/account names,
# GPU resource syntax, and module names all vary by site).
#
# Usage:
#   sbatch scripts/run_pilot.slurm                      # full pilot matrix
#   sbatch scripts/run_pilot.slurm --datasets BBBP CLINTOX --architectures mlp rf
#   (any arguments after the script name are passed straight through to
#   scripts/run_pilot.py -- see that script's own --help)

#SBATCH --job-name=almolml-pilot
#SBATCH --partition=gpu              # <-- adjust: your cluster's GPU partition name
#SBATCH --gres=gpu:h100:1            # <-- adjust: your cluster's GPU resource syntax
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
# #SBATCH --account=your_account     # <-- adjust: uncomment and set if your cluster requires one

set -euo pipefail

# module load cuda/12.x              # <-- adjust: uncomment if your cluster needs an explicit
                                      #     CUDA module loaded before torch can see the GPU

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# uv manages its own Python + venv, so no module-load for Python itself is
# needed unless the compute nodes block outbound network access -- in that
# case, install uv system-wide (or in $HOME) before submitting, per
# https://docs.astral.sh/uv/getting-started/installation/, and/or run
# `uv sync`/`prepare_new_datasets.py` once from a login node with network
# access instead of inside the job.
uv sync --extra gpu --extra graph --extra test

# Datasets only need downloading once -- prepare_new_datasets.py skips any
# file that already exists, so re-running this is harmless/cheap on a
# resubmitted job.
uv run python scripts/prepare_new_datasets.py

uv run python scripts/run_pilot.py "$@"

# To scale up to the full 10-iteration study across more GPUs at once
# rather than one long serial job, turn this into a job array: add
# `#SBATCH --array=0-N` and replace the run_pilot.py call above with one
# that passes a single (dataset, architecture) pair selected by
# $SLURM_ARRAY_TASK_ID (e.g. via a small lookup table in this script, or
# `--datasets ${DATASETS[$SLURM_ARRAY_TASK_ID]}` etc.) -- run_pilot.py's
# --datasets/--architectures/--strategies flags already accept a subset, so
# no changes to the Python side are needed for this.
