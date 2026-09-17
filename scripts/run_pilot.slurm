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
#SBATCH --gres=gpu:a100:1            # matches the srun request you said works on this cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G                    # matches your working srun request; bump if a run OOMs
#SBATCH --time=02:00:00
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
# #SBATCH --account=your_account     # <-- adjust: uncomment and set if your cluster requires one
#
# No --partition here on purpose: some sites' job_submit plugin rejects an
# explicit partition alongside --gres and routes GPU jobs to the right
# partition automatically from --gres alone (this is what the error
# "Specifying gpu partition is not allowed, Please use '--gres'..." means).
# If your site *does* need an explicit partition, add it back as
# `#SBATCH --partition=<name>` -- ask your cluster admins or check
# `sinfo` for the GPU partition's name.
#
# --gres/--mem/--time above match a request you'd already confirmed works
# interactively on this cluster (`srun --gres=gpu:a100:1 --mem=10G
# --time=02:00:00 --pty bash`); --cpus-per-task=4 is the one value not
# taken from that, so if sbatch still rejects the job ("Job violates
# accounting/QOS policy..."), that's the one to look at first -- check
# `sacctmgr -p show assoc user=$USER format=account,qos` and `sacctmgr show
# qos format=name,maxwall,maxtresperuser` for your actual limits.

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
