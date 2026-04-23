# JCIM SCAM Classification Pipeline

Benchmarking pipeline for SCAM classification experiments across classical ML, deep learning, and active learning settings.

This repository packages a research workflow for comparing sampling strategies and splitting strategies on SCAM datasets, with reproducible result artifacts and visualization notebooks included in the repo.

## Project Summary

The project evaluates whether training-data sampling can improve the performance of SCAM classification models. It compares:

- baseline training without sampling
- over-sampling with `SMOTE` and `ADASYN`
- under-sampling with `CondensedNearestNeighbour` and `InstanceHardnessThreshold`
- active learning with entropy-based sample selection
- multiple dataset variants and train/validation/test split strategies

The codebase computes molecular descriptors from SMILES strings, trains several model variants, tracks evaluation metrics across repeated runs, and writes per-study result tables for downstream analysis.

## Highlights

- packaged Python source under [`jcim/`](jcim)
- runnable CLI entrypoint via `python -m jcim`
- benchmark outputs and analysis notebooks included for traceability
- project metadata and formatter config centralized in [`pyproject.toml`](pyproject.toml)
- repository cleanup for portfolio-friendly presentation

## Repository Layout

```text
jcim/          Core pipeline, models, splitters, validation, and utilities
Datasets/      Input datasets used by the benchmark pipeline
Results/       Generated experiment outputs and analysis figures
modAL/         Local modAL code used by the active-learning workflow
```

## Running The Pipeline

Preferred entrypoint:

```bash
python -m jcim --study_name N_SF_TTS
```

Equivalent console script after installation:

```bash
jcim --study_name N_SF_TTS
```

Optional dataset path override:

```bash
python -m jcim --study_name N_SF_TTS --datasets_path ./Datasets
```

Study names follow the pattern:

```text
<sampling>_<dataset>_<split>
```

Examples:

- `N_SF_TTS`: no sampling, `SCAMS_filtered.csv`, train/test split
- `SMOTE_SP1_B`: SMOTE, balanced-positive dataset, Butina split
- `ADASYN_SP2_SS`: ADASYN, augmented-positive dataset, scaffold split

## Method Overview

Pipeline stages:

1. Load dataset and held-out test set from `Datasets/`.
2. Convert SMILES strings into molecular descriptors.
3. Split train and validation sets with the configured strategy.
4. Train non-active-learning models.
5. Train the active-learning model over iterative query rounds.
6. Evaluate on validation and test sets.
7. Save run-level metrics and aggregate study outputs into `Results/`.

Tracked metrics include:

- ROC AUC with confidence interval bounds
- accuracy
- F1
- MCC

## Environment Notes

The repository includes a historical `requirements.txt`, but some chemistry and deep learning dependencies are environment-sensitive. For research reproduction, a containerized environment is still the safest option.

Existing Docker image reference from the original project:

```bash
docker pull khalimat/jcim_f_holly
docker run -it --name jcim -v <repo-dir>:/root/mydir khalimat/jcim_f_holly
```

## Results And Analysis

Representative outputs and visualizations are committed under [`Results/`](Results). The notebooks used for figure generation remain in the repository for inspection and reproduction, including:

- [Results/Study_1/Study_1_analysis.ipynb](Results/Study_1/Study_1_analysis.ipynb)
- [Results/Study_2/Study_2_analysis.ipynb](Results/Study_2/Study_2_analysis.ipynb)
- [Results/Study_3/AA_validation.ipynb](Results/Study_3/AA_validation.ipynb)

Architecture diagram:

![Pipeline UML](Pipeline_UML.jpg)

## Notes

- This repo is presented as a research engineering project, not a polished production service.
- Result folders are intentionally kept to show experimental output and analysis artifacts.
- Some code paths depend on legacy TensorFlow and cheminformatics tooling.
