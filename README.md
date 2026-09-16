# al_mol_ml: Active-Learning SCAM Classification Pipeline

Benchmarking pipeline for SCAM classification experiments across classical ML, deep learning, and active learning settings.

This repository packages a research workflow for comparing active-learning query strategies and dataset-splitting strategies on SCAM datasets, reproducible via Docker or a local Python environment (see "Running The Pipeline" below).

## Project Summary

The project evaluates whether active learning can identify a more informative training subset than
labeling the full pool, for SCAM classification models. It compares:

- baseline training on the full labeled pool
- active learning with entropy, BALD, Core-Set, and DIRECT query strategies
- multiple dataset variants and train/validation/test split strategies

An earlier version of this project also compared class-imbalance resampling techniques (`SMOTE`,
`ADASYN`, `CondensedNearestNeighbour`, `InstanceHardnessThreshold`); that functionality has since been
removed to keep the codebase focused on the active-learning question above -- see git history if you
need it.

The codebase computes molecular descriptors from SMILES strings, trains several model variants, tracks evaluation metrics across repeated runs, and writes per-study result tables for downstream analysis.

## Repository Layout

```text
.
├── almolml/
│   ├── cli.py               argparse entrypoint
│   ├── pipeline.py          orchestrates one study: split -> train -> score -> write CSVs
│   ├── dataset.py           loads a CSV of (ID, SMILES, label) into featurized (X, Y)
│   ├── featurization.py     SMILES -> descriptor vectors; Butina clustering; scaffold grouping
│   ├── splitters.py         TTS / Butina / scaffold / "almost no validation" splitters
│   ├── models.py            DeepSCAMs (sklearn MLP), TorchMLPModel, ActiveLearningModel
│   ├── active_learning.py   a small pluggable active learner (no third-party AL library)
│   ├── query_strategies.py  acquisition functions: entropy, BALD (MC-Dropout), Core-Set, DIRECT
│   ├── validation.py        AUC/accuracy/F1/MCC for a model on one (X, Y) set
│   ├── delong.py            DeLong's method for the ROC AUC confidence interval
│   ├── utilities.py         small general-purpose helpers (arg parsing, filesystem setup)
│   └── paths.py             REPO_ROOT
├── Datasets/                input datasets used by the benchmark pipeline
├── Results/                 generated experiment outputs and analysis figures
└── tests/                   pytest suite: one test module per almolml module, plus a full pipeline smoke test
```

## Running The Pipeline

### With Docker (recommended)

```bash
docker build -t almolml .
docker run --rm -v $(pwd)/Results:/app/Results almolml --study_name SF_TTS
```

The `-v` mount persists results on the host; without it, results only exist
inside the container's filesystem. Builds a CPU-only image by default; for
a CUDA image on a machine with a GPU, add `--build-arg TORCH_VARIANT=gpu`.

### With a local Python environment

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.10+ (`.python-version`
pins 3.11, which `uv` will install automatically if it's missing).

torch ships as two mutually exclusive builds -- pick one when syncing:

```bash
uv sync --extra cpu   # portable, no GPU driver needed
# or
uv sync --extra gpu   # CUDA build; requires a matching NVIDIA driver
```

Then:

```bash
uv run almolml --study_name SF_TTS
```

Optional dataset path override:

```bash
uv run almolml --study_name SF_TTS --datasets_path ./Datasets
```

Useful flags for faster or non-interactive (e.g. CI) runs:

- `--iterations N` -- number of repeated train/evaluate runs (default 10)
- `--epochs N` -- training epochs for the neural network models (default 50)
- `--max_queries N` -- cap on active-learning query rounds per run (default:
  use the full training pool, which is what makes a full study slow)
- `--al_strategy {entropy,bald,core_set,direct}` -- which acquisition
  function picks the next active-learning query (default: entropy; see
  "Active Learning Query Strategies" below)
- `--overwrite` -- replace an existing results directory for the study
  without an interactive yes/no prompt
- `--seed N` -- base random seed (default: 0). Iteration `i` of a study is
  seeded with `seed + i`, so each iteration is an independent, reproducible
  draw instead of inheriting whatever RNG state prior calls happened to
  leave behind.

Study names follow the pattern:

```text
<dataset>_<split>
```

Examples:

- `SF_TTS`: `SCAMS_filtered.csv`, train/test split
- `SP1_B`: balanced-positive dataset, Butina split
- `SP2_SS`: augmented-positive dataset, scaffold split

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

## Active Learning Query Strategies

All four strategies share the same loop (`almolml/active_learning.py`'s `ActiveLearner`): start from a
small labeled seed set, repeatedly pick **one** unlabeled pool point to query next, label it, retrain
the model from scratch on the accumulated labeled set, and repeat. What differs between them is purely
*which point gets picked* -- the acquisition function -- implemented in `almolml/query_strategies.py`.
Select one with `--al_strategy {entropy,bald,core_set,direct}`.

### Entropy sampling (`entropy_query`)

Classic uncertainty sampling: run the model's single forward pass over the pool and pick the point
whose predicted class probabilities are closest to uniform (highest Shannon entropy). For binary
classification this is the point closest to `P(class=1) = 0.5` -- the model's single best guess is
least confident there. It uses only the model's current point estimate, with no notion of *why* the
model is unsure (a genuinely ambiguous point and a point the model just hasn't seen enough of yet look
identical to it).

> Lewis, D. D., & Gale, W. A. (1994). *A Sequential Algorithm for Training Text Classifiers*. SIGIR '94.

### BALD (`bald_query`, `bald_scores`)

Bayesian Active Learning by Disagreement, estimated via MC-Dropout. Instead of one forward pass, it
runs `n_mc_samples` stochastic passes with dropout left *on* at inference time, giving a small ensemble
of predictions per pool point. The BALD score is the mutual information between the prediction and the
model's (dropout-approximated) posterior over parameters: `predictive_entropy - expected_entropy`,
i.e. how uncertain the *averaged* prediction is, minus how uncertain each individual pass is on
average. This is high specifically when individual passes are each confident but disagree with each
other (epistemic uncertainty -- the model hasn't learned enough) and low when passes agree even if the
averaged prediction itself is uncertain (aleatoric/data noise) -- the distinction entropy sampling
can't make.

> Gal, Y., Islam, R., & Ghahramani, Z. (2017). *Deep Bayesian Active Learning with Image Data*. ICML
> 2017. (BALD itself originates in Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011).
> *Bayesian Active Learning for Classification and Preference Learning*. arXiv:1112.5745 -- Gal et al.
> contribute the MC-Dropout approximation used here to make it tractable for neural networks.)

### Core-Set (`core_set_query`, `farthest_point_index`)

Greedy k-center diversity sampling, with no model uncertainty involved at all. Every point (labeled and
pool) is projected into the model's learned embedding space (the penultimate layer, via `_embed`), and
the strategy queries whichever pool point is *farthest* from its nearest already-labeled neighbor --
i.e. whichever region of the input space the current labeled set covers most poorly. The premise: a
model trained on a labeled set that's geometrically well-spread over the data manifold generalizes
better than one trained on an uncertainty-selected set, which can clump around a single decision
boundary and miss whole regions of the space.

> Sener, O., & Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set
> Approach*. ICLR 2018.

### DIRECT (`direct_query`, `separation_threshold`)

Reduces active learning to a 1-D separation-threshold problem, aimed at class-imbalanced pools. From
the currently labeled set, `separation_threshold` finds the score value (predicted `P(minority class)`)
that best splits labeled points into majority/minority by minimizing balanced misclassification count;
`direct_query` then queries the pool point whose predicted `P(minority)` is closest to that threshold --
i.e. the point nearest the model's current best guess at the decision boundary. Falls back to entropy
sampling if the labeled set doesn't yet contain both classes (the threshold is undefined).

This is an adaptation, not a full implementation: the original paper's DIRECT is a multi-round batch
algorithm whose VReduce subroutine spends part of each round's budget *narrowing* a candidate threshold
interval before annotating near it, refining the estimate over several rounds within one query batch.
This repo's `ActiveLearner` queries one point at a time and re-estimates the threshold from scratch on
every single call -- a much weaker, single-shot version of the same core idea, since the threshold
estimate here never accumulates the benefit of that iterative narrowing. A properly batched,
multi-round DIRECT would need restructuring `ActiveLearner`'s single-point query/teach loop into a
batch acquisition loop, which hasn't been attempted here.

> Zhang, S., Katz-Samuels, J., & Nowak, R. (2025). *Improved Algorithms for Deep Active Learning under
> Imbalance via Optimal Separation*. ICML 2025. arXiv:2312.09196.

(Margin sampling, another common baseline, was considered and skipped: for binary classification it
ranks samples identically to entropy sampling, so it wouldn't add a distinct comparison point.)
