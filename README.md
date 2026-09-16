# JCIM SCAM Classification Pipeline

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
jcim/
  cli.py             argparse entrypoint
  pipeline.py         orchestrates one study: split -> train -> score -> write CSVs
  dataset.py          loads a CSV of (ID, SMILES, label) into featurized (X, Y)
  featurization.py    SMILES -> descriptor vectors; Butina clustering; scaffold grouping
  splitters.py        TTS / Butina / scaffold / "almost no validation" splitters
  models.py           DeepSCAMs (sklearn MLP), TorchMLPModel, ActiveLearningModel
  active_learning.py  a small pluggable active learner (no third-party AL library)
  query_strategies.py acquisition functions: entropy, BALD (MC-Dropout), Core-Set, DIRECT
  validation.py       AUC/accuracy/F1/MCC for a model on one (X, Y) set
  delong.py           DeLong's method for the ROC AUC confidence interval
  utilities.py        small general-purpose helpers (arg parsing, filesystem setup)
  paths.py            REPO_ROOT
Datasets/      Input datasets used by the benchmark pipeline
Results/       Generated experiment outputs and analysis figures
tests/         Pytest suite: one test module per jcim module, plus a full pipeline smoke test
```

## Running The Pipeline

### With Docker (recommended)

```bash
docker build -t jcim .
docker run --rm -v $(pwd)/Results:/app/Results jcim --study_name SF_TTS
```

The `-v` mount persists results on the host; without it, results only exist
inside the container's filesystem.

### With a local Python environment

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m jcim --study_name SF_TTS
```

Equivalent console script after installation:

```bash
pip install -e .
jcim --study_name SF_TTS
```

Optional dataset path override:

```bash
python -m jcim --study_name SF_TTS --datasets_path ./Datasets
```

Useful flags for faster or non-interactive (e.g. CI) runs:

- `--iterations N` -- number of repeated train/evaluate runs (default 10)
- `--epochs N` -- training epochs for the neural network models (default 50)
- `--max_queries N` -- cap on active-learning query rounds per run (default:
  use the full training pool, which is what makes a full study slow)
- `--al_strategy {entropy,bald,core_set,direct}` -- which acquisition
  function picks the next active-learning query (default: entropy; see
  "Can active learning identify a more informative subset..." below)
- `--overwrite` -- replace an existing results directory for the study
  without an interactive yes/no prompt
- `--seed N` -- base random seed (default: 0). Iteration `i` of a study is
  seeded with `seed + i`, so each iteration is an independent, reproducible
  draw instead of inheriting whatever RNG state prior calls happened to
  leave behind -- see the seeding note under "Why this pipeline previously
  stopped working" below.

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

## Environment Notes

Class-imbalance resampling (`SMOTE`, `ADASYN`, `CondensedNearestNeighbour`,
`InstanceHardnessThreshold`, via `imbalanced-learn`) has been removed
entirely: the study-name grammar dropped its `<sampling>` segment (now
`<dataset>_<split>`, was `<sampling>_<dataset>_<split>`), `imbalanced-learn`
is no longer a dependency, and every code path that resampled the training
set before fitting a model is gone. This was tangential to the project's
actual question -- whether active learning finds a more informative
training subset than the full pool -- and cutting it keeps the codebase
focused on that.

`requirements.txt` and `pyproject.toml` pin a dependency set that is
verified to work together (see CI). Model training uses PyTorch (via
[skorch](https://skorch.readthedocs.io/), which gives PyTorch models a
sklearn-compatible `fit`/`predict_proba` interface) rather than the
project's original TensorFlow/Keras implementation. Active learning is a
small, pluggable query loop implemented directly in
[`jcim/active_learning.py`](jcim/active_learning.py), rather than a
dependency on the third-party `modAL` library (see below for why). Its
acquisition functions, in [`jcim/query_strategies.py`](jcim/query_strategies.py):

- `entropy_query` -- classic uncertainty sampling (Lewis & Gale, SIGIR 1994).
- `bald_query` -- Bayesian Active Learning by Disagreement via MC-Dropout
  (Gal, Islam & Ghahramani, "Deep Bayesian Active Learning with Image
  Data", ICML 2017).
- `core_set_query` -- greedy k-center diversity sampling in the model's
  learned embedding space (Sener & Savarese, "Active Learning for
  Convolutional Neural Networks: A Core-Set Approach", ICLR 2018).
- `direct_query` -- imbalance-aware separation-threshold sampling, adapted
  from DIRECT (Zhang, Katz-Samuels & Nowak, "Improved Algorithms for Deep
  Active Learning under Imbalance via Optimal Separation", ICML 2025,
  arXiv:2312.09196). See "Results And Analysis" below -- this one came out
  worst in our comparison, likely due to how the adaptation collapses
  their batch/multi-round algorithm into a single-point query.

(Margin sampling, another common baseline, was considered and skipped: for
binary classification it ranks samples identically to entropy sampling, so
it wouldn't add a distinct comparison point.)

### Why this pipeline previously stopped working

Two independent issues silently broke every run:

1. **`requirements.txt` listed the active-learning library as `modAL`.**
   PyPI package names are matched case-insensitively, and `modAL`
   normalizes to `modal` -- the unrelated Modal Labs cloud SDK. `pip
   install modAL` therefore installed the wrong package, and `from
   modAL.models import ActiveLearner` failed. The active-learning library
   this project actually needed shipped as `modAL-python`, and even that
   library's newer versions had changed its `ActiveLearner` API in ways
   that no longer matched how this project used it. Rather than depend on
   (and patch) that library for the ~15 lines of behavior actually used
   here -- fit, entropy-based query, teach -- that logic now lives directly
   in `jcim/active_learning.py`, with no external AL dependency at all.
2. **A hardcoded feature-vector size.** The neural network's input layer
   defaulted to `shape=2255`, matching whatever version of rdkit's
   descriptor list was installed at the time the code was written. Newer
   rdkit versions ship a different descriptor count, so the actual
   featurized data no longer matched the model's expected input shape. The
   model now infers its input shape from the training data instead.

A few smaller correctness bugs were also fixed along the way:

- `Validation` unpacked `confusion_matrix(...).ravel()` into 4 values,
  which crashes when a validation split (some splitters deliberately
  produce a near-empty "almost no validation" split) contains only one
  class; and the AUC confidence-interval calculation asserted both classes
  were present in the ground truth, also violated by that same split. Both
  now degrade to a `NaN` metric instead of crashing.
- A `numpy`-based DeLong-AUC helper used `np.float`/`np.bool`, which numpy
  has since removed.
- **Every result CSV had its "test" and "validation" columns swapped.**
  The code built each output row as `[iteration] + validation_values +
  test_values`, but the CSV header lists all `_test` columns before all
  `_validation` columns. So a column literally named `AUC_test` actually
  held the validation-set AUC, and vice versa, in every historical run
  under `Results/`. Row construction now matches the header order.
- `ALModel.__init__` (now `ActiveLearningModel`) called its own
  `initialize_al()` once directly, discarding the result, and then called
  `run()`, which called `initialize_al()` again -- training an initial
  model from scratch twice per run for no reason.
- `ActiveLearningModel` reported the **best** score across its whole query
  trajectory (dozens to hundreds of steps) as "the" performance, rather
  than the score of the model it actually finished training. Since the
  network is retrained from scratch at every step (no warm start), each
  step's score is an independent noisy draw -- taking the max over many
  of them is a look-elsewhere-biased estimate, not a real result. It now
  reports the final step's score, matching how every other model in this
  codebase reports one number for one trained model.
- **`numpy`/`torch` were seeded once at `models.py` import time**, not per
  run. Every process therefore replayed the exact same random sequence
  from its first call onward, which has two consequences: (1) the first
  iteration of any study is fully deterministic and identical across
  completely separate invocations of the pipeline, and (2) later
  iterations within one multi-iteration study "differ" only because their
  RNG state happens to have drifted from whatever random calls preceded
  them, not because they were independently seeded -- rerunning the same
  study reproduces the exact same sequence of "different" iterations every
  time. This understates true run-to-run variance and makes any
  measured spread across iterations hard to trust. `seed_everything(seed)`
  (`jcim/models.py`) is now called once per iteration, with `seed + i` for
  iteration `i` (see `--seed` above), so each iteration is an independent,
  reproducible draw and the whole study is exactly reproducible given the
  same base seed.

## Testing And CI

```bash
pip install -r requirements.txt pytest
pytest tests/
```

GitHub Actions (`.github/workflows/ci.yml`) runs this test suite on every
push/PR to `main`, then separately builds the Docker image and runs a
reduced end-to-end experiment inside it (`--iterations 1 --epochs 2
--max_queries 3`) as an integration smoke test.

## Results And Analysis

`Results/` holds both generated pipeline output and some older, hand-curated
studies (`Study_1/`, `Study_2/`, `Study_3/`, `N_SP1_TTS/`) kept for
historical reference, including analysis notebooks from before this
project's PyTorch/active-learning rewrite -- those predate the sampling
removal below and used study names in the old `<sampling>_<dataset>_<split>`
format. Running a fresh study writes a new subdirectory without touching
those:

```bash
docker build -t jcim .
docker run --rm -v $(pwd)/Results:/app/Results jcim \
  --study_name SF_ANV --iterations 3 --epochs 50 --max_queries 200 --overwrite
```

[`Results/N_SF_ANV_{entropy,bald,core_set,direct}/`](Results) in this repo
are the output of that command under the pipeline's previous (pre-sampling-removal)
CLI, once per `--al_strategy`: the `AlmostNoValidation` splitter, 3 repeated
iterations, 50 training epochs, 200 active-learning query rounds out of the
~914 real training examples in `Datasets/SCAMS_filtered.csv` (no synthetic
data anywhere in this comparison -- the synthetic fixtures under
`tests/conftest.py` are for unit tests only). A fresh run with the current
CLI would name that same combination `SF_ANV_{entropy,bald,core_set,direct}`.

### Can active learning identify a more informative subset than the full dataset?

The core question: does training only on the subset active learning
chooses to label do as well as -- or better than -- training on every
available example? Held-out test AUC, 3 iterations per condition:

| Training set | Examples used | Test AUC | vs. full-data |
|---|---|---|---|
| Full training pool | 914 (100%) | 0.817 ± 0.022 (n=12, pooled across the 4 runs below) | -- |
| AL subset -- Core-Set (diversity) | 210 (23%) | 0.785 ± 0.018 | 96.1% |
| AL subset -- BALD (MC-Dropout) | 210 (23%) | 0.779 ± 0.015 | 95.4% |
| AL subset -- entropy sampling | 210 (23%) | 0.754 ± 0.007 | 92.3% |
| AL subset -- DIRECT | 210 (23%) | 0.737 ± 0.018 | 90.3% |

**No strategy beats full-data training** -- unsurprising, since using every
available label is an upper bound for a fixed model class, and confirms
this is a real result rather than a data-loading bug. The practically
interesting number is how close a 23%-of-the-data subset gets: Core-Set
and BALD both land at 95-96% of full-data AUC; entropy and DIRECT trail
further behind.

**Which acquisition function matters -- and DIRECT is a genuine negative
result, not a win.** BALD and Core-Set each beat entropy sampling in all 3
of 3 iterations. DIRECT (see `jcim/query_strategies.py`'s `direct_query`,
adapted from Zhang, Katz-Samuels & Nowak, "Improved Algorithms for Deep
Active Learning under Imbalance via Optimal Separation", ICML 2025,
arXiv:2312.09196) came out *worst* of the four here, below even entropy.
Two likely reasons, both about the adaptation rather than the original
paper:

- DIRECT's actual algorithm is a multi-round, batch procedure: a
  version-space-shrinking search (their VReduce subroutine) spends part of
  each round's budget narrowing a candidate separation-threshold interval
  *before* annotating near it, refining the estimate over several rounds.
  Our active learner queries one point at a time and re-estimates the
  threshold from scratch on every call (`separation_threshold` in
  `query_strategies.py`) -- a much weaker, single-shot version of their
  core idea, since the threshold estimate never gets to accumulate the
  benefit of that iterative narrowing.
- Querying "near the current best guess of the boundary" is pure
  exploitation with no exploration mechanism. If the labeled set so far
  gives a noisy or wrong threshold estimate -- likely early on, with a
  from-scratch-retrained, poorly-calibrated model -- DIRECT keeps
  querying near that same wrong region rather than correcting course, unlike
  entropy/BALD (which re-derive their score from the model's actual
  current predictions across the whole pool each time) or Core-Set (which
  explicitly seeks out unrepresented regions).

A properly batched, multi-round DIRECT would need restructuring
`ActiveLearner`'s single-point query/teach loop into a batch acquisition
loop -- a real architectural change, not attempted here.

A separate sanity check (not part of the pipeline, see git history for the
one-off script) confirmed training on 210 *randomly* chosen examples lands
around the same AUC as entropy sampling's actively-selected 210 -- i.e.
plain entropy sampling here is barely distinguishable from picking points
at random, while BALD and Core-Set are not (and DIRECT, per the above, is
worse than random).

## Notes

- This repo is presented as a research engineering project, not a polished production service.
- Some code paths depend on cheminformatics tooling (rdkit) that ships prebuilt wheels for common platforms.
