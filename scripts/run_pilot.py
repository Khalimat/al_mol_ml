"""Pilot run for the generalization study (docs/generalization_study_design.md,
section 5, phases 3-4): validates the extended pipeline -- new datasets,
all three architecture backends, active learning -- end to end, at a
reduced iteration count and query budget, before committing to the full
10-iteration matrix. Meant to run on a GPU (an H100 cluster, in
particular): the GNN backend and the larger new datasets (Tox21, the HIV
subsample) are meaningfully slower on CPU.

Prerequisite: the new datasets must already exist under Datasets/ --
run scripts/prepare_new_datasets.py first if they don't.

Usage (from the repo root):
    uv run python scripts/run_pilot.py
    uv run python scripts/run_pilot.py --iterations 5 --epochs 50
    uv run python scripts/run_pilot.py --datasets BBBP CLINTOX --architectures mlp rf
    uv run python scripts/run_pilot.py --max_queries -1   # uncap query budget (full pool)

Each (dataset, architecture, strategy) combination is one independent
SCAMsPipeline call, written to Results/pilot/<dataset>_SS_<architecture>_
<strategy>/. A failure in one combination is caught, logged, and skipped --
not fatal to the rest of the matrix, since a partial pilot run is still
useful information (e.g. "the GNN backend works on BBBP and ClinTox but
OOMs on the HIV subsample" is exactly the kind of thing this pilot is
for). Writes a summary CSV (Results/pilot/pilot_summary.csv) of test-set
AUC/MCC (mean +/- std across iterations) per combination at the end.
"""

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from almolml.pipeline import SCAMsPipeline  # noqa: E402

# The 5 new datasets from the generalization study -- SCAM itself is
# already covered by the existing per-strategy studies under Results/. See
# docs/generalization_study_design.md section 2 for what each one tests.
DEFAULT_DATASETS = ["BBBP", "CLINTOX", "TOX21NRAR", "TOX21SRP53", "HIVSUB"]
DEFAULT_ARCHITECTURES = ["mlp", "rf", "gnn"]
# The no-AL baseline is always in every run's own TF_ML_non_AL.csv output,
# so these are only the *active-learning* strategies to compare it against:
# entropy (classic uncertainty sampling) and batched DIRECT (the
# strongest performer -- directionally -- in the existing SCAM batch-
# strategy results; see the main README). Not the full 7-strategy list --
# see design doc section 5's phased rollout for why.
DEFAULT_STRATEGIES = ["entropy", "direct_batch"]
# Scaffold split only (design doc section 3): most realistic for
# prospective drug discovery, and the split most likely to reveal
# architecture differences.
SPLIT = "SS"
BATCH_SIZE = 5
PILOT_RESULTS_DIR = REPO_ROOT / "Results" / "pilot"


def run_one(dataset, architecture, strategy, iterations, epochs, max_queries, datasets_path, seed):
    study_name = f"{dataset}_{SPLIT}"
    batch_size = BATCH_SIZE if strategy.endswith("_batch") else 1

    pipeline = SCAMsPipeline(
        study_name,
        datasets_path,
        iterations=iterations,
        epochs=epochs,
        max_queries=max_queries,
        al_strategy=strategy,
        batch_size=batch_size,
        architecture=architecture,
        overwrite=True,
        seed=seed,
    )

    # SCAMsPipeline always writes to Results/<study_name>/, regardless of
    # architecture/strategy -- move it somewhere combo-specific immediately,
    # so the next combo on the same (dataset, split) doesn't clobber it.
    combo_dir = PILOT_RESULTS_DIR / f"{study_name}_{architecture}_{strategy}"
    if combo_dir.exists():
        shutil.rmtree(combo_dir)
    shutil.move(str(pipeline.results_dir), str(combo_dir))
    return combo_dir


def _summarize(combo_dir, dataset, architecture, strategy, elapsed_s):
    al = pd.read_csv(combo_dir / "TF_ML_AL.csv", index_col=0)
    baseline = pd.read_csv(combo_dir / "TF_ML_non_AL.csv", index_col=0)
    return {
        "dataset": dataset,
        "split": SPLIT,
        "architecture": architecture,
        "strategy": strategy,
        "n_iterations": len(al),
        "baseline_AUC_test_mean": baseline["AUC_test"].mean(),
        "baseline_AUC_test_std": baseline["AUC_test"].std(),
        "baseline_MCC_test_mean": baseline["MCC_test"].mean(),
        "al_AUC_test_mean": al["AUC_test"].mean(),
        "al_AUC_test_std": al["AUC_test"].std(),
        "al_MCC_test_mean": al["MCC_test"].mean(),
        "elapsed_seconds": round(elapsed_s, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--max_queries",
        type=int,
        default=300,
        help="Cap on active-learning query rounds per iteration (default: 300, "
        "a deliberate pilot-scope limit -- see module docstring). Pass a "
        "value <= 0 to use the full pool instead, matching the final study.",
    )
    parser.add_argument("--datasets_path", default=str(REPO_ROOT / "Datasets"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_queries = None if args.max_queries is not None and args.max_queries <= 0 else args.max_queries

    PILOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    combos = [
        (dataset, architecture, strategy)
        for dataset in args.datasets
        for architecture in args.architectures
        for strategy in args.strategies
    ]
    print(f"Pilot run: {len(combos)} (dataset, architecture, strategy) combinations")

    for i, (dataset, architecture, strategy) in enumerate(combos, 1):
        label = f"[{i}/{len(combos)}] {dataset}_{SPLIT} / {architecture} / {strategy}"
        print(f"\n=== {label} ===")
        start = time.time()
        try:
            combo_dir = run_one(
                dataset,
                architecture,
                strategy,
                args.iterations,
                args.epochs,
                max_queries,
                args.datasets_path,
                args.seed,
            )
            elapsed = time.time() - start
            row = _summarize(combo_dir, dataset, architecture, strategy, elapsed)
            summary_rows.append(row)
            print(
                f"OK ({elapsed:.0f}s): baseline AUC {row['baseline_AUC_test_mean']:.3f}, "
                f"AL AUC {row['al_AUC_test_mean']:.3f}"
            )
        except Exception:
            elapsed = time.time() - start
            print(f"FAILED after {elapsed:.0f}s:")
            traceback.print_exc()
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": SPLIT,
                    "architecture": architecture,
                    "strategy": strategy,
                    "n_iterations": 0,
                    "elapsed_seconds": round(elapsed, 1),
                    "error": "failed -- see stdout traceback above",
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = PILOT_RESULTS_DIR / "pilot_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote summary of {len(summary)} combinations to {summary_path}")
    n_failed = summary["error"].notna().sum() if "error" in summary.columns else 0
    if n_failed:
        print(f"{n_failed} combination(s) failed -- see the 'error' column / stdout above.")


if __name__ == "__main__":
    main()
