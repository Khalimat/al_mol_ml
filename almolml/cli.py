import argparse

from .paths import REPO_ROOT


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the SCAM classification benchmark pipeline."
    )
    parser.add_argument("-s_n", "--study_name", required=True, help="Study name")
    parser.add_argument(
        "-d_p",
        "--datasets_path",
        required=False,
        default=str(REPO_ROOT / "Datasets"),
        help="Path to the datasets directory",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of repeated train/evaluate runs for the study (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs for the neural network models (default: 50)",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Cap on the number of active-learning query rounds per iteration. "
        "Defaults to using the entire training pool.",
    )
    parser.add_argument(
        "--al_strategy",
        choices=["entropy", "bald", "core_set", "direct"],
        default="entropy",
        help="Active-learning query strategy: 'entropy' (uncertainty sampling), "
        "'bald' (Bayesian Active Learning by Disagreement via MC-Dropout), "
        "'core_set' (greedy k-center diversity sampling), or 'direct' "
        "(imbalance-aware separation-threshold sampling, adapted from DIRECT, "
        "ICML 2025). Default: entropy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing results directory for this study without prompting. "
        "Required for non-interactive runs (e.g. CI) targeting an existing study name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Iteration i of the run is seeded with seed + i, so "
        "each iteration is an independent, reproducible draw rather than inheriting "
        "whatever RNG state prior calls happened to leave behind (default: 0).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    from .pipeline import SCAMsPipeline

    SCAMsPipeline(
        args.study_name,
        args.datasets_path,
        iterations=args.iterations,
        epochs=args.epochs,
        max_queries=args.max_queries,
        al_strategy=args.al_strategy,
        overwrite=args.overwrite,
        seed=args.seed,
    )
