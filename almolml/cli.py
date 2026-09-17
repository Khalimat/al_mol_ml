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
        choices=[
            "entropy",
            "bald",
            "core_set",
            "direct",
            "bald_batch",
            "core_set_batch",
            "direct_batch",
        ],
        default="entropy",
        help="Active-learning query strategy: 'entropy' (uncertainty sampling), "
        "'bald' (Bayesian Active Learning by Disagreement via MC-Dropout), "
        "'core_set' (greedy k-center diversity sampling), 'direct' "
        "(imbalance-aware separation-threshold sampling, adapted from DIRECT, "
        "ICML 2025), or the batch versions 'bald_batch' (BatchBALD, NeurIPS "
        "2019), 'core_set_batch' (the paper's actual batch algorithm), and "
        "'direct_batch' (queries straddling the threshold from both sides). "
        "The batch strategies require --batch_size > 1 to have any effect "
        "over their single-point counterparts. Default: entropy.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of pool points to query and label per round (default: 1, "
        "the original point-at-a-time loop). Only meaningful with a batch "
        "--al_strategy (bald_batch, core_set_batch, direct_batch); passing "
        "> 1 with a single-point strategy raises an error.",
    )
    parser.add_argument(
        "--architecture",
        choices=["mlp", "rf", "gnn"],
        default="mlp",
        help="Model architecture backend (see almolml/model_backends.py): "
        "'mlp' (default, the original torch MLP on Morgan fingerprint + "
        "RDKit descriptors), 'rf' (Random Forest on the same features), or "
        "'gnn' (a graph neural network over the molecular graph directly -- "
        "requires the 'graph' extra, e.g. `uv sync --extra cpu --extra "
        "graph`). Applies to both the non-active-learning baseline "
        "(TorchMLPModel) and the active-learning model.",
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
    parser.add_argument(
        "--test_split_r",
        type=float,
        default=0.2,
        help="Only used for a dataset with no separately curated external test "
        "file (every dataset added by the generalization study -- see "
        "docs/generalization_study_design.md): the fraction of the whole "
        "dataset held out as the test set before the train/validation split "
        "(default: 0.2). Ignored for the original SCAM datasets, which use "
        "test_DLS.csv instead.",
    )
    parser.add_argument(
        "--validation_split_r",
        type=float,
        default=0.3,
        help="Only used for a dataset with no separately curated external test "
        "file: the fraction of the remaining (non-test) data held out as the "
        "validation set (default: 0.3, matching the splitters' own default "
        "test_split_r). Ignored for the original SCAM datasets.",
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
        batch_size=args.batch_size,
        architecture=args.architecture,
        overwrite=args.overwrite,
        test_split_r=args.test_split_r,
        validation_split_r=args.validation_split_r,
        seed=args.seed,
    )
