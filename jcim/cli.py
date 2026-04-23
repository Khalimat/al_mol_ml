import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    return parser


def main():
    args = build_parser().parse_args()
    from .pipeline import SCAMsPipeline

    SCAMsPipeline(args.study_name, args.datasets_path)
