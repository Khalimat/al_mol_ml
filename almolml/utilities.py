"""Small, general-purpose helpers used across the pipeline: CLI argument
parsing, filesystem setup, and feeding a Dataset into a splitter.
"""

import argparse
import sys
from pathlib import Path


def dataset_to_splitter(splitter, dataset, test_split_r=None):
    """Run `dataset` through `splitter`, returning the fitted splitter
    (which exposes X_train/Y_train/X_test/Y_test).

    `test_split_r=None` (default) lets `splitter` use its own class
    default (`BaseSplitter.__init__`'s `test_split_r=0.3`) rather than
    forcing a specific ratio here.
    """
    kwargs = {} if test_split_r is None else {"test_split_r": test_split_r}
    return splitter(
        dataset.X,
        dataset.Y,
        dataset.SMILES,
        dataset.dataset.mols,
        dataset.ID_name,
        **kwargs,
    )


def str2bool(v):
    """Parse a yes/no-ish string (or pass through an existing bool)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def rm_tree(path):
    """Recursively delete `path` and everything under it."""
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def prepare_results_dir(path, overwrite, allow_rename=False):
    """
    Ensure `path` exists and is empty, deciding how to handle a pre-existing
    directory without ever blocking on input() in a non-interactive context
    (e.g. CI), where stdin isn't a TTY.

    Parameters
    ----------
    path: Path, directory to create
    overwrite: bool, if True delete an existing directory without prompting
    allow_rename: bool, if True and running interactively, offer the user a
        chance to pick a new name instead of deleting the existing directory

    Returns
    -------
    Path, the directory that was created (equal to `path` unless the user
    picked a new name interactively)
    """
    if path.is_dir():
        if overwrite:
            rm_tree(path)
        elif not sys.stdin.isatty():
            raise FileExistsError(
                "Results directory {} already exists. Re-run with --overwrite "
                "to replace it.".format(path)
            )
        else:
            out_srt = input(
                "The path exists. Do you want to delete the existing path and create a "
                "new? Please, enter yes or no: "
            )
            if str2bool(out_srt):
                rm_tree(path)
            elif allow_rename:
                new_study = input("Please, enter new name: ")
                path = path.parent / new_study
            else:
                raise FileExistsError(
                    "Results directory {} already exists.".format(path)
                )
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path, file_name):
    """Return `path / file_name` if it exists, else raise FileNotFoundError."""
    full_file_path = Path(path) / file_name
    if not full_file_path.is_file():
        raise FileNotFoundError(
            "File {} not found in location {}. Please enter a valid path and "
            "file name".format(file_name, path)
        )
    return full_file_path


perf_columns = [
    "AUC lower estimate",
    "AUC",
    "AUC upper estimate",
    "accuracy",
    "F1",
    "MCC",
]
