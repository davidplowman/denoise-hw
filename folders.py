"""Locate dataset folders containing paired noisy.dng and gt.dng files."""

import os
from pathlib import Path


def find_folders(
    root: str = "../datasets/PI_RAW",
    filter: list[str] | None = None,
) -> list[str]:
    """Find all folders under *root* that contain both noisy.dng and gt.dng.

    Args:
        root: Root directory to search recursively.
        filter: Optional list of strings that must all appear somewhere in
                the folder path (case-sensitive).  A folder is included only
                when every string in the list matches.

    Returns:
        Sorted list of absolute folder paths that satisfy the criteria.
    """
    matches: list[str] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        lower = {f.lower() for f in filenames}
        if "noisy.dng" in lower and "gt.dng" in lower:
            folder = str(Path(dirpath).resolve())
            if filter is None or all(token in folder for token in filter):
                matches.append(folder)

    return sorted(matches)
