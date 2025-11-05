"""Utility to copy the provided dataset into the expected location."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from houseprice.config import DATA_PATH as DEFAULT_DATA_PATH
except ModuleNotFoundError:
    DEFAULT_DATA_PATH = Path("data/kc_house_data.csv")


def copy_dataset(source: Path, destination: Path, overwrite: bool = False) -> Path:
    """Copy the dataset to the project data directory.

    Parameters
    ----------
    source
        Path to the dataset provided by the user.
    destination
        Target path inside the repository where the pipeline expects the data.
    overwrite
        If ``True`` and ``destination`` already exists, replace it. Otherwise a
        ``FileExistsError`` is raised to prevent accidental overwrites.
    """

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Use --force to overwrite."
        )

    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the dataset file provided by the user.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Destination path inside the repository. Defaults to the configured "
            "DATA_PATH (data/kc_house_data.csv)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )

    args = parser.parse_args()

    copied_to = copy_dataset(args.source, args.destination, overwrite=args.force)
    print(f"Dataset copied to {copied_to}")


if __name__ == "__main__":
    main()
