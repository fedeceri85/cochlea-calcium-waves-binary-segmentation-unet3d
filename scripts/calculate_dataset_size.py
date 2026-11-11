"""Calculate the on-disk size of images and labels listed in dfdata.csv."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


BYTES_PER_GB = 1_000_000_000
BYTES_PER_GIB = 1024**3
MAC_USERS_PREFIX = "/Volumes/Data/Users"


def resolve_data_path(value: str, windows_users_root: str) -> Path:
    """Resolve paths using the same Windows drive mapping as train_binary.py."""
    value = value.strip()
    if os.name == "nt" and (
        value == MAC_USERS_PREFIX or value.startswith(f"{MAC_USERS_PREFIX}/")
    ):
        suffix = value[len(MAC_USERS_PREFIX) :].lstrip("/")
        return Path(windows_users_root) / Path(suffix)
    return Path(value)


def path_size(path: Path) -> int:
    """Return the size of a file or all files below a directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(path)


def format_size(size: int) -> str:
    return f"{size / BYTES_PER_GB:.3f} GB ({size / BYTES_PER_GIB:.3f} GiB)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum the sizes of image and label paths referenced by dfdata.csv."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/dfdata.csv",
        type=Path,
        help="Path to dfdata.csv (default: data/dfdata.csv)",
    )
    parser.add_argument(
        "--windows-users-root",
        default="Z:/Users",
        help=(
            "Windows replacement for /Volumes/Data/Users "
            "(default: Z:/Users)"
        ),
    )
    args = parser.parse_args()

    image_total = 0
    label_total = 0
    missing: list[tuple[int, str, Path]] = []
    seen: set[Path] = set()
    duplicates = 0

    with args.csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"image", "label"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            parser.error(
                f"CSV is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        for row_number, row in enumerate(reader, start=2):
            for column in ("image", "label"):
                path = resolve_data_path(
                    row[column] or "", args.windows_users_root
                ).resolve()
                if path in seen:
                    duplicates += 1
                    continue
                seen.add(path)

                try:
                    size = path_size(path)
                except (FileNotFoundError, OSError):
                    missing.append((row_number, column, path))
                    continue

                if column == "image":
                    image_total += size
                else:
                    label_total += size

            cumulative = image_total + label_total
            print(f"Row {row_number}: cumulative {format_size(cumulative)}")

    print("\nSummary")
    print(f"Images: {format_size(image_total)}")
    print(f"Labels: {format_size(label_total)}")
    print(f"Total:  {format_size(image_total + label_total)}")
    print(f"Unique paths measured: {len(seen) - len(missing)}")
    print(f"Duplicate references skipped: {duplicates}")
    print(f"Missing/unreadable paths: {len(missing)}")

    for row_number, column, path in missing:
        print(f"  row {row_number}, {column}: {path}")


if __name__ == "__main__":
    main()
