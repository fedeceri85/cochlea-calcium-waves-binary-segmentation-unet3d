"""Copy dataset paths from dfdata.csv while preserving their folder tree."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


MAC_SOURCE_ROOT = Path("/Volumes/Data/Users")
WINDOWS_SOURCE_ROOT = Path("Z:/Users")


def default_source_root() -> Path:
    return WINDOWS_SOURCE_ROOT if os.name == "nt" else MAC_SOURCE_ROOT


def source_path(csv_value: str, source_root: Path) -> Path:
    """Map a dfdata macOS path to the selected source root."""
    value = csv_value.strip().replace("\\", "/")
    mac_prefix = MAC_SOURCE_ROOT.as_posix()
    if value == mac_prefix:
        relative = Path()
    elif value.startswith(f"{mac_prefix}/"):
        relative = Path(value[len(mac_prefix) + 1 :])
    else:
        raise ValueError(
            f"Path is outside {mac_prefix!r} and cannot be mapped safely: {csv_value}"
        )
    return source_root / relative


def destination_path(csv_value: str, destination_root: Path) -> Path:
    """Recreate the path below /Volumes/Data/Users at destination_root."""
    value = csv_value.strip().replace("\\", "/")
    mac_prefix = MAC_SOURCE_ROOT.as_posix()
    if value == mac_prefix:
        relative = Path()
    elif value.startswith(f"{mac_prefix}/"):
        relative = Path(value[len(mac_prefix) + 1 :])
    else:
        raise ValueError(
            f"Path is outside {mac_prefix!r} and cannot be mapped safely: {csv_value}"
        )
    return destination_root / relative


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy paths referenced by dfdata.csv and recreate their directory tree "
            "below a new root."
        )
    )
    parser.add_argument("destination_root", type=Path, help="Local destination root")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/dfdata.csv"),
        help="Input CSV (default: data/dfdata.csv)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_source_root(),
        help=(
            "Root replacing /Volumes/Data/Users in CSV paths "
            f"(default: {default_source_root()})"
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["image", "label", "waves"],
        help="CSV path columns to copy (default: image label waves)",
    )
    parser.add_argument(
        "--skip-every",
        type=int,
        default=0,
        metavar="X",
        help=(
            "After each selected CSV row, skip the next X rows; "
            "0 disables skipping (default: 0)"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip destinations that already exist (default). With "
            "--no-skip-existing, stop instead of overwriting them."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without copying anything",
    )
    args = parser.parse_args()

    if args.skip_every < 0:
        parser.error("--skip-every must be zero or a positive integer")

    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    if source_root == destination_root:
        parser.error("Source and destination roots must be different")

    copied = 0
    skipped_interval = 0
    skipped_existing = 0
    missing = 0
    seen: set[Path] = set()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(args.columns).difference(reader.fieldnames or [])
        if missing_columns:
            parser.error(
                f"CSV is missing columns: {', '.join(sorted(missing_columns))}"
            )

        for data_index, row in enumerate(reader, start=1):
            row_number = data_index + 1
            # Select one complete dataset row, then skip X rows so image and
            # label paths from a selected row always remain together.
            if args.skip_every and (data_index - 1) % (args.skip_every + 1) != 0:
                print(f"SKIP interval: row {row_number}")
                skipped_interval += 1
                continue

            for column in args.columns:
                value = (row[column] or "").strip()
                if not value:
                    print(f"MISSING row {row_number} {column}: empty path")
                    missing += 1
                    continue

                try:
                    source = source_path(value, source_root).resolve()
                    destination = destination_path(value, destination_root).resolve()
                except ValueError as error:
                    parser.error(f"row {row_number}, column {column}: {error}")

                if source in seen:
                    continue
                seen.add(source)

                if not source.exists():
                    print(f"MISSING: {source}")
                    missing += 1
                    continue

                if destination.exists():
                    if args.skip_existing:
                        print(f"SKIP existing: {destination}")
                        skipped_existing += 1
                        continue
                    raise FileExistsError(
                        f"Destination already exists: {destination}. Use "
                        "--skip-existing or remove it deliberately before retrying."
                    )

                print(f"COPY {source} -> {destination}")
                if not args.dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        shutil.copytree(source, destination, copy_function=shutil.copy2)
                    else:
                        shutil.copy2(source, destination)
                copied += 1

    action = "Would copy" if args.dry_run else "Copied"
    print("\nSummary")
    print(f"{action}: {copied}")
    print(f"Skipped by interval: {skipped_interval}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
