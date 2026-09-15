#!/usr/bin/env python3
"""Check generated LaTeX fragments against the paper table inventory.

This is stricter than ``check_paper_inventory.py``.  The inventory checker asks
whether the manuscript and manifest agree.  This checker asks whether the
current generated table fragments are aligned with that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path


LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def _split_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def load_inventory(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def labels_in_file(path: Path) -> set[str]:
    return set(LABEL_RE.findall(path.read_text()))


def check_fragments(repo_root: Path, inventory_path: Path) -> int:
    inventory = load_inventory(inventory_path)
    manuscript = inventory["manuscript"]
    tables = inventory.get("tables", [])

    expected_by_source: dict[str, list[str]] = defaultdict(list)
    for table in tables:
        source_table = table.get("source_table")
        if not source_table or source_table == manuscript:
            continue
        expected_by_source[source_table].append(table["label"])

    errors: list[str] = []
    warnings: list[str] = []

    for source_table, expected_labels in sorted(expected_by_source.items()):
        path = repo_root / source_table
        if not path.exists():
            errors.append(f"missing table fragment: {source_table}")
            continue

        found_labels = labels_in_file(path)
        for label in expected_labels:
            if label not in found_labels:
                errors.append(f"{source_table} does not contain expected label {label}")

        extra_tab_labels = sorted(
            label
            for label in found_labels
            if label.startswith("tab:") and label not in expected_labels
        )
        if extra_tab_labels:
            warnings.append(
                f"{source_table} contains non-manifest table labels: "
                + ", ".join(extra_tab_labels)
            )

    embedded = [
        table["label"]
        for table in tables
        if table.get("source_table") == manuscript
    ]

    print(f"inventory: {inventory_path}")
    print("generated fragment sources:")
    for source_table, expected_labels in sorted(expected_by_source.items()):
        print(f"  {source_table}: {', '.join(expected_labels)}")
    if embedded:
        print("embedded manuscript tables without generated fragments yet:")
        for label in embedded:
            print(f"  {label}")

    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nerrors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nTable fragment check passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        default="paper_experiments.toml",
        help="Path to the paper experiment inventory.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    return check_fragments(repo_root, repo_root / args.inventory)


if __name__ == "__main__":
    sys.exit(main())
