#!/usr/bin/env python3
"""Check that the paper-code inventory matches the manuscript.

The inventory in ``paper_experiments.toml`` is meant to be the paper-facing
contract.  This script catches simple drift: missing table labels, unexpected
table counts, and referenced source files that are not present yet.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


TABLE_LABEL_RE = re.compile(r"\\label\{(tab:[^}]+)\}")

# Exit status meaning "the manuscript is not available here, so only the
# file-existence part of the check ran".  reproduce.py reports these
# separately from failures.
SKIPPED = 3


def _split_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def load_inventory(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def manuscript_table_labels(path: Path) -> list[str]:
    text = path.read_text()
    return TABLE_LABEL_RE.findall(text)


def check_inventory(
    repo_root: Path, inventory_path: Path, manuscript: Path | None = None
) -> int:
    inventory = load_inventory(inventory_path)
    manuscript_path = manuscript or repo_root / inventory["manuscript"]
    have_manuscript = manuscript_path.exists()
    labels_in_manuscript = (
        manuscript_table_labels(manuscript_path) if have_manuscript else []
    )
    manifest_tables = inventory.get("tables", [])
    labels_in_manifest = [table["label"] for table in manifest_tables]

    errors: list[str] = []
    warnings: list[str] = []

    expected_count = inventory.get("table_count")
    if expected_count is not None and len(labels_in_manifest) != expected_count:
        errors.append(
            f"manifest lists {len(labels_in_manifest)} tables, expected {expected_count}"
        )
    if have_manuscript and expected_count is not None and len(labels_in_manuscript) != expected_count:
        errors.append(
            f"manuscript has {len(labels_in_manuscript)} table labels, expected {expected_count}"
        )

    missing_from_manuscript = [
        label for label in labels_in_manifest if label not in labels_in_manuscript
    ]
    missing_from_manifest = [
        label for label in labels_in_manuscript if label not in labels_in_manifest
    ]
    if have_manuscript and missing_from_manuscript:
        errors.append(
            "table labels in manifest but not manuscript: "
            + ", ".join(missing_from_manuscript)
        )
    if missing_from_manifest:
        errors.append(
            "table labels in manuscript but not manifest: "
            + ", ".join(missing_from_manifest)
        )

    if len(labels_in_manifest) != len(set(labels_in_manifest)):
        errors.append("manifest has duplicate table labels")
    if len(labels_in_manuscript) != len(set(labels_in_manuscript)):
        errors.append("manuscript has duplicate table labels")

    for table in manifest_tables:
        for key in ("source_raw", "source_seeds", "source_table"):
            for rel_path in _split_sources(table.get(key)):
                if not (repo_root / rel_path).exists():
                    warnings.append(
                        f"{table['label']} references missing {key}: {rel_path}"
                    )

    for example in inventory.get("examples", []):
        for key in ("source_script", "source_note", "source_raw"):
            for rel_path in _split_sources(example.get(key)):
                if not (repo_root / rel_path).exists():
                    warnings.append(
                        f"{example['id']} references missing {key}: {rel_path}"
                    )

    print(f"inventory: {inventory_path}")
    print(f"manuscript: {manuscript_path}" + ("" if have_manuscript else " (not found)"))
    print(f"tables in manifest: {len(labels_in_manifest)}")
    if have_manuscript:
        print(f"tables in manuscript: {len(labels_in_manuscript)}")
    print("labels:")
    for label in labels_in_manifest:
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

    if not have_manuscript:
        print("\nInventory files present; manuscript label comparison skipped.")
        print("Pass --manuscript to point at Lorentz.tex.")
        return SKIPPED
    print("\nInventory check passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        default="paper_experiments.toml",
        help="Path to the paper experiment inventory.",
    )
    parser.add_argument(
        "--manuscript",
        default=None,
        help="Path to Lorentz.tex (default: the manuscript entry in the inventory).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    manuscript = Path(args.manuscript) if args.manuscript else None
    return check_inventory(repo_root, repo_root / args.inventory, manuscript)


if __name__ == "__main__":
    sys.exit(main())
