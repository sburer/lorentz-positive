#!/usr/bin/env python3
"""Check that every method row of a raw CSV was solved on the same instance.

A ``--resume --methods gurobi`` rerun that omits the generator flags silently
appends Gurobi rows for *different* random instances under the same
``(panel, n, m, seed)`` key.  This check fails when the rows of one instance
disagree on the generator or on ``data_scale``, or when a Gurobi bracket
excludes the exact SEP value of that instance.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "outputs" / "raw"

# (raw csv, key columns, generator columns, exact method)
PANELS = [
    (RAW / "bilinear_raw.csv", ("panel", "n", "m", "seed"),
     ("generator", "generator_alpha", "generator_sigma", "data_scale"), "Full SEP"),
    (RAW / "maxdist_raw.csv", ("panel", "n", "m", "seed"),
     ("generator", "condition", "center_scale"), None),
    (RAW / "maxdist_large_raw.csv", ("panel", "n", "m", "seed"),
     ("generator", "condition", "center_scale"), None),
    (RAW / "ttrs_generated_large_raw.csv", ("n", "seed"), ("generator", "loga_radius"), None),
]


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def purge_mismatched(path: Path, key_cols, gen_cols) -> int:
    """Drop rows whose generator columns disagree with the exact-method rows.

    The removed rows are Gurobi (or other) rows solved on a different random
    instance; after purging, rerun that method with ``--resume`` *and* the
    generator flags of the panel.  A copy of the original file is kept as
    ``<name>.before_purge`` (git-ignored).
    """

    rows = list(csv.DictReader(path.open(newline="")))
    fieldnames = list(rows[0].keys())
    gen_cols = [c for c in gen_cols if c in fieldnames]
    key_cols = [c for c in key_cols if c in fieldnames]
    by_key: dict[tuple, list[dict]] = {}
    for row in rows:
        by_key.setdefault(tuple(row[c] for c in key_cols), []).append(row)
    keep: list[dict] = []
    dropped = 0
    for group in by_key.values():
        # The conic methods are run first, so their (majority) generator
        # metadata identifies the instance.
        counts: dict[tuple, int] = {}
        for row in group:
            sig = tuple(row.get(c, "") for c in gen_cols)
            counts[sig] = counts.get(sig, 0) + 1
        majority = max(counts, key=counts.get)
        for row in group:
            if tuple(row.get(c, "") for c in gen_cols) == majority:
                keep.append(row)
            else:
                dropped += 1
    if dropped:
        path.with_suffix(path.suffix + ".before_purge").write_text(path.read_text())
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(keep)
    return dropped


def check_panel(path: Path, key_cols, gen_cols, exact_method) -> list[str]:
    if not path.exists():
        return [f"{path.name}: missing"]
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        return [f"{path.name}: empty"]
    gen_cols = [c for c in gen_cols if c in rows[0]]
    key_cols = [c for c in key_cols if c in rows[0]]
    problems: list[str] = []
    by_key: dict[tuple, list[dict]] = {}
    for row in rows:
        by_key.setdefault(tuple(row[c] for c in key_cols), []).append(row)
    for key, group in by_key.items():
        label = f"{path.name} {dict(zip(key_cols, key))}"
        for col in gen_cols:
            values = {row.get(col, "") for row in group}
            if len(values) > 1:
                problems.append(
                    f"{label}: {col} differs across methods: "
                    + ", ".join(f"{r['method']}={r.get(col, '')!r}" for r in group)
                )
        if exact_method is not None:
            exact = next((r for r in group if r["method"] == exact_method), None)
            gurobi = next((r for r in group if r["method"] == "Gurobi"), None)
            if exact and gurobi:
                z = _float(exact.get("lower_bound"))
                lb, ub = _float(gurobi.get("lower_bound")), _float(gurobi.get("upper_bound"))
                if None not in (z, lb, ub):
                    tol = 1e-5 * max(1.0, abs(z))  # Gurobi MIPGap is 1e-6
                    if not (lb - tol <= z <= ub + tol):
                        problems.append(
                            f"{label}: Gurobi bracket [{lb:.6f}, {ub:.6f}] excludes "
                            f"{exact_method} value {z:.6f} (different instance?)"
                        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--purge",
        action="store_true",
        help="remove rows whose generator metadata disagrees with the other "
        "methods of the same instance, then re-check",
    )
    args = parser.parse_args(argv)
    problems: list[str] = []
    for path, key_cols, gen_cols, exact_method in PANELS:
        if args.purge and path.exists():
            dropped = purge_mismatched(path, key_cols, gen_cols)
            if dropped:
                print(f"{path.name}: purged {dropped} mismatched row(s)")
        found = check_panel(path, key_cols, gen_cols, exact_method)
        print(f"{path.name}: {'ok' if not found else f'{len(found)} problem(s)'}")
        problems.extend(found)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        print("\nRaw provenance check FAILED.")
        return 1
    print("\nRaw provenance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
