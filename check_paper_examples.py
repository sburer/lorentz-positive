#!/usr/bin/env python3
"""Check the named numerical examples of the manuscript against their sources.

For every ``[[examples]]`` entry in ``paper_experiments.toml`` this script
takes each ``required_values`` item (``"name = number"``) and verifies that

* the number appears verbatim in ``paper/Lorentz.tex``; and
* some number in the example's source (the verification log or raw CSV)
  rounds to it at the printed precision.

Percentages (``"... = 15.74 percent"``) are checked in the manuscript only.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Exit status meaning "the manuscript is not available here, nothing checked".
# reproduce.py reports these separately from failures.
SKIPPED = 3

VALUE_RE = re.compile(r"^(?P<name>.*?)\s*=\s*(?P<number>-?\d+(?:\.\d+)?)\s*(?P<unit>percent)?\s*$")
NUMBER_RE = re.compile(r"-?\d+\.\d+")


def _split_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def source_numbers(paths: list[Path]) -> list[float]:
    numbers: list[float] = []
    for path in paths:
        numbers.extend(float(tok) for tok in NUMBER_RE.findall(path.read_text()))
    return numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=ROOT / "paper_experiments.toml")
    parser.add_argument(
        "--manuscript",
        default=None,
        help="Path to Lorentz.tex (default: the manuscript entry in the inventory).",
    )
    args = parser.parse_args(argv)

    with Path(args.inventory).open("rb") as f:
        inventory = tomllib.load(f)
    manuscript = Path(args.manuscript or ROOT / inventory["manuscript"])
    if not manuscript.exists():
        print(f"manuscript not found: {manuscript}")
        print("Skipping this check; pass --manuscript to point at Lorentz.tex.")
        return SKIPPED
    paper_text = manuscript.read_text()

    problems: list[str] = []
    checked = 0
    for example in inventory.get("examples", []):
        ident = example["id"]
        sources = [ROOT / p for p in _split_sources(example.get("source_raw"))]
        missing = [p for p in sources if not p.exists()]
        if missing:
            problems.append(f"{ident}: missing source {missing}")
            continue
        numbers = source_numbers(sources)
        for item in example.get("required_values", []):
            match = VALUE_RE.match(item)
            if match is None:
                problems.append(f"{ident}: cannot parse required value {item!r}")
                continue
            checked += 1
            text = match.group("number")
            if text not in paper_text:
                problems.append(f"{ident}: {item!r} not found in {manuscript.name}")
            if match.group("unit"):
                continue
            decimals = len(text.split(".")[1]) if "." in text else 0
            if not any(f"{value:.{decimals}f}" == text for value in numbers):
                problems.append(
                    f"{ident}: {item!r} not reproduced by "
                    + ", ".join(str(p.relative_to(ROOT)) for p in sources)
                )

    print(f"manuscript: {manuscript}")
    print(f"example values checked: {checked}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        print("\nPaper example check FAILED.")
        return 1
    print("\nPaper example check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
