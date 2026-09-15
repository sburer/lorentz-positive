#!/usr/bin/env python3
"""Compare the numbers in the manuscript tables with the generated fragments.

``check_paper_inventory.py`` checks that the labels line up and
``check_table_fragments.py`` checks that the fragments match the manifest.
This script closes the loop: for every ``tab:*`` label in
``paper_experiments.toml`` it extracts the ``tabular`` body from
``paper/Lorentz.tex`` and from the generated fragment, and compares the two
row by row, number by number.

The manuscript tables are hand-pasted and the manuscript restyles headers, rules, and
column labels, so this is deliberately a *numeric* comparison: the first cell
of each data row is the row key, every other cell is reduced to the list of
numbers it contains, and those lists must agree exactly as printed.  Header
rows are compared only by column count.
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

TABULAR_RE = re.compile(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", re.S)
RULE_TOKENS = ("\\toprule", "\\midrule", "\\bottomrule", "\\hline", "\\cline")
# Plain decimals, scientific notation, and the LaTeX form 1.2\times10^{-4}.
NUMBER_RE = re.compile(
    r"-?\d+(?:\.\d+)?(?:\s*\\times\s*10\^\{-?\d+\}|e-?\d+)?"
)


def _split_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def tabular_after_label(text: str, label: str) -> str | None:
    """Return the first tabular body that follows ``\\label{label}``."""

    marker = f"\\label{{{label}}}"
    start = text.find(marker)
    if start < 0:
        return None
    # Some tables put the label after the caption but before the tabular; a
    # table could also put the label after the tabular, so search backwards
    # to the enclosing table environment first.
    env_start = text.rfind("\\begin{table", 0, start)
    if env_start < 0:
        env_start = start
    match = TABULAR_RE.search(text, env_start)
    return match.group(1) if match else None


def normalize_number(token: str) -> str:
    token = token.replace(" ", "")
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}", token)
    if match:
        return f"{match.group(1)}e{int(match.group(2))}"
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)e(-?\d+)", token)
    if match:
        return f"{match.group(1)}e{int(match.group(2))}"
    return token


def normalize_key(cell: str) -> str:
    cell = re.sub(r"\\(times|rm|textrm|text|mathrm)\b", "", cell)
    cell = re.sub(r"[\s${}~]", "", cell)
    return cell.replace("\\", "")


def parse_rows(body: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in body.split("\\\\"):
        line = raw
        for token in RULE_TOKENS:
            line = line.replace(token, "")
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split("&")])
    return rows


def row_numbers(cells: list[str]) -> list[list[str]]:
    return [
        [normalize_number(m.group(0)) for m in NUMBER_RE.finditer(cell)]
        for cell in cells
    ]


def compare_table(label: str, paper_body: str, frag_body: str) -> list[str]:
    problems: list[str] = []
    paper_rows = parse_rows(paper_body)
    frag_rows = parse_rows(frag_body)
    if not paper_rows or not frag_rows:
        return [f"{label}: empty tabular body"]

    if len(paper_rows[0]) != len(frag_rows[0]):
        problems.append(
            f"{label}: header has {len(paper_rows[0])} columns in the paper "
            f"but {len(frag_rows[0])} in the fragment"
        )

    paper_data = {normalize_key(r[0]): r for r in paper_rows[1:]}
    frag_data = {normalize_key(r[0]): r for r in frag_rows[1:]}
    if list(paper_data) != list(frag_data):
        problems.append(
            f"{label}: row keys differ\n    paper:    {list(paper_data)}\n"
            f"    fragment: {list(frag_data)}"
        )
    for key in paper_data:
        if key not in frag_data:
            continue
        p = row_numbers(paper_data[key][1:])
        f = row_numbers(frag_data[key][1:])
        if p != f:
            problems.append(
                f"{label}: row {key!r} differs\n"
                f"    paper:    {' | '.join(paper_data[key][1:])}\n"
                f"    fragment: {' | '.join(frag_data[key][1:])}"
            )
    return problems


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
    paper_text = strip_comments(manuscript.read_text())

    problems: list[str] = []
    checked = 0
    for table in inventory["tables"]:
        label = table["label"]
        fragment = ROOT / table["source_table"]
        frag_text = strip_comments(fragment.read_text())
        paper_body = tabular_after_label(paper_text, label)
        frag_body = tabular_after_label(frag_text, label)
        if paper_body is None:
            problems.append(f"{label}: no tabular found in {manuscript.name}")
            continue
        if frag_body is None:
            problems.append(f"{label}: no tabular found in {fragment.name}")
            continue
        checked += 1
        problems.extend(compare_table(label, paper_body, frag_body))

    print(f"manuscript: {manuscript}")
    print(f"tables compared: {checked}")
    if problems:
        print("\nMISMATCHES:")
        for p in problems:
            print("  " + p)
        print("\nPaper table check FAILED.")
        return 1
    print("\nPaper table check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
