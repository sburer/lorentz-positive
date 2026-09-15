#!/usr/bin/env python3
"""Generate paper-facing tables for the noxious-facility experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

from make_bilinear_tables import _float


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RAW = OUTPUTS / "raw"
DEFAULT_REGULAR = (
    RAW / "noxious_regular_m3_m12.csv",
    RAW / "noxious_regular_m16.csv",
)
# Larger regular polygons quoted in the Section 4.3 prose ("Full SEP also
# gives 1.001735 at m=20,24,32") but not tabulated.
DEFAULT_REGULAR_LARGE = (RAW / "noxious_regular_large.csv",)
DEFAULT_RANDOM = (
    RAW / "noxious_random_panel.csv",
    RAW / "noxious_random10.csv",
)
DEFAULT_TARGETED = RAW / "noxious_targeted_410.csv"
DEFAULT_TEX = OUTPUTS / "tables" / "noxious_results_fragment.tex"
DEFAULT_SUMMARY = OUTPUTS / "summary" / "noxious_summary.csv"

METHODS = ("Shor", "RLT", "KRON", "Full SEP")
METHOD_LABELS = {
    "Shor": "Shor",
    "RLT": "RLT",
    "KRON": "KRON",
    "Full SEP": "Full SEP",
}


def load_rows(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    """Read rows from one or more CSV files."""

    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def bound(row: dict[str, str]) -> float:
    """Return the noxious distance upper bound from a result row."""

    value = _float(row.get("distance_bound"))
    if value is None:
        raise ValueError(f"missing distance_bound in {row.get('instance')}")
    return value


def reference(row: dict[str, str]) -> float:
    """Return the exact planar reference distance."""

    value = _float(row.get("theta_reference"))
    if value is None:
        raise ValueError(f"missing theta_reference in {row.get('instance')}")
    return value


def regular_table(rows: list[dict[str, str]]) -> str:
    """Return the regular-polygon manuscript table."""

    by_size_method: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        if method in METHODS:
            by_size_method[(int(row["num_sites"]), method)] = row

    sizes = [3, 4, 5, 6, 7, 8, 10, 12, 16]
    lines = [
        r"\begin{table}",
        r"\caption{Distance upper bounds for regular \(m\)-gons}",
        r"\vskip 10pt",
        r"\label{tab:regular}",
        r"\centering",
        r"\begin{tabular}{r|rrrr}",
        r"\toprule",
        r"\(m\) & Shor & RLT & KRON & Full SEP \\",
        r"\midrule",
    ]
    for size in sizes:
        values = [bound(by_size_method[(size, method)]) for method in METHODS]
        lines.append(
            f"{size:<2d} & "
            + " & ".join(f"{value:.6f}" for value in values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def regular_large_note(rows: list[dict[str, str]]) -> str:
    """Return the prose values for the larger regular polygons as a comment."""

    values = {
        int(row["num_sites"]): bound(row)
        for row in rows
        if row["method"] == "Full SEP"
    }
    parts = ", ".join(f"m={m}: {values[m]:.6f}" for m in sorted(values))
    return f"% Full SEP distance upper bound for larger regular m-gons ({parts})"


def random_table(rows: list[dict[str, str]]) -> str:
    """Return the normalized random-instance manuscript table."""

    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        method = row["method"]
        if method not in METHODS:
            continue
        size = int(row["num_sites"])
        gap = bound(row) - reference(row)
        groups[(size, method)].append(gap)

    sizes = [4, 6, 8, 10, 15, 20]
    lines = [
        r"\begin{table}",
        r"\caption{Results for normalized random instances}",
        r"\label{tab:random}",
        r"\vskip 10pt",
        r"\centering",
        r"\begin{tabular}{rr|c|c|c|c}",
        r"\toprule",
        r" & &Shor gap &  RLT gap & KRON gap & SEP gap\\",
        r"\(m\) & cases & median/max & median/max & median/max& median/max\\",
        r"\midrule",
    ]
    for size in sizes:
        cells: list[str] = []
        case_count = None
        for method in METHODS:
            values = groups[(size, method)]
            if not values:
                raise ValueError(f"missing random rows for m={size}, {method}")
            case_count = len(values) if case_count is None else case_count
            cells.append(f"{median(values):.3f}/{max(values):.3f}")
        lines.append(f"{size:<2d} & {case_count} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def random_timing_note(rows: list[dict[str, str]]) -> str:
    """Return the median KRON and Full SEP times quoted in the prose as a comment."""

    times: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] in ("KRON", "Full SEP"):
            times[(int(row["num_sites"]), row["method"])].append(float(row["runtime"]))
    sizes = [4, 6, 8, 10, 15, 20]
    parts = []
    for method in ("Full SEP", "KRON"):
        values = ", ".join(f"{median(times[(size, method)]):.3f}" for size in sizes)
        parts.append(f"median {method} times for m={','.join(map(str, sizes))}: {values} s")
    return "% " + "; ".join(parts)


def targeted_table(rows: list[dict[str, str]]) -> str:
    """Return the targeted four-point manuscript table."""

    by_method = {row["method"]: row for row in rows if row["method"] in METHODS}
    exact = reference(next(iter(by_method.values())))
    lines = [
        r"\begin{table}\caption{Bounds for instance with points \eqref{eq:targeted}}",
        r"\label{tab:targeted}",
        r"\centering",
        r"\vskip 10pt",
        r"\begin{tabular}{r|r}",
        r"\toprule",
        r"method & distance upper bound \\",
        r"\midrule",
    ]
    for method in METHODS:
        lines.append(f"{METHOD_LABELS[method]:<8s} & {bound(by_method[method]):.10f} " + r"\\")
    lines.extend(
        [
            r"\midrule",
            f"exact value & {exact:.10f} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_summary(
    path: Path,
    regular_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
    targeted_rows: list[dict[str, str]],
) -> None:
    """Write a compact numeric audit trail for the generated tables."""

    fieldnames = [
        "panel",
        "size",
        "method",
        "cases",
        "median_gap",
        "max_gap",
        "median_time",
        "bound",
        "reference",
    ]
    out_rows: list[dict[str, object]] = []
    for row in regular_rows + targeted_rows:
        if row["method"] not in METHODS:
            continue
        panel = "targeted" if row["generator"] == "targeted-410" else "regular"
        out_rows.append(
            {
                "panel": panel,
                "size": row["num_sites"],
                "method": METHOD_LABELS[row["method"]],
                "cases": 1,
                "median_gap": "",
                "max_gap": "",
                "median_time": "",
                "bound": bound(row),
                "reference": reference(row),
            }
        )
    random_groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    random_times: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in random_rows:
        if row["method"] in METHODS:
            key = (int(row["num_sites"]), row["method"])
            random_groups[key].append(bound(row) - reference(row))
            random_times[key].append(float(row["runtime"]))
    for (size, method), values in sorted(random_groups.items()):
        out_rows.append(
            {
                "panel": "random",
                "size": size,
                "method": METHOD_LABELS[method],
                "cases": len(values),
                "median_gap": median(values),
                "max_gap": max(values),
                "median_time": median(random_times[(size, method)]),
                "bound": "",
                "reference": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)


def write_tex(path: Path, regular: str, random: str, targeted: str) -> None:
    """Write a standalone noxious table fragment."""

    insert = "\n\n".join([regular, random, targeted])
    content = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb,booktabs}",
            "",
            r"\begin{document}",
            r"\section*{Noxious-Facility Computational Results}",
            "",
            r"The equations below are placeholders so this fragment compiles",
            r"independently.  Copy only the block marked ``BEGIN PAPER INSERT''",
            r"into paper/Lorentz.tex.",
            r"\begin{equation}",
            r"\label{eq:targeted}",
            r"p_i\in\mathbb{R}^2,\qquad i=1,\ldots,4.",
            r"\end{equation}",
            "",
            r"% ======================== BEGIN PAPER INSERT ========================",
            insert,
            r"% ========================= END PAPER INSERT =========================",
            "",
            r"\end{document}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regular", type=Path, nargs="*", default=list(DEFAULT_REGULAR))
    parser.add_argument(
        "--regular-large", type=Path, nargs="*", default=list(DEFAULT_REGULAR_LARGE)
    )
    parser.add_argument("--random", type=Path, nargs="*", default=list(DEFAULT_RANDOM))
    parser.add_argument("--targeted", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    args = parser.parse_args()

    regular_rows = load_rows(tuple(args.regular))
    regular_large_rows = load_rows(tuple(args.regular_large))
    random_rows = load_rows(tuple(args.random))
    targeted_rows = load_rows((args.targeted,))
    write_summary(args.summary, regular_rows, random_rows, targeted_rows)
    write_tex(
        args.tex,
        regular_table(regular_rows) + "\n" + regular_large_note(regular_large_rows),
        random_table(random_rows) + "\n" + random_timing_note(random_rows),
        targeted_table(targeted_rows),
    )
    print(f"wrote {args.summary}")
    print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
