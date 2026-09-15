#!/usr/bin/env python3
"""Generate large-instance table fragments for max-distance and TTRS.

These tables mirror the current manuscript tables but intentionally omit the
methods not run in the large panels.  Since no Gurobi or beta reference is used
for the generated large cases, "tight" means that the relative gap between the
SDP bound and the recovered first-column feasible point is at most ``1e-5``.
Rank-one and feasibility residuals are retained as separate diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

from make_bilinear_tables import _float, _fmt_sci, _fmt_time, load_rows
from paper_code.metrics import PAPER_TIGHT_REL_TOL, relative_gap


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEFAULT_MAXDIST_RAW = OUTPUTS / "raw" / "maxdist_large_raw.csv"
DEFAULT_TTRS_RAW = OUTPUTS / "raw" / "ttrs_generated_large_raw.csv"
DEFAULT_TEX = OUTPUTS / "tables" / "large_instance_results_fragment.tex"
DEFAULT_MAXDIST_SUMMARY = OUTPUTS / "summary" / "maxdist_large_summary.csv"
DEFAULT_TTRS_SUMMARY = OUTPUTS / "summary" / "ttrs_generated_large_summary.csv"

MAXDIST_METHODS = ("Lazy KRON", "Lazy SEP")
TTRS_METHODS = ("Lazy KRON", "Lazy SEP")


def certification_gap(row: dict[str, str]) -> float | None:
    """Return the relative recovered-primal versus SDP-bound gap."""

    saved = _float(row.get("relative_certification_gap"))
    if saved is not None:
        return saved
    return relative_gap(_float(row.get("upper_bound")), _float(row.get("lower_bound")))


def recovered_feasibility(row: dict[str, str]) -> float | None:
    """Return the saved recovered-point violation, with a legacy fallback."""

    feas = _float(row.get("primal_feasibility_violation"))
    if feas is None:
        x_trace = _float(row.get("x_trace"))
        y_trace = _float(row.get("y_trace"))
        if x_trace is not None or y_trace is not None:
            feas = max(0.0, (x_trace or 0.0) - 1.0, (y_trace or 0.0) - 1.0)
    return feas


def is_tight(row: dict[str, str], *, tight_tol: float) -> bool:
    """Apply the sole paper-wide tightness test: relative gap <= tolerance."""

    gap = certification_gap(row)
    return gap is not None and gap <= tight_tol


def median_max(values: list[float]) -> str:
    if not values:
        return "--"
    return f"{median(values):.0f}/{max(values):.0f}"


def summarize_maxdist(
    rows: list[dict[str, str]], *, tight_tol: float
) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("method") in MAXDIST_METHODS:
            groups[int(row["n"])].append(row)

    summaries: list[dict[str, object]] = []
    for n, group in sorted(groups.items()):
        seeds = sorted({int(row["seed"]) for row in group})
        by_method = {
            method: {int(row["seed"]): row for row in group if row["method"] == method}
            for method in MAXDIST_METHODS
        }
        summary: dict[str, object] = {"n": n, "cases": len(seeds)}
        for method in MAXDIST_METHODS:
            method_rows = list(by_method[method].values())
            times = [_float(row.get("runtime")) for row in method_rows]
            gaps = [certification_gap(row) for row in method_rows]
            ranks = [_float(row.get("relative_rank1_residual")) for row in method_rows]
            feasibilities = [recovered_feasibility(row) for row in method_rows]
            rounds = [_float(row.get("n_rounds")) for row in method_rows]
            cuts = [_float(row.get("n_cuts")) for row in method_rows]
            if method == "Lazy KRON":
                violations = [
                    _float(row.get("relative_max_kron_violation"))
                    for row in method_rows
                ]
            else:
                violations = [
                    _float(row.get("relative_max_skew_violation"))
                    for row in method_rows
                ]
            summary[f"{method}_cases"] = len(method_rows)
            summary[f"{method}_median_time"] = (
                median([v for v in times if v is not None]) if times else None
            )
            summary[f"{method}_tight"] = sum(
                is_tight(row, tight_tol=tight_tol)
                for row in method_rows
            )
            summary[f"{method}_max_gap"] = (
                max(v for v in gaps if v is not None)
                if any(v is not None for v in gaps)
                else None
            )
            summary[f"{method}_max_rank1"] = (
                max(v for v in ranks if v is not None)
                if any(v is not None for v in ranks)
                else None
            )
            summary[f"{method}_max_feasibility_violation"] = (
                max(v for v in feasibilities if v is not None)
                if any(v is not None for v in feasibilities)
                else None
            )
            summary[f"{method}_rounds"] = median_max([v for v in rounds if v is not None])
            summary[f"{method}_cuts"] = median_max([v for v in cuts if v is not None])
            summary[f"{method}_max_violation"] = (
                max(v for v in violations if v is not None)
                if any(v is not None for v in violations)
                else None
            )
        summaries.append(summary)
    return summaries


def summarize_ttrs(
    rows: list[dict[str, str]], *, tight_tol: float
) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("method") in TTRS_METHODS:
            groups[int(row["n"])].append(row)

    summaries: list[dict[str, object]] = []
    for n, group in sorted(groups.items()):
        files = sorted({row["file"] for row in group})
        summary: dict[str, object] = {"n": n, "cases": len(files)}
        for method in TTRS_METHODS:
            method_rows = [row for row in group if row["method"] == method]
            times = [_float(row.get("runtime")) for row in method_rows]
            gaps = [certification_gap(row) for row in method_rows]
            ranks = [_float(row.get("relative_rank1_residual")) for row in method_rows]
            feasibilities = [recovered_feasibility(row) for row in method_rows]
            rounds = [_float(row.get("n_rounds")) for row in method_rows]
            cuts = [_float(row.get("n_cuts")) for row in method_rows]
            violation_field = (
                "relative_max_kron_violation"
                if method == "Lazy KRON"
                else "relative_max_skew_violation"
            )
            violations = [_float(row.get(violation_field)) for row in method_rows]
            summary[f"{method}_cases"] = len(method_rows)
            summary[f"{method}_median_time"] = (
                median([v for v in times if v is not None]) if times else None
            )
            summary[f"{method}_tight"] = sum(
                is_tight(row, tight_tol=tight_tol)
                for row in method_rows
            )
            summary[f"{method}_max_gap"] = (
                max(v for v in gaps if v is not None)
                if any(v is not None for v in gaps)
                else None
            )
            summary[f"{method}_max_rank1"] = (
                max(v for v in ranks if v is not None)
                if any(v is not None for v in ranks)
                else None
            )
            summary[f"{method}_max_feasibility_violation"] = (
                max(v for v in feasibilities if v is not None)
                if any(v is not None for v in feasibilities)
                else None
            )
            summary[f"{method}_rounds"] = median_max(
                [v for v in rounds if v is not None]
            )
            summary[f"{method}_cuts"] = median_max(
                [v for v in cuts if v is not None]
            )
            summary[f"{method}_max_violation"] = (
                max(v for v in violations if v is not None)
                if any(v is not None for v in violations)
                else None
            )
        summaries.append(summary)
    return summaries


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in summaries:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def maxdist_combined_table(summaries: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Results for larger max-distance instances}",
        r"\label{tab:maxdist_larger}",
        r"\vskip 10pt",
        r"\begingroup",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\small",
        r"\begin{tabular}{c c|r c r|r c r|r r r|r r r}",
        r"\hline",
        r"& &\multicolumn{3}{c|}{Lazy KRON} & \multicolumn{3}{c|}{Lazy SEP} & \multicolumn{3}{c|}{Lazy KRON} & \multicolumn{3}{c}{Lazy SEP} \\",
        r"$n$& cases& time &tight & max gap & time &tight & max gap & rounds & cuts & max viol. & rounds & cuts & max skew \\",
        r"\hline",
    ]
    for row in summaries:
        lk_cases = int(row.get("Lazy KRON_cases", 0) or 0)
        ls_cases = int(row.get("Lazy SEP_cases", 0) or 0)
        lines.append(
            rf"${row['n']}$ &{int(row['cases'])}&"
            rf"{_fmt_time(row.get('Lazy KRON_median_time'))} & "
            rf"{int(row.get('Lazy KRON_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Lazy KRON_max_gap'))} & "
            rf"{_fmt_time(row.get('Lazy SEP_median_time'))} &"
            rf"{int(row.get('Lazy SEP_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Lazy SEP_max_gap'))} & "
            rf"{row.get('Lazy KRON_rounds', '--')} & "
            rf"{row.get('Lazy KRON_cuts', '--')} & "
            rf"{_fmt_sci(row.get('Lazy KRON_max_violation'))} & "
            rf"{row.get('Lazy SEP_rounds', '--')} & "
            rf"{row.get('Lazy SEP_cuts', '--')} & "
            rf"{_fmt_sci(row.get('Lazy SEP_max_violation'))} \\"
        )
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def ttrs_combined_table(summaries: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Results for additional TTRS instances}",
        r"\label{tab:TTRS_larger}",
        r"\vskip 10pt",
        r"\begingroup",
        r"\setlength{\tabcolsep}{2pt}",
        r"\small",
        r"\begin{tabular}{c c|r c r|r c r|r r r|r r r}",
        r"\hline",
        r"& &\multicolumn{3}{c|}{Lazy KRON} & \multicolumn{3}{c|}{Lazy SEP} & \multicolumn{3}{c|}{Lazy KRON} & \multicolumn{3}{c}{Lazy SEP} \\",
        r"\(n\)& cases & time &tight & max gap & time& tight & max gap & rounds & cuts & max viol. & rounds & cuts & max skew \\",
        r"\hline",
    ]
    for row in summaries:
        lk_cases = int(row.get("Lazy KRON_cases", 0) or 0)
        ls_cases = int(row.get("Lazy SEP_cases", 0) or 0)
        lines.append(
            rf"{row['n']} & {int(row['cases'])}&   "
            rf"{_fmt_time(row.get('Lazy KRON_median_time'))}  &"
            rf"{int(row.get('Lazy KRON_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Lazy KRON_max_gap'))} & "
            rf"{_fmt_time(row.get('Lazy SEP_median_time'))}   &  "
            rf"{int(row.get('Lazy SEP_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Lazy SEP_max_gap'))} & "
            rf"{row.get('Lazy KRON_rounds', '--')} & "
            rf"{row.get('Lazy KRON_cuts', '--')} & "
            rf"{_fmt_sci(row.get('Lazy KRON_max_violation'))} & "
            rf"{row.get('Lazy SEP_rounds', '--')} & "
            rf"{row.get('Lazy SEP_cuts', '--')} & "
            rf"{_fmt_sci(row.get('Lazy SEP_max_violation'))} \\"
        )
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def write_tex(
    path: Path,
    maxdist_summaries: list[dict[str, object]],
    ttrs_summaries: list[dict[str, object]],
    run_note: str = "",
) -> None:
    note = f"\n\n\\noindent {run_note.strip()}" if run_note.strip() else ""
    blocks: list[str] = []
    if maxdist_summaries:
        total = sum(int(row["cases"]) for row in maxdist_summaries)
        blocks.append(
            rf"""
\subsection*{{Larger max-distance instances}}

\noindent We generated {total} additional max-distance instances using the same generator
and rank-one Shor prefilter as in Tables~\ref{{tab:maxdist_times}} and
\ref{{tab:maxdist_diagnostics}}.  Full KRON, Full SEP, and Gurobi were omitted.
The table columns therefore retain only the large-scale methods, Lazy KRON and
Lazy SEP.  In the diagnostics table, tight means that the relative gap between
the relaxation bound and the objective value of the recovered first-column
point is at most \(10^{-5}\).  Relative rank-one residuals and recovered-point
feasibility violations are also calculated and retained in the summary CSV,
but neither enters the definition of tight.
{note}

{maxdist_combined_table(maxdist_summaries)}
""".strip()
        )
    if ttrs_summaries:
        total = sum(int(row["cases"]) for row in ttrs_summaries)
        blocks.append(
            rf"""
\subsection*{{Larger generated TTRS instances}}

\noindent We generated {total} centered affine-diagonal TTRS instances and used Shor
only as a prefilter to discard instances already certified by rank-one recovery.
The reported large-scale methods are Lazy KRON and Lazy SEP.  Since no beta or
Gurobi reference is used in this generated panel, tight means that the relative
gap between a relaxation bound and the objective value of its recovered point
is at most \(10^{-5}\).  Relative rank-one residuals and recovered-point
feasibility violations are also calculated and retained in the summary CSV,
but neither enters the definition of tight.
{note}

{ttrs_combined_table(ttrs_summaries)}
""".strip()
        )
    body = "\n\n".join(blocks)
    content = (
        r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}

\begin{document}
\section*{Large-Instance Computational Fragments}

The displayed labels match the current manuscript labels only by reference.
The blocks below are intended as add-on fragments, not automatic replacements
for the existing tables.

"""
        + body
        + "\n\n\\end{document}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxdist-raw", type=Path, default=DEFAULT_MAXDIST_RAW)
    parser.add_argument("--ttrs-raw", type=Path, default=DEFAULT_TTRS_RAW)
    parser.add_argument("--maxdist-summary", type=Path, default=DEFAULT_MAXDIST_SUMMARY)
    parser.add_argument("--ttrs-summary", type=Path, default=DEFAULT_TTRS_SUMMARY)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument(
        "--run-note",
        default="",
        help="optional short implementation note inserted before each table block",
    )
    args = parser.parse_args()

    maxdist_rows = load_rows(args.maxdist_raw) if args.maxdist_raw.exists() else []
    ttrs_rows = load_rows(args.ttrs_raw) if args.ttrs_raw.exists() else []
    maxdist_summaries = summarize_maxdist(
        maxdist_rows,
        tight_tol=PAPER_TIGHT_REL_TOL,
    )
    ttrs_summaries = summarize_ttrs(
        ttrs_rows,
        tight_tol=PAPER_TIGHT_REL_TOL,
    )
    if maxdist_summaries:
        write_summary_csv(args.maxdist_summary, maxdist_summaries)
        print(f"wrote {args.maxdist_summary}")
    if ttrs_summaries:
        write_summary_csv(args.ttrs_summary, ttrs_summaries)
        print(f"wrote {args.ttrs_summary}")
    write_tex(args.tex, maxdist_summaries, ttrs_summaries, args.run_note)
    print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
