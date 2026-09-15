#!/usr/bin/env python3
"""Generate paper-facing tables for ball-constraints TTRS runs."""

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
DEFAULT_RAW = OUTPUTS / "raw" / "ttrs_raw.csv"
DEFAULT_SUMMARY = OUTPUTS / "summary" / "ttrs_summary.csv"
DEFAULT_TEX = OUTPUTS / "tables" / "ttrs_results_fragment.tex"

METHODS = ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP", "Gurobi")
METHOD_ALIASES = {
    "shor": "Shor",
    "kron": "KRON",
    "lazy-kron": "Lazy KRON",
    "lkron": "Lazy KRON",
    "full": "Full SEP",
    "lazy": "Lazy SEP",
    "gurobi": "Gurobi",
}


def present_methods(rows: list[dict[str, str]]) -> tuple[str, ...]:
    """Return known methods that are actually present in the raw CSV."""

    found = {row["method"] for row in rows}
    return tuple(method for method in METHODS if method in found)


def parse_methods(text: str | None, rows: list[dict[str, str]]) -> tuple[str, ...]:
    """Parse optional method aliases, defaulting to methods in the raw CSV."""

    if text is None:
        found = set(present_methods(rows))
        defaults = [method for method in ("Full SEP", "Lazy SEP") if method in found]
        if "Gurobi" in found:
            defaults.append("Gurobi")
        if defaults:
            return tuple(defaults)
        return present_methods(rows)
    methods = []
    for part in text.split(","):
        alias = part.strip().lower()
        if not alias:
            continue
        if alias not in METHOD_ALIASES:
            raise ValueError(f"unknown method alias {alias!r}")
        methods.append(METHOD_ALIASES[alias])
    if not methods:
        raise ValueError("at least one method is required")
    found = {row["method"] for row in rows}
    missing = [method for method in methods if method not in found]
    if missing:
        raise ValueError(f"requested methods absent from raw CSV: {missing}")
    return tuple(methods)


def summarize(
    rows: list[dict[str, str]],
    *,
    methods: tuple[str, ...],
    tight_tol: float = PAPER_TIGHT_REL_TOL,
    recovery_feas_tol: float = 1e-7,
):
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[int(row["n"])].append(row)

    summaries: list[dict[str, object]] = []
    for n, group in sorted(groups.items()):
        files = sorted({row["file"] for row in group})
        by_method = {
            method: {row["file"]: row for row in group if row["method"] == method}
            for method in methods
        }
        summary: dict[str, object] = {"n": n, "cases": len(files)}

        for method in methods:
            method_rows = list(by_method[method].values())
            summary[f"{method}_cases"] = len(method_rows)
            if not method_rows:
                summary[f"{method}_median_time"] = None
                summary[f"{method}_max_time"] = None
                summary[f"{method}_tight"] = None
                summary[f"{method}_max_gap"] = None
                summary[f"{method}_max_relative_pd_gap"] = None
                continue
            times = [_float(row.get("runtime")) for row in method_rows]
            gaps = [
                _recovered_gap(row, recovery_feas_tol) for row in method_rows
            ]
            gaps = [gap for gap in gaps if gap is not None]
            summary[f"{method}_median_time"] = median([t for t in times if t is not None])
            summary[f"{method}_max_time"] = max(t for t in times if t is not None)
            summary[f"{method}_tight"] = sum(gap <= tight_tol for gap in gaps)
            summary[f"{method}_max_gap"] = max(gaps) if gaps else None
            pd_gaps = [
                abs(v)
                for row in method_rows
                if (v := _float(row.get("relative_pd_gap"))) is not None
            ]
            summary[f"{method}_max_relative_pd_gap"] = max(pd_gaps) if pd_gaps else None
            if method == "Gurobi":
                summary[f"{method}_optimal"] = sum(
                    row.get("status") == "OPTIMAL" for row in method_rows
                )
                summary[f"{method}_time_limit"] = sum(
                    row.get("status") == "TIME_LIMIT" for row in method_rows
                )

        if "Lazy SEP" in methods:
            lazy_rows = list(by_method["Lazy SEP"].values())
            lazy_rounds = [_float(row.get("n_rounds")) for row in lazy_rows]
            lazy_cuts = [_float(row.get("n_cuts")) for row in lazy_rows]
            lazy_skews = [_float(row.get("relative_max_skew_violation")) for row in lazy_rows]
            lazy_rank1 = [_float(row.get("relative_rank1_residual")) for row in lazy_rows]
            lazy_cert = [_float(row.get("relative_certification_gap")) for row in lazy_rows]
            lazy_feas = [_float(row.get("primal_feasibility_violation")) for row in lazy_rows]
            summary.update(
                {
                    "median_lazy_rounds": median([v for v in lazy_rounds if v is not None]),
                    "max_lazy_rounds": max(v for v in lazy_rounds if v is not None),
                    "median_lazy_cuts": median([v for v in lazy_cuts if v is not None]),
                    "max_lazy_cuts": max(v for v in lazy_cuts if v is not None),
                    "max_lazy_skew": max(v for v in lazy_skews if v is not None),
                    "max_lazy_rank1": max(v for v in lazy_rank1 if v is not None),
                    "max_lazy_certification_gap": max(v for v in lazy_cert if v is not None),
                    "max_lazy_feasibility_violation": max(v for v in lazy_feas if v is not None),
                }
            )
        summaries.append(summary)
    return summaries


def _recovered_gap(row: dict[str, str], feasibility_tolerance: float) -> float | None:
    """Return a method's own feasible-upper-bound gap after validation."""

    violation = _float(row.get("primal_feasibility_violation"))
    if violation is None:
        raise ValueError(
            f"{row.get('method')} row {row.get('file')} lacks recovered-point "
            "feasibility diagnostics"
        )
    if violation > feasibility_tolerance:
        raise ValueError(
            f"infeasible recovered point for {row.get('method')} "
            f"{row.get('file')}: violation={violation:.3e}"
        )
    return relative_gap(_float(row.get("upper_bound")), _float(row.get("lower_bound")))


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in summaries:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def combined_table(summaries: list[dict[str, object]], methods: tuple[str, ...]) -> str:
    """Return one paper table with timing, bound quality, and Lazy SEP diagnostics."""

    if methods == ("Full SEP", "Lazy SEP", "Gurobi"):
        return manuscript_results_table(summaries)

    methods = tuple(method for method in methods if method != "Gurobi")
    if methods == ("Full SEP", "Lazy SEP"):
        return sep_lazy_table(summaries)

    lazy_present = "Lazy SEP" in methods
    colspec = "c r" + "|r" * len(methods) + "|r r" * len(methods)
    if lazy_present:
        colspec += "|r r r"
    font_size = r"\tiny" if len(methods) >= 4 else r"\scriptsize"
    tabcolsep = "0.4pt" if len(methods) >= 4 else "2.5pt"
    display = {
        "Lazy KRON": "L-KRON",
        "Full SEP": "Full",
        "Lazy SEP": "L-SEP",
    }
    method_labels = [display.get(method, method) for method in methods]
    time_header = " & ".join(method_labels)
    group_header = " & ".join(
        rf"\multicolumn{{2}}{{c|}}{{{label}}}" for label in method_labels[:-1]
    )
    if methods:
        last_method = method_labels[-1]
        suffix = "|" if lazy_present else ""
        last_group = rf"\multicolumn{{2}}{{c{suffix}}}{{{last_method}}}"
        group_header = (
            last_group if not group_header else group_header + " & " + last_group
        )
    detail_header = time_header + " & " + " & ".join("tight & max gap" for _ in methods)
    if lazy_present:
        group_header += r" & \multicolumn{3}{c}{Lazy SEP diagnostics}"
        detail_header += " & rounds & cuts & max skew"
    quality_phrase = (
        methods[0]
        if len(methods) == 1
        else f"{methods[0]} and {methods[1]}"
        if len(methods) == 2
        else ", ".join(methods[:-1]) + f", and {methods[-1]}"
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Results for SEP and Lazy SEP on TTRS instances.  The column \(n\) is the dimension and cases is the number of instances in the panel.  For each method in {quality_phrase}, time reports the median wall-clock solve time, tight counts instances whose method-specific recovered feasible upper bound agrees with its relaxation lower bound to relative tolerance \(10^{{-5}}\), and max gap is the largest such relative gap.  The Lazy SEP diagnostics report median/maximum rounds, median/maximum coordinate skew-skew cuts, and maximum final coordinate skew-skew equation violation.}}",
        r"\label{tab:TTRS_results}",
        r"\begingroup",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        font_size,
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        rf"\multicolumn{{2}}{{c|}}{{}} & \multicolumn{{{len(methods)}}}{{c|}}{{Median time}} & {group_header} \\",
        rf"\(n\) & cases & {detail_header} \\",
        r"\hline",
    ]
    for row in summaries:
        time_values = [_fmt_time(row[f"{method}_median_time"]) for method in methods]
        quality_values = []
        for method in methods:
            if int(row.get(f"{method}_cases", 0)) == 0:
                quality_values.extend(["--", "--"])
            else:
                quality_values.extend(
                    [
                        f"{int(row[f'{method}_tight'])}/{int(row[f'{method}_cases'])}",
                        _fmt_sci(row[f"{method}_max_gap"]),
                    ]
                )
        values = time_values + quality_values
        if lazy_present:
            values.extend(
                [
                    f"{row['median_lazy_rounds']:.0f}/{row['max_lazy_rounds']:.0f}",
                    f"{row['median_lazy_cuts']:.0f}/{row['max_lazy_cuts']:.0f}",
                    _fmt_sci(row["max_lazy_skew"]),
                ]
            )
        lines.append(rf"{row['n']} & {int(row['cases'])} & " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def sep_lazy_table(summaries: list[dict[str, object]]) -> str:
    """Return the TTRS Table 5 layout used in the manuscript draft."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Results for SEP and Lazy SEP on TTRS instances.  The column \(n\) is the dimension and cases is the number of instances.  For each method, time is the median wall-clock solve time, tight counts method-specific recovered feasible upper bounds agreeing with their relaxation lower bounds to relative tolerance \(10^{-5}\), and max gap is the largest such relative gap.  For Lazy SEP, rounds and cuts are median/maximum counts, and max skew is the largest final relative coordinate skew-skew equation violation.}",
        r"\label{tab:TTRS_results}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\scriptsize",
        r"\begin{tabular}{c r|r r r|r r r r r r}",
        r"\hline",
        r"\multicolumn{2}{c|}{} & \multicolumn{3}{c|}{Full SEP} & \multicolumn{6}{c}{Lazy SEP} \\",
        r"\(n\) & cases & time & tight & max gap & time & tight & max gap & rounds & cuts & max skew \\",
        r"\hline",
    ]
    for row in summaries:
        full_cases = int(row.get("Full SEP_cases", 0) or 0)
        lazy_cases = int(row.get("Lazy SEP_cases", 0) or 0)
        lines.append(
            rf"{row['n']} & {int(row['cases'])} & "
            rf"{_fmt_time(row.get('Full SEP_median_time'))} & "
            rf"{int(row.get('Full SEP_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Full SEP_max_gap'))} & "
            rf"{_fmt_time(row.get('Lazy SEP_median_time'))} & "
            rf"{int(row.get('Lazy SEP_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Lazy SEP_max_gap'))} & "
            rf"{row['median_lazy_rounds']:.0f}/{row['max_lazy_rounds']:.0f} & "
            rf"{row['median_lazy_cuts']:.0f}/{row['max_lazy_cuts']:.0f} & "
            rf"{_fmt_sci(row['max_lazy_skew'])} \\"
        )
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def manuscript_results_table(summaries: list[dict[str, object]]) -> str:
    """Return the combined Full SEP, Lazy SEP, and Gurobi manuscript table."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Results on TTRS instances from \cite{Burer.Anstreicher.2013}}",
        r"\label{tab:TTRS_results}",
        r"\vskip 10pt",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\small",
        r"\begin{tabular}{c r|r r r| r r r|r r r }",
        r"\hline",
        r"\multicolumn{2}{c|}{} & \multicolumn{3}{c|}{Full SEP} & \multicolumn{3}{c|}{Lazy SEP}& \multicolumn{3}{c}{Gurobi} \\",
        r"\(n\) & cases & time & tight & max gap & time & tight & max gap   & time&tight&max gap \\",
        r"\hline",
    ]
    for row in summaries:
        full_cases = int(row.get("Full SEP_cases", 0) or 0)
        lazy_cases = int(row.get("Lazy SEP_cases", 0) or 0)
        gurobi_cases = int(row.get("Gurobi_cases", 0) or 0)
        lines.append(
            rf"{row['n']} & {int(row['cases'])} & "
            rf"{_fmt_time(row.get('Full SEP_median_time'))} & "
            rf"{int(row.get('Full SEP_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Full SEP_max_gap'))} & "
            rf"{_fmt_time(row.get('Lazy SEP_median_time'))} & "
            rf"{int(row.get('Lazy SEP_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Lazy SEP_max_gap'))} & "
            rf"{_fmt_time(row.get('Gurobi_median_time'))} & "
            rf"{int(row.get('Gurobi_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Gurobi_max_gap'))} \\"
        )
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def manuscript_lazy_diagnostics_table(summaries: list[dict[str, object]]) -> str:
    """Return the separate Lazy SEP diagnostics table used by the manuscript."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Additional diagnostics for Lazy SEP on TTRS instances from \cite{Burer.Anstreicher.2013}}",
        r"\label{tab:TTRS_diagnostics}",
        r"\vskip 10pt",
        r"\begingroup",
        r"\begin{tabular}{c|r r r}",
        r"\hline",
        r"$n$& rounds & cuts&max skew\\",
        r"\hline",
    ]
    for row in summaries:
        lines.append(
            rf"{row['n']} & "
            rf"{row['median_lazy_rounds']:.0f}/{row['max_lazy_rounds']:.0f} & "
            rf"{row['median_lazy_cuts']:.0f}/{row['max_lazy_cuts']:.0f} & "
            rf"{_fmt_sci(row['max_lazy_skew'])}\\"
        )
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def gurobi_table(summaries: list[dict[str, object]]) -> str:
    """Return a separate, optional table for the Gurobi panel."""

    rows = [
        row for row in summaries if int(row.get("Gurobi_cases", 0) or 0) > 0
    ]
    if not rows:
        return ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Gurobi results on the ball-constraints TTRS instances.  The column \(n\) is the dimension, cases is the number of instances in the panel, time is the median wall-clock solve time, tight counts instances whose Gurobi incumbent agrees with its own global lower bound to relative tolerance \(10^{-5}\), and max gap is the largest such relative gap.}",
        r"\label{tab:ttrs_gurobi_results}",
        r"\small",
        r"\begin{tabular}{c r|r r r}",
        r"\hline",
        r"\(n\) & cases & time & tight & max gap \\",
        r"\hline",
    ]
    for row in rows:
        run = int(row.get("Gurobi_cases", 0) or 0)
        lines.append(
            rf"{row['n']} & {int(row['cases'])} & "
            rf"{_fmt_time(row.get('Gurobi_median_time'))} & "
            rf"{int(row.get('Gurobi_tight', 0) or 0)} & "
            rf"{_fmt_sci(row.get('Gurobi_max_gap'))} \\"
        )
    lines.extend([r"\hline", r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def write_tex(
    path: Path, summaries: list[dict[str, object]], methods: tuple[str, ...]
) -> None:
    total = sum(int(row["cases"]) for row in summaries)
    main_methods = tuple(method for method in methods if method != "Gurobi")
    has_gurobi = "Gurobi" in methods
    method_phrase = _method_phrase(main_methods)
    if main_methods == ("Lazy SEP",):
        omitted = [
            method
            for method in ("Shor", "KRON", "Lazy KRON", "Full SEP", "Gurobi")
            if method not in main_methods
        ]
        omitted_phrase = _method_phrase(tuple(omitted)) if omitted else "No methods"
        setup = rf"""
\noindent We ran Lazy SEP on the {total} centered diagonal TTRS instances from
the ball-constraints repository.  {omitted_phrase} are omitted
from this panel.  The implementation uses the transpose of the paper's bilinear
convention.  It forms the square SEP lift
\[
Z=\begin{{pmatrix}}1&y^\top\\ x&V\end{{pmatrix}},\qquad
y=\operatorname{{diag}}(a)x+b,
\]
and represents \(V_{{ij}}=x_i y_j=x_i(a_jx_j+b_j)\) linearly through the
\(x\)-moment matrix.  Violated skew-skew equations are then added by separation.
For every SDP row, bound quality is measured using that method's own recovered
first-column feasible point as an upper bound; neither Gurobi nor the corrected
beta values are used to calculate an SDP gap.
""".strip()
    elif "Full SEP" in main_methods and "Lazy SEP" in main_methods and not any(
        method in main_methods for method in ("Shor", "KRON")
    ):
        full_total = sum(int(row.get("Full SEP_cases", 0)) for row in summaries)
        full_dims = [
            str(row["n"])
            for row in summaries
            if int(row.get("Full SEP_cases", 0)) == int(row["cases"])
        ]
        full_dim_text = ", ".join(full_dims)
        partial_full = [
            (
                int(row.get("Full SEP_cases", 0)),
                int(row["cases"]),
                int(row["n"]),
            )
            for row in summaries
            if 0 < int(row.get("Full SEP_cases", 0)) < int(row["cases"])
        ]
        partial_phrases = [
            rf"{done} of {cases} instances with \(n={n}\)"
            for done, cases, n in partial_full
        ]
        partial_text = "; ".join(partial_phrases)
        partial_note = ""
        if partial_full:
            partial_note = (
                r"  Because the first few \(n=20\) Full SEP runs each took many "
                r"minutes, the remaining \(n=20\) Full SEP rows have not been "
                r"run in this panel."
            )
        full_coverage = (
            rf"all instances with \(n\in\{{{full_dim_text}\}}\)"
            if not partial_text
            else rf"all instances with \(n\in\{{{full_dim_text}\}}\) and {partial_text}"
        )
        omitted = [method for method in ("Shor", "KRON") if method not in main_methods]
        omitted_text = (
            ""
            if not omitted
            else rf"  {_method_phrase(tuple(omitted))} are omitted from this panel."
        )
        gurobi_text = ""
        lazy_kron_text = (
            ""
            if "Lazy KRON" not in main_methods
            else (
                r"  Lazy KRON enforces the KRON condition by adding separated "
                r"scalar cuts."
            )
        )
        setup = rf"""
\noindent We ran Lazy SEP on the {total} centered diagonal TTRS instances from
the ball-constraints repository, and Full SEP on {full_total} instances:
{full_coverage}.{omitted_text}  The implementation uses the
transpose of the paper's bilinear convention.  It forms the square SEP lift
\[
Z=\begin{{pmatrix}}1&y^\top\\ x&V\end{{pmatrix}},\qquad
y=\operatorname{{diag}}(a)x+b,
\]
and represents \(V_{{ij}}=x_i y_j=x_i(a_jx_j+b_j)\) linearly through the
\(x\)-moment matrix.  Full SEP imposes all skew-skew equations at once, while
Lazy SEP adds violated coordinate skew-skew equations by separation.{lazy_kron_text}{gurobi_text}{partial_note}  For every SDP row, bound quality is measured using that method's own recovered first-column feasible point as an upper bound; neither Gurobi nor the corrected beta values are used to calculate an SDP gap.
""".strip()
    else:
        if has_gurobi and not main_methods:
            setup = rf"""
\noindent We ran Gurobi on the centered diagonal TTRS instances
from the ball-constraints repository.  For Gurobi, the time limit for each
instance is \(\max\{{5,3t_{{\max}}\}}\), where \(t_{{\max}}\) is the maximum
runtime among the conic methods already available for that instance.  Gurobi
bound quality compares its incumbent with its own global lower bound.
""".strip()
        else:
            setup = rf"""
\noindent We ran {len(main_methods)} conic relaxations on the {total} centered diagonal TTRS
instances from the ball-constraints repository: {method_phrase}.  Gurobi is
intentionally omitted from this panel.  Each SDP gap compares the relaxation
lower bound with the objective value of that method's own recovered
first-column feasible point.
""".strip()
    blocks = [setup]
    if main_methods:
        if has_gurobi and main_methods == ("Full SEP", "Lazy SEP"):
            blocks.append(manuscript_results_table(summaries))
        else:
            blocks.append(combined_table(summaries, main_methods))
    if "Lazy SEP" in main_methods:
        blocks.append(manuscript_lazy_diagnostics_table(summaries))
    if has_gurobi and not (
        "Full SEP" in main_methods and "Lazy SEP" in main_methods
    ):
        if main_methods:
            gurobi_setup = r"""
The Gurobi runs are reported separately because they answer a different
comparison question from the SEP timing and bound results in
Table~\ref{tab:ttrs_results}.
""".strip()
        else:
            gurobi_setup = r"""
The table below reports the Gurobi rows for the ball-constraints TTRS
instances.
""".strip()
        table = gurobi_table(summaries)
        if table:
            blocks.extend([gurobi_setup, table])
    insert = "\n\n".join(blocks)
    content = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\providecommand{\suchthat}{:\,}",
            "",
            r"\begin{document}",
            r"\section*{TTRS Computational Results}",
            "",
            r"The displayed formulation below is included only so that this file compiles",
            r"independently.  Copy only the block marked ``BEGIN PAPER INSERT'' into",
            r"paper/Lorentz.tex.",
            r"\begin{equation}",
            r"\min\{x^\top Qx+2g^\top x\suchthat \|x\|\le 1,\ "
            r"\|\operatorname{diag}(a)x+b\|\le 1\}.",
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


def _method_phrase(methods: tuple[str, ...]) -> str:
    if not methods:
        return ""
    if len(methods) == 1:
        return methods[0]
    if len(methods) == 2:
        return f"{methods[0]} and {methods[1]}"
    return ", ".join(methods[:-1]) + f", and {methods[-1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--extra-raw",
        action="append",
        type=Path,
        default=[],
        help="additional raw CSV to combine with --raw, keyed by file and method",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--recovery-feas-tol", type=float, default=1e-7)
    parser.add_argument(
        "--methods",
        default=None,
        help="optional comma-separated subset: shor,kron,lazy-kron,full,lazy,gurobi",
    )
    args = parser.parse_args()

    rows = load_combined_rows(args.raw, args.extra_raw)
    methods = parse_methods(args.methods, rows)
    rows = [row for row in rows if row["method"] in methods]
    summaries = summarize(
        rows,
        methods=methods,
        tight_tol=PAPER_TIGHT_REL_TOL,
        recovery_feas_tol=args.recovery_feas_tol,
    )
    write_summary_csv(args.summary, summaries)
    write_tex(args.tex, summaries, methods)
    print(f"wrote {args.summary}")
    print(f"wrote {args.tex}")
    return 0


def load_combined_rows(raw: Path, extra_raws: list[Path]) -> list[dict[str, str]]:
    """Load raw rows, with later extra files replacing duplicate method rows."""

    keyed: dict[tuple[str, str], dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    for path in [raw, *extra_raws]:
        for row in load_rows(path):
            key = (row["file"], row["method"])
            if key not in keyed:
                order.append(key)
            keyed[key] = row
    return [keyed[key] for key in order]


if __name__ == "__main__":
    raise SystemExit(main())
