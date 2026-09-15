#!/usr/bin/env python3
"""Generate LaTeX table fragments from the bilinear experiment CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from paper_code.metrics import PAPER_TIGHT_REL_TOL, relative_error, relative_gap


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEFAULT_RAW = OUTPUTS / "raw" / "bilinear_raw.csv"
DEFAULT_SUMMARY = OUTPUTS / "summary" / "bilinear_summary.csv"
DEFAULT_TEX = OUTPUTS / "tables" / "bilinear_results_fragment.tex"

METHODS = ("Shor", "Full SEP", "Lazy SEP", "Dual LOP/TRS", "Gurobi")
PANEL_ORDER = {"square": 0, "rectangular": 1}


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    if seconds < 1.0:
        return f"{1000.0 * seconds:.0f} ms"
    return f"{seconds:.2f} s"


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "--"
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.1f}\\cdot 10^{{{exponent}}}$"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(
    rows: list[dict[str, str]], *, tight_tol: float = PAPER_TIGHT_REL_TOL
) -> list[dict[str, object]]:
    """Summarize rows by size, preserving square/rectangular panel labels."""

    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["panel"], int(row["n"]), int(row["m"]))].append(row)

    summaries: list[dict[str, object]] = []
    def sort_key(item: tuple[tuple[str, int, int], list[dict[str, str]]]) -> tuple[int, int, int]:
        panel, n, m = item[0]
        return (PANEL_ORDER.get(panel, 99), n, m)

    for (panel, n, m), group in sorted(groups.items(), key=sort_key):
        seeds = sorted({int(row["seed"]) for row in group})
        by_method = {method: [row for row in group if row["method"] == method] for method in METHODS}
        exact_by_seed: dict[int, float] = {}
        for method in ("Full SEP", "Lazy SEP", "Dual LOP/TRS"):
            for row in by_method[method]:
                seed = int(row["seed"])
                if seed in exact_by_seed:
                    continue
                value = _float(row["upper_bound"])
                if value is not None:
                    exact_by_seed[seed] = value
        conic_diffs = []
        shor_gaps = []
        dual_gaps = []
        dual_oracles = []
        lazy_skews = []
        lazy_cuts = []
        lazy_rounds = []
        for seed in seeds:
            exact = exact_by_seed.get(seed)
            if exact is None:
                continue
            shor_match = [row for row in by_method["Shor"] if int(row["seed"]) == seed]
            if shor_match:
                shor_lower = _float(shor_match[0]["lower_bound"])
                if shor_lower is not None:
                    gap = _float(shor_match[0].get("relative_shor_gap"))
                    shor_gaps.append(
                        gap
                        if gap is not None
                        else relative_gap(exact, shor_lower)
                    )
            for method in ("Lazy SEP", "Dual LOP/TRS"):
                match = [row for row in by_method[method] if int(row["seed"]) == seed]
                if match:
                    upper = _float(match[0]["upper_bound"])
                    if upper is not None:
                        conic_diffs.append(relative_error(upper, exact))
            dual_match = [row for row in by_method["Dual LOP/TRS"] if int(row["seed"]) == seed]
            if dual_match:
                gap = _float(dual_match[0].get("relative_dual_bracket_gap"))
                if gap is None:
                    raw_gap = _float(dual_match[0]["dual_bracket_gap"])
                    lower = _float(dual_match[0].get("lower_bound"))
                    gap = None if raw_gap is None else raw_gap / max(1.0, abs(lower or 0.0))
                oracles = _float(dual_match[0]["oracle_solves"])
                if gap is not None:
                    dual_gaps.append(gap)
                if oracles is not None:
                    dual_oracles.append(oracles)
            lazy_match = [row for row in by_method["Lazy SEP"] if int(row["seed"]) == seed]
            if lazy_match:
                skew = _float(lazy_match[0].get("relative_max_skew_violation"))
                if skew is None:
                    skew = _float(lazy_match[0]["max_skew_violation"])
                cuts = _float(lazy_match[0]["n_cuts"])
                rounds = _float(lazy_match[0].get("n_rounds") or lazy_match[0].get("n_iters"))
                if skew is not None:
                    lazy_skews.append(skew)
                if cuts is not None:
                    lazy_cuts.append(cuts)
                if rounds is not None:
                    lazy_rounds.append(rounds)

        summary: dict[str, object] = {
            "panel": panel,
            "n": n,
            "m": m,
            "cases": len(seeds),
            "shor_tight": sum(1 for gap in shor_gaps if abs(gap) <= tight_tol),
            "max_shor_gap": max(shor_gaps) if shor_gaps else None,
            "max_conic_disagreement": max(conic_diffs) if conic_diffs else None,
            "max_dual_bracket_gap": max(dual_gaps) if dual_gaps else None,
            "max_lazy_skew_violation": max(lazy_skews) if lazy_skews else None,
            "median_lazy_rounds": median(lazy_rounds) if lazy_rounds else None,
            "max_lazy_rounds": max(lazy_rounds) if lazy_rounds else None,
            "median_lazy_cuts": median(lazy_cuts) if lazy_cuts else None,
            "max_lazy_cuts": max(lazy_cuts) if lazy_cuts else None,
            "median_dual_oracles": median(dual_oracles) if dual_oracles else None,
            "max_dual_oracles": max(dual_oracles) if dual_oracles else None,
            "gurobi_closed": sum(1 for row in by_method["Gurobi"] if row["status"] == "OPTIMAL"),
        }
        for method in METHODS:
            times = [_float(row["runtime"]) for row in by_method[method]]
            times = [time for time in times if time is not None]
            summary[f"{method}_median_time"] = median(times) if times else None
            summary[f"{method}_max_time"] = max(times) if times else None
        summaries.append(summary)
    return summaries


def experiment_metadata(rows: list[dict[str, str]]) -> dict[str, str]:
    """Extract generator metadata from the raw CSV."""

    if not rows:
        return {}
    first = rows[0]
    return {
        "generator": first.get("generator", "iid"),
        "generator_alpha": first.get("generator_alpha", ""),
        "generator_sigma": first.get("generator_sigma", ""),
        "prefilter_relative_gap": first.get("prefilter_relative_gap", ""),
        "screen_relative_gap": first.get("screen_relative_gap", ""),
    }


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def timing_table(summaries: list[dict[str, object]]) -> str:
    """Return the manuscript timing table for all bilinear instances."""

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Median wall-clock times for bilinear instances.}",
        "\\label{tab:bilinear_times}",
        "\\small",
        "\\begin{tabular}{c r r r r r r r}",
        "\\hline",
        "$n\\times m$ & cases & Shor & Full SEP & Lazy SEP & LOP/TRS & Gurobi & Gurobi opt \\\\",
        "\\hline",
    ]
    previous_panel = None
    for row in summaries:
        if previous_panel is not None and row["panel"] != previous_panel:
            lines.append("\\hline")
        previous_panel = row["panel"]
        label = f"${row['n']}\\times {row['m']}$"
        cases = int(row["cases"])
        lines.append(
            f"{label} & {cases} & "
            f"{_fmt_time(row['Shor_median_time'])} & "
            f"{_fmt_time(row['Full SEP_median_time'])} & "
            f"{_fmt_time(row['Lazy SEP_median_time'])} & "
            f"{_fmt_time(row['Dual LOP/TRS_median_time'])} & "
            f"{_fmt_time(row['Gurobi_median_time'])} & "
            f"{int(row['gurobi_closed'])} \\\\"
        )
    lines.extend(["\\hline", "\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def diagnostics_table(summaries: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Additional diagnostics for bilinear instances.}",
        "\\label{tab:bilinear_diagnostics}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\scriptsize",
        "\\begin{tabular}{c|r|r|r r r|r r}",
        "\\hline",
        "& Shor & Conic & \\multicolumn{3}{c|}{Lazy SEP} & \\multicolumn{2}{c}{LOP/TRS} \\\\",
        "$n\\times m$ & max gap & max diff & rounds & cuts & max skew & oracles & max gap \\\\",
        "\\hline",
    ]
    previous_panel = None
    for row in summaries:
        if previous_panel is not None and row["panel"] != previous_panel:
            lines.append("\\hline")
        previous_panel = row["panel"]
        label = f"${row['n']}\\times {row['m']}$"
        lazy_rounds = (
            "--"
            if row["median_lazy_rounds"] is None
            else f"{row['median_lazy_rounds']:.0f}/{row['max_lazy_rounds']:.0f}"
        )
        lazy_cuts = (
            "--"
            if row["median_lazy_cuts"] is None
            else f"{row['median_lazy_cuts']:.0f}/{row['max_lazy_cuts']:.0f}"
        )
        dual_oracles = (
            "--"
            if row["median_dual_oracles"] is None
            else f"{row['median_dual_oracles']:.0f}/{row['max_dual_oracles']:.0f}"
        )
        lines.append(
            f"{label} & {_fmt_sci(row['max_shor_gap'])} & "
            f"{_fmt_sci(row['max_conic_disagreement'])} & "
            f"{lazy_rounds} & {lazy_cuts} & "
            f"{_fmt_sci(row['max_lazy_skew_violation'])} & {dual_oracles} & "
            f"{_fmt_sci(row['max_dual_bracket_gap'])} \\\\"
        )
    lines.extend(["\\hline", "\\hline", "\\end{tabular}", "\\endgroup", "\\end{table}"])
    return "\n".join(lines)


def write_tex(path: Path, summaries: list[dict[str, object]], metadata: dict[str, str]) -> None:
    """Write a standalone LaTeX file with a clearly marked manuscript block."""

    total_cases = sum(int(row["cases"]) for row in summaries)
    if metadata.get("generator") == "correlated-top":
        generator_text = (
            r"For each candidate instance, \(R\in\Rbb^{n\times m}\) was sampled "
            r"with independent standard normal entries.  Let \(u_1\in\Rbb^n\) "
            r"and \(v_1\in\Rbb^m\) be the leading left and right singular "
            r"vectors of \(R\).  We then set \(c=\alpha v_1+\sigma g\) and "
            r"\(d=\alpha u_1+\sigma h\), "
            r"where \(g\) and \(h\) have independent standard normal entries, "
            rf"\(\alpha={metadata.get('generator_alpha', '3.0')}\), and "
            rf"\(\sigma={metadata.get('generator_sigma', '0.1')}\).  Finally, "
            r"the triple \((c,d,R)\) was divided by "
            r"\(\max\{\norm{c},\norm{d}\}\), so that "
            r"\(\max\{\norm{c},\norm{d}\}=1\)."
        )
        screen = metadata.get("prefilter_relative_gap") or "0.1"
        prefilter_text = (
            r"For each size we scanned seeds starting from zero and selected the "
            rf"first ten candidates for which the Shor relative gap "
            rf"\((z_{{\rm exact}}-z_{{\rm Shor}})/|z_{{\rm Shor}}|\) "
            rf"exceeded \({screen}\)."
        )
    else:
        generator_text = (
            r"For each instance the entries of \(c\in\Rbb^m\), \(d\in\Rbb^n\), "
            r"and \(R\in\Rbb^{n\times m}\) were sampled independently from the "
            r"standard normal distribution, and then the triple \((c,d,R)\) "
            r"was divided by \(\max\{\norm{c},\norm{d}\}\)."
        )
        prefilter_text = ""

    setup = r"""
\noindent We generated __TOTAL_CASES__ random instances of \eqref{eq:bilinear} with
\(\Ecal_x=\{x\in\Rbb^m\suchthat \norm{x}\le 1\}\) and
\(\Ecal_y=\{y\in\Rbb^n\suchthat \norm{y}\le 1\}\).  __GENERATOR_TEXT__
__PREFILTER_TEXT__  We compared the Shor SDP relaxation, Full SEP, Lazy SEP,
the Dual LOP/TRS bisection method, and Gurobi.
Gurobi was run after the three conic methods with explicit bounds
\(-1\le x_j\le 1\), \(-1\le y_i\le 1\) for each \(i\) and \(j\), the two
ball constraints, and time limit \(\max\{5,3\,t_{\max}\}\), where
\(t_{\max}\) is the maximum time used by the three conic methods on that
instance.  Lazy SEP added at most 50 violated coordinate skew-skew equalities
in each outer-approximation round, using violation tolerance \(10^{-7}\).
The Dual LOP/TRS method used bisection tolerance \(10^{-6}\).
""".strip()
    setup = (
        setup.replace("__TOTAL_CASES__", str(total_cases))
        .replace("__GENERATOR_TEXT__", generator_text)
        .replace("__PREFILTER_TEXT__", prefilter_text)
    )
    timing_text = r"""
The median times required by different methods on these test instances are
given in Table~\ref{tab:bilinear_times}.  In the table, the column
\(n\times m\) gives the dimensions of \(y\) and \(x\), respectively.  Ten
cases were run for each size.  The Shor, Full SEP, Lazy SEP, LOP/TRS, and
Gurobi columns give median wall-clock times for the five methods.  The
Gurobi opt column gives the number of instances solved to global optimality
by Gurobi within its adaptive time limit.
""".strip()

    diagnostics_text = r"""
In Table~\ref{tab:bilinear_diagnostics} we give additional numerical
diagnostics for the same instances as in Table~\ref{tab:bilinear_times}.  By
construction none of the Shor instances has a lower bound that agrees with
the exact conic value within \(10^{-5}\).  The ``Shor max
gap'' column gives the largest gap between the exact conic value and
the Shor lower bound.  The ``Conic max diff'' column is the largest
objective difference
among all the exact conic methods over all instances of that size.  The Lazy
SEP columns give the median/maximum outer-approximation rounds, total added
coordinate skew-skew cuts and the largest final coordinate skew-skew equation
violation.  The
LOP/TRS columns report the median/maximum TRS separation oracle solves for
the Dual LOP/TRS method as well as the maximum final bisection
bracket width.
""".strip()

    summary_text = r"""
To summarize these results, Lazy SEP is slower than Full SEP on smaller
instances due to the overhead of checking for violated skew-skew constraints
and repeated conic solves, but the time for Full SEP blows up faster as
problem size increases.  The Dual LOP/TRS bisection algorithm is very fast
and robust and scales much better than Full SEP or Lazy SEP.  These problems
are difficult for Gurobi, even with the explicit variable bounds added, and
cannot be solved in time competitive to the conic methods.
""".strip()

    paper_insert = "\n\n".join(
        [
            setup,
            timing_table(summaries),
            timing_text,
            diagnostics_text,
            diagnostics_table(summaries),
            summary_text,
        ]
    )
    content = (
        r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}

% Standalone macro fallbacks matching the notation used in paper/Lorentz.tex.
\providecommand{\Rbb}{\mathbb{R}}
\providecommand{\Ecal}{\mathcal{E}}
\providecommand{\suchthat}{:\,}
\providecommand{\norm}[1]{\left\lVert #1\right\rVert}

\begin{document}
\section*{Pure Bilinear Computational Results}

The displayed formulation below is included only so that this file compiles
independently.  Copy only the block marked ``BEGIN PAPER INSERT'' into
paper/Lorentz.tex.
\begin{equation}
\label{eq:bilinear}
\min\{c^\top x+d^\top y+y^\top Rx:\ x\in\Ecal_x,\ y\in\Ecal_y\}.
\end{equation}

% ======================== BEGIN PAPER INSERT ========================
"""
        + paper_insert
        + r"""
% ========================= END PAPER INSERT =========================

\end{document}
"""
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    args = parser.parse_args()

    rows = load_rows(args.raw)
    summaries = summarize(rows, tight_tol=PAPER_TIGHT_REL_TOL)
    metadata = experiment_metadata(rows)
    write_summary_csv(args.summary, summaries)
    write_tex(args.tex, summaries, metadata)
    print(f"wrote {args.summary}")
    print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
