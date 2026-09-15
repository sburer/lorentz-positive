#!/usr/bin/env python3
"""Generate LaTeX table fragments from the Section 4.1 max-distance CSV.

The tables mirror the manuscript's bilinear results: a timing table plus a
diagnostics table.  Since no relaxation is assumed exact for the max-distance
problem, diagnostics are stated as tight counts and worst remaining relative
gaps using each method's own recovered feasible point.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

from paper_code.metrics import PAPER_TIGHT_REL_TOL, relative_gap

from make_bilinear_tables import _float, _fmt_sci, _fmt_time, load_rows


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEFAULT_RAW = OUTPUTS / "raw" / "maxdist_raw.csv"
DEFAULT_SUMMARY = OUTPUTS / "summary" / "maxdist_summary.csv"
DEFAULT_TEX = OUTPUTS / "tables" / "maxdist_results_fragment.tex"

METHODS = ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP", "Gurobi")
RELAXATIONS = ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP")


def summarize(
    rows: list[dict[str, str]],
    *,
    tight_tol: float = PAPER_TIGHT_REL_TOL,
    recovery_feas_tol: float = 1e-7,
) -> list[dict[str, object]]:
    """Summarize rows using one relative-gap definition of tightness."""

    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["panel"], int(row["n"]), int(row["m"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (panel, n, m), group in sorted(groups.items(), key=lambda item: item[0][1]):
        seeds = sorted({int(row["seed"]) for row in group})
        by_method = {
            method: {int(row["seed"]): row for row in group if row["method"] == method}
            for method in METHODS
        }

        shor_gaps: list[float] = []
        shor_tight: int = 0
        kron_gaps: list[float] = []
        kron_tight: int = 0
        lazy_kron_gaps: list[float] = []
        lazy_kron_tight: int = 0
        lazy_kron_minus_full: list[float] = []
        lazy_kron_violations: list[float] = []
        lazy_kron_cuts: list[float] = []
        lazy_kron_rounds: list[float] = []
        sep_gaps: list[float] = []
        lazy_sep_gaps: list[float] = []
        lazy_minus_full: list[float] = []
        sep_tight: int = 0
        lazy_sep_tight: int = 0
        lazy_skews: list[float] = []
        lazy_cuts: list[float] = []
        lazy_rounds: list[float] = []

        for seed in seeds:
            shor_row = by_method["Shor"].get(seed)
            if shor_row:
                gap = _recovered_gap(shor_row, recovery_feas_tol)
                if gap is not None:
                    shor_gaps.append(gap)
                    shor_tight += gap <= tight_tol

            bounds: dict[str, float | None] = {}
            for method in ("KRON", "Lazy KRON", "Full SEP", "Lazy SEP"):
                row = by_method[method].get(seed)
                bounds[method] = _float(row["lower_bound"]) if row else None

            if bounds["KRON"] is not None:
                row = by_method["KRON"].get(seed)
                if row:
                    gap = _recovered_gap(row, recovery_feas_tol)
                    if gap is not None:
                        kron_gaps.append(gap)
                        kron_tight += gap <= tight_tol
            if bounds["Lazy KRON"] is not None:
                row = by_method["Lazy KRON"].get(seed)
                if row:
                    gap = _recovered_gap(row, recovery_feas_tol)
                    if gap is not None:
                        lazy_kron_gaps.append(gap)
                        lazy_kron_tight += gap <= tight_tol
            sep_row = by_method["Full SEP"].get(seed) or by_method["Lazy SEP"].get(seed)
            if sep_row:
                sep_gap = _recovered_gap(sep_row, recovery_feas_tol)
                if sep_gap is not None:
                    sep_gaps.append(sep_gap)
                    sep_tight += sep_gap <= tight_tol
            if bounds["Full SEP"] is not None and bounds["Lazy SEP"] is not None:
                lazy_minus_full.append(abs(bounds["Lazy SEP"] - bounds["Full SEP"]))
            if bounds["KRON"] is not None and bounds["Lazy KRON"] is not None:
                lazy_kron_minus_full.append(
                    abs(bounds["Lazy KRON"] - bounds["KRON"])
                )

            lazy_kron_row = by_method["Lazy KRON"].get(seed)
            if lazy_kron_row:
                kron_violation = _float(
                    lazy_kron_row.get("relative_max_kron_violation")
                )
                if kron_violation is None:
                    kron_violation = _float(lazy_kron_row.get("max_kron_violation"))
                if kron_violation is not None:
                    lazy_kron_violations.append(kron_violation)
                for key, sink in (
                    ("n_cuts", lazy_kron_cuts),
                    ("n_rounds", lazy_kron_rounds),
                ):
                    value = _float(lazy_kron_row.get(key))
                    if value is not None:
                        sink.append(value)

            lazy_row = by_method["Lazy SEP"].get(seed)
            if lazy_row:
                gap = _recovered_gap(lazy_row, recovery_feas_tol)
                if gap is not None:
                    lazy_sep_gaps.append(gap)
                    lazy_sep_tight += gap <= tight_tol
                skew = _float(lazy_row.get("relative_max_skew_violation"))
                if skew is None:
                    skew = _float(lazy_row.get("max_skew_violation"))
                if skew is not None:
                    lazy_skews.append(skew)
                for key, sink in (
                    ("n_cuts", lazy_cuts),
                    ("n_rounds", lazy_rounds),
                ):
                    value = _float(lazy_row.get(key))
                    if value is not None:
                        sink.append(value)

        summary: dict[str, object] = {
            "panel": panel,
            "n": n,
            "m": m,
            "cases": len(seeds),
            "shor_tight": shor_tight,
            "max_shor_gap": max(shor_gaps) if shor_gaps else None,
            "kron_tight": kron_tight,
            "max_kron_gap": max(kron_gaps) if kron_gaps else None,
            "lazy_kron_tight": lazy_kron_tight,
            "max_lazy_kron_gap": max(lazy_kron_gaps)
            if lazy_kron_gaps
            else None,
            "max_lazy_kron_minus_full": max(lazy_kron_minus_full)
            if lazy_kron_minus_full
            else None,
            "max_lazy_kron_violation": max(lazy_kron_violations)
            if lazy_kron_violations
            else None,
            "median_lazy_kron_rounds": median(lazy_kron_rounds)
            if lazy_kron_rounds
            else None,
            "max_lazy_kron_rounds": max(lazy_kron_rounds)
            if lazy_kron_rounds
            else None,
            "median_lazy_kron_cuts": median(lazy_kron_cuts)
            if lazy_kron_cuts
            else None,
            "max_lazy_kron_cuts": max(lazy_kron_cuts)
            if lazy_kron_cuts
            else None,
            "sep_tight": sep_tight,
            "max_sep_gap": max(sep_gaps) if sep_gaps else None,
            "lazy_sep_tight": lazy_sep_tight,
            "max_lazy_sep_gap": max(lazy_sep_gaps) if lazy_sep_gaps else None,
            "max_lazy_minus_full": max(lazy_minus_full) if lazy_minus_full else None,
            "max_lazy_skew_violation": max(lazy_skews) if lazy_skews else None,
            "median_lazy_rounds": median(lazy_rounds) if lazy_rounds else None,
            "max_lazy_rounds": max(lazy_rounds) if lazy_rounds else None,
            "median_lazy_cuts": median(lazy_cuts) if lazy_cuts else None,
            "max_lazy_cuts": max(lazy_cuts) if lazy_cuts else None,
            "gurobi_closed": sum(
                1 for row in by_method["Gurobi"].values() if row["status"] == "OPTIMAL"
            ),
        }
        for method in METHODS:
            times = [_float(row["runtime"]) for row in by_method[method].values()]
            times = [t for t in times if t is not None]
            summary[f"{method}_median_time"] = median(times) if times else None
            summary[f"{method}_max_time"] = max(times) if times else None
        summaries.append(summary)
    return summaries


def _recovered_gap(row: dict[str, str], feasibility_tolerance: float) -> float | None:
    """Return one method's own recovered-point gap after checking feasibility."""

    violation = _float(row.get("primal_feasibility_violation"))
    if violation is None:
        raise ValueError(
            f"{row.get('method')} row n={row.get('n')} seed={row.get('seed')} "
            "does not contain recovered-point feasibility diagnostics"
        )
    if violation > feasibility_tolerance:
        raise ValueError(
            f"infeasible recovered point for {row.get('method')} n={row.get('n')} "
            f"seed={row.get('seed')}: violation={violation:.3e}"
        )
    return relative_gap(_float(row.get("upper_bound")), _float(row.get("lower_bound")))


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def timing_table(summaries: list[dict[str, object]]) -> str:
    has_gurobi = any(row.get("Gurobi_median_time") is not None for row in summaries)
    if has_gurobi:
        tabular = "\\begin{tabular}{c r r r r r r r r}"
        header = (
            "$n$ & cases & Shor & KRON & Lazy KRON & Full SEP "
            "& Lazy SEP & Gurobi & Gurobi opt \\\\"
        )
    else:
        tabular = "\\begin{tabular}{c r r r r r r}"
        header = (
            "$n$ & cases & Shor & KRON & Lazy KRON & Full SEP "
            "& Lazy SEP \\\\"
        )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Median wall-clock times for max-distance instances.}",
        "\\label{tab:maxdist_times}",
        "\\small",
        tabular,
        "\\hline",
        header,
        "\\hline",
    ]
    for row in summaries:
        cases = int(row["cases"])
        prefix = (
            f"{row['n']} & {cases} & "
            f"{_fmt_time(row['Shor_median_time'])} & "
            f"{_fmt_time(row['KRON_median_time'])} & "
            f"{_fmt_time(row['Lazy KRON_median_time'])} & "
            f"{_fmt_time(row['Full SEP_median_time'])} & "
            f"{_fmt_time(row['Lazy SEP_median_time'])}"
        )
        if has_gurobi:
            prefix += (
                f" & {_fmt_time(row['Gurobi_median_time'])} & "
                f"{int(row['gurobi_closed'])}"
            )
        lines.append(prefix + " \\\\")
    lines.extend(["\\hline", "\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def diagnostics_table(summaries: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Bound quality and diagnostics for max-distance instances.}",
        "\\label{tab:maxdist_diagnostics}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{1.6pt}",
        "\\tiny",
        "\\begin{tabular}{c|r r|r r|r r|r r r|r r r}",
        "\\hline",
        "& \\multicolumn{2}{c|}{Shor} "
        "& \\multicolumn{2}{c|}{KRON} "
        "& \\multicolumn{2}{c|}{SEP} "
        "& \\multicolumn{3}{c|}{Lazy KRON} "
        "& \\multicolumn{3}{c}{Lazy SEP} \\\\",
        "$n$ & tight & max gap & tight & max gap & tight & max gap "
        "& rounds & cuts & max viol. "
        "& rounds & cuts & max skew \\\\",
        "\\hline",
    ]
    for row in summaries:
        lazy_kron_rounds = (
            "--"
            if row["median_lazy_kron_rounds"] is None
            else f"{row['median_lazy_kron_rounds']:.0f}/{row['max_lazy_kron_rounds']:.0f}"
        )
        lazy_kron_cuts = (
            "--"
            if row["median_lazy_kron_cuts"] is None
            else f"{row['median_lazy_kron_cuts']:.0f}/{row['max_lazy_kron_cuts']:.0f}"
        )
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
        lines.append(
            f"{row['n']} & "
            f"{int(row['shor_tight'])} & "
            f"{_fmt_sci(row['max_shor_gap'])} & "
            f"{int(row['kron_tight'])} & "
            f"{_fmt_sci(row['max_kron_gap'])} & "
            f"{int(row['sep_tight'])} & "
            f"{_fmt_sci(row['max_sep_gap'])} & "
            f"{lazy_kron_rounds} & {lazy_kron_cuts} & "
            f"{_fmt_sci(row['max_lazy_kron_violation'])} & "
            f"{lazy_rounds} & {lazy_cuts} & "
            f"{_fmt_sci(row['max_lazy_skew_violation'])} \\\\"
        )
    lines.extend(["\\hline", "\\hline", "\\end{tabular}", "\\endgroup", "\\end{table}"])
    return "\n".join(lines)


def experiment_metadata(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    first = rows[0]
    # The largest seed retained is a factual measure of how deep the screen had
    # to scan, which is the honest way to report how rare these instances are.
    seeds = [int(row["seed"]) for row in rows if row.get("seed") not in (None, "")]
    return {
        "generator": first.get("generator", ""),
        "condition": first.get("condition", ""),
        "center_scale": first.get("center_scale", ""),
        "prefiltered": first.get("prefiltered", ""),
        "prefilter_relative_gap": first.get("prefilter_relative_gap", ""),
        "max_seed": str(max(seeds)) if seeds else "",
    }


def write_tex(
    path: Path, summaries: list[dict[str, object]], metadata: dict[str, str]
) -> None:
    total_cases = sum(int(row["cases"]) for row in summaries)
    has_gurobi = any(row.get("Gurobi_median_time") is not None for row in summaries)
    prefiltered = metadata.get("prefiltered") in ("True", "true", "1")
    if not prefiltered:
        prefiltered = metadata.get("prefilter_relative_gap") not in ("", "None", None)
    cases_per_size = max((int(row["cases"]) for row in summaries), default=0)
    max_seed = metadata.get("max_seed") or ""
    if not prefiltered:
        prefilter_text = (
            r"No filtering was applied: for each size the instances are simply "
            rf"the first {cases_per_size} seeds.  As the ``Shor tight'' column "
            r"of Table~\ref{tab:maxdist_diagnostics} shows, the Shor "
            r"relaxation \eqref{eq:shor} is already exact on most max-distance "
            r"instances generated this way."
        )
    else:
        deep_scan = (
            ""
            if max_seed == ""
            else (
                r"  Such instances are rare: retaining "
                rf"{cases_per_size} per size required scanning up to "
                rf"{int(max_seed) + 1} seeds."
            )
        )
        prefilter_text = (
            r"Because the Shor relaxation \eqref{eq:shor} is already exact on "
            r"most max-distance instances, the instances reported here were "
            r"selected by a rank-only screen, applied to seeds starting from zero.  The "
            r"screen first solves \eqref{eq:shor} and discards the instance "
            r"when the two largest eigenvalues of the optimal \(U\) satisfy "
            r"\(\lambda_1/\lambda_2>10^4\), treating a numerically negative "
            r"\(\lambda_2\) by using \(|\lambda_2|\) in the denominator.  Such "
            r"a solution is effectively rank one, in which case \(U\) is itself "
            r"a feasible point attaining the bound and the relaxation is exact.  "
            r"For the remainder, the screen re-solves the Shor relaxation with "
            r"the Shor objective held fixed up to solver tolerance and with a "
            r"random linear objective; this exposes rank-one optima when the "
            r"original solve returned a higher-rank point on a flat optimal "
            r"face.  If this second solve is still not effectively rank one, "
            r"the instance is retained; no feasible reference value is computed "
            r"during this preselection step." + deep_scan + r"  The reported results are "
            r"therefore conditional on the Shor relaxation not being certified "
            r"rank one by this screen, and are "
            r"not representative of the class as a whole."
        )

    if has_gurobi:
        gurobi_text = (
            r"The retained Gurobi runs used explicit bounds "
            r"\(-1\le x_j\le 1\), \(-1\le y_i\le 1\), the two ball constraints, "
            r"and time limit \(\max\{5,3\,t_{\max}\}\), where \(t_{\max}\) is the "
            r"maximum conic-method time available when that Gurobi run was "
            r"started.  These Gurobi rows were not rerun after the conic refresh "
            r"and are reported only as an independent comparison."
        )
    else:
        gurobi_text = r"Gurobi method rows are not included in this output."

    setup = r"""
\noindent We generated __TOTAL_CASES__ random max-distance instances, that is,
instances of \eqref{eq:quadratic} of the form
\[
\max\{\norm{(Ax+a)-(By+b)}^2\suchthat \norm{x}\le 1,\ \norm{y}\le 1\},
\]
with \(n=m\), so that \(\Ecal_x=\{Ax+a\suchthat\norm{x}\le 1\}\) and
\(\Ecal_y=\{By+b\suchthat\norm{y}\le 1\}\) are the two ellipsoids whose
farthest pair of points is sought.  The maps \(A\) and \(B\) were generated as
__GENERATOR_TEXT__, and the centers \(a\) and \(b\) have independent normal
entries.  __PREFILTER_TEXT__
We compared the Shor SDP relaxation \eqref{eq:shor}, Shor strengthened by the
KRON constraint \(K(Z)\succeq 0\), Lazy KRON, Full SEP \eqref{eq:fullsep}, and
Lazy SEP.
Unlike the bilinear problem of Section 3, none of these relaxations is
guaranteed exact for the class \eqref{eq:quadratic}.  For each SDP method, we extracted
\((x,y)\) from the first column of its optimal moment matrix, verified the two
ball constraints to tolerance \(10^{-7}\), and evaluated the original
minimization objective at that point.  This gives a method-specific feasible
upper bound; no Gurobi incumbent or point recovered by another method is used
to calculate an SDP gap.  __GUROBI_TEXT__  Lazy KRON added at most 50 violated KRON cuts in each
outer-approximation round.  Lazy SEP added at most 50 violated coordinate
skew-skew equalities in each round.  Both used relative violation tolerance
\(10^{-7}\).
""".strip()
    setup = (
        setup.replace("__TOTAL_CASES__", str(total_cases))
        .replace(
            "__GENERATOR_TEXT__",
            _generator_phrase(metadata),
        )
        .replace("__PREFILTER_TEXT__", prefilter_text)
        .replace("__GUROBI_TEXT__", gurobi_text)
    )

    if has_gurobi:
        timing_text = r"""
The median times required by the different methods are given in
Table~\ref{tab:maxdist_times}.  In the table, the column \(n\) gives the
dimensions of \(y\) and \(x\).  The Gurobi opt column gives the
number of instances solved to global optimality by Gurobi within its adaptive
time limit.  The Full KRON solve for \(n=15\), seed 278, used two MOSEK
threads after repeated memory-related termination with the default parallel
setting; its formulation and tolerances were unchanged.
""".strip()
    else:
        timing_text = r"""
The median times required by the conic methods are given in
Table~\ref{tab:maxdist_times}.  In the table, the column \(n\) gives the
dimensions of \(y\) and \(x\).  Gurobi timings are omitted because
this refresh did not run Gurobi as a full solution method.
""".strip()

    lazy_kron_differences = [
        row for row in summaries if row["lazy_kron_tight"] != row["kron_tight"]
    ]
    lazy_sep_differences = [
        row for row in summaries if row["lazy_sep_tight"] != row["sep_tight"]
    ]
    if lazy_kron_differences:
        comparisons = "; ".join(
            rf"at \(n={row['n']}\), {int(row['lazy_kron_tight'])}/{int(row['cases'])} "
            rf"versus {int(row['kron_tight'])}/{int(row['cases'])} "
            rf"(Lazy KRON max gap {_fmt_sci(row['max_lazy_kron_gap'])})"
            for row in lazy_kron_differences
        )
        lazy_kron_text = (
            r"Lazy KRON matched the Full KRON tight count at all other sizes, but "
            + comparisons
            + "."
        )
    else:
        lazy_kron_text = r"Lazy KRON matched the Full KRON tight count at every size."
    if lazy_sep_differences:
        comparisons = "; ".join(
            rf"at \(n={row['n']}\), {int(row['lazy_sep_tight'])}/{int(row['cases'])} "
            rf"versus {int(row['sep_tight'])}/{int(row['cases'])}"
            for row in lazy_sep_differences
        )
        lazy_sep_text = r"Lazy SEP and Full SEP differed in tight count: " + comparisons + "."
    else:
        lazy_sep_text = r"Lazy SEP matched the Full SEP tight count at every size."
    lazy_recovery_text = lazy_kron_text + "  " + lazy_sep_text

    diagnostics_text = r"""
Table~\ref{tab:maxdist_diagnostics} reports bound quality on the same instances.
For Shor, Full KRON, and Full SEP (labeled SEP), the tight column counts instances whose
method-specific relative recovered-point gap is at most \(10^{-5}\), while max
gap gives the largest such gap over the instances of that size.  If
\(z_{\rm sdp}\) is the SDP lower bound and \(z_{\rm rec}\) is that method's
recovered feasible upper bound, this gap is
\((z_{\rm rec}-z_{\rm sdp})/
\max\{1,|z_{\rm rec}|,|z_{\rm sdp}|\}\).  __LAZY_RECOVERY_TEXT__  The Lazy KRON and Lazy SEP columns give the
median/maximum outer-approximation rounds and total added cuts.  The max viol.
column is the largest final relative KRON cut violation, while max skew is the
largest final relative coordinate skew-skew equation violation.
""".strip().replace("__LAZY_RECOVERY_TEXT__", lazy_recovery_text)

    paper_insert = "\n\n".join(
        [
            setup,
            timing_table(summaries),
            timing_text,
            diagnostics_text,
            diagnostics_table(summaries),
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
\section*{Max-Distance Computational Results}

The displayed formulations below are included only so that this file compiles
independently.  Copy only the block marked ``BEGIN PAPER INSERT'' into
paper/Lorentz.tex.
\begin{equation}
\label{eq:quadratic}
\min\{c^\top x+d^\top y+x^\top Qx+y^\top Py+y^\top Rx:\ x\in\Ecal_x,\ y\in\Ecal_y\}.
\end{equation}
\begin{equation}
\label{eq:shor}
\min\{\langle C,U\rangle:\ \operatorname{tr}X\le 1,\ \operatorname{tr}Y\le 1,\ U\succeq 0\}.
\end{equation}
\begin{equation}
\label{eq:fullsep}
\eqref{eq:shor}\ \text{with}\ Z=\mathcal{W}^*(T),\ \langle T,X\rangle=0\ \forall X\in\mathcal{A}(n)\otimes\mathcal{A}(m),\ T\succeq 0.
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


def _generator_phrase(metadata: dict[str, str]) -> str:
    generator = metadata.get("generator", "")
    condition = metadata.get("condition", "5.0")
    phrases = {
        "balls": r"random scalar multiples of the identity (both ellipsoids are balls)",
        "axis": (
            r"random diagonal maps with singular values log-uniform on "
            rf"\([1,{condition}]\) (both ellipsoids are axis-aligned)"
        ),
        "ellipsoids": (
            r"\(U\Sigma V^\top\) with Haar-random orthogonal \(U,V\) and singular "
            rf"values log-uniform on \([1,{condition}]\)"
        ),
        "tied": r"maps with two nearly equal dominant axes",
        "spiked": r"maps with a dominant two-dimensional range",
        "graded": r"maps with oppositely graded singular values",
    }
    return phrases.get(generator, r"described in the accompanying code")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--recovery-feas-tol", type=float, default=1e-7)
    args = parser.parse_args()

    rows = load_rows(args.raw)
    summaries = summarize(
        rows,
        tight_tol=PAPER_TIGHT_REL_TOL,
        recovery_feas_tol=args.recovery_feas_tol,
    )
    metadata = experiment_metadata(rows)
    write_summary_csv(args.summary, summaries)
    write_tex(args.tex, summaries, metadata)
    print(f"wrote {args.summary}")
    print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
