#!/usr/bin/env python3
"""Assemble the current computational note from generated table fragments."""

from __future__ import annotations

from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
DEFAULT_TEX = TABLES / "current_comp_results_note.tex"


def extract_insert(path: Path) -> str:
    """Return the copy-paste block from a generated standalone fragment."""

    text = path.read_text()
    begin = "% ======================== BEGIN PAPER INSERT ========================"
    end = "% ========================= END PAPER INSERT ========================="
    try:
        return text.split(begin, 1)[1].split(end, 1)[0].strip()
    except IndexError as exc:
        raise RuntimeError(f"{path} does not contain paper insert markers") from exc


def extract_large_body(path: Path) -> str:
    """Return the generated large-panel body without its standalone wrapper."""

    text = path.read_text()
    try:
        body = text.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"{path} does not look like a standalone LaTeX file") from exc
    return body.replace(r"\section*{Large-Instance Computational Fragments}", "").strip()


def main() -> int:
    bilinear = extract_insert(TABLES / "bilinear_results_fragment.tex")
    maxdist = extract_insert(TABLES / "maxdist_results_fragment.tex")
    ttrs = extract_insert(TABLES / "ttrs_results_fragment.tex")
    large = extract_large_body(TABLES / "large_instance_results_fragment.tex")
    noxious = extract_insert(TABLES / "noxious_results_fragment.tex")
    today = date.today()
    date_text = f"{today.strftime('%B')} {today.day}, {today.year}"

    content = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs}}
\usepackage[hidelinks]{{hyperref}}

\providecommand{{\Rbb}}{{\mathbb{{R}}}}
\providecommand{{\Ecal}}{{\mathcal{{E}}}}
\providecommand{{\suchthat}}{{:\,}}
\providecommand{{\norm}}[1]{{\left\lVert #1\right\rVert}}

\title{{Current Computational Results for Lorentz/SEP Paper}}
\author{{}}
\date{{{date_text}}}

\begin{{document}}
\maketitle

\noindent This note collects the current generated computational fragments in
the style of the latest manuscript draft.  The raw CSV files have been
refreshed so that every Lazy SEP row uses coordinate skew-skew cuts and reports
coordinate skew-skew residuals.  In the max-distance and TTRS tables, each SDP
gap uses the feasible point recovered from that method's own first column as
its upper bound; neither Gurobi nor a beta reference supplies an SDP upper
bound.  Throughout the note, ``tight'' has the single definition that this
method-specific relative gap is at most \(10^{{-5}}\).  The larger generated
panels also calculate relative rank-one residuals and recovered-point
feasibility violations, but those diagnostics do not enter the tight count.

\paragraph{{Displayed formulations.}}
The following equations are included so the generated fragments compile outside
\texttt{{paper/Lorentz.tex}}.
\begin{{equation}}
\label{{eq:bilinear}}
\min\{{c^\top x+d^\top y+y^\top Rx:\ x\in\Ecal_x,\ y\in\Ecal_y\}}.
\end{{equation}}
\begin{{equation}}
\label{{eq:quadratic}}
\min\{{c^\top x+d^\top y+x^\top Qx+y^\top Py+y^\top Rx:\ x\in\Ecal_x,\ y\in\Ecal_y\}}.
\end{{equation}}
\begin{{equation}}
\label{{eq:shor}}
\min\{{\langle C,U\rangle:\ \operatorname{{tr}}X\le 1,\ \operatorname{{tr}}Y\le 1,\ U\succeq 0\}}.
\end{{equation}}
\begin{{equation}}
\label{{eq:fullsep}}
\eqref{{eq:shor}}\ \text{{with}}\ Z=\mathcal{{W}}^*(T),\ \langle T,X\rangle=0\ \forall X\in\mathcal{{A}}(n)\otimes\mathcal{{A}}(m),\ T\succeq 0.
\end{{equation}}

\section{{Pure Bilinear Instances}}
{bilinear}

\section{{Max-Distance Instances}}
{maxdist}

\section{{Ball-Constraints TTRS Instances}}
{ttrs}

\section{{Larger Generated Instances}}
{large}

\section{{Noxious-Facility Instances}}
{noxious}

\end{{document}}
"""
    DEFAULT_TEX.write_text(content)
    print(f"wrote {DEFAULT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
