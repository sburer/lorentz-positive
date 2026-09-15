#!/usr/bin/env python3
"""Regenerate the paper's tables and named examples from the committed raw data.

This is the fast path of the reproduction (seconds, plus the single
counterexample SDP solve).  It

1. rebuilds every LaTeX table fragment in outputs/tables from the raw CSVs in
   outputs/raw (and the combined current_comp_results_note.tex);
2. re-solves the named Section 4.2 counterexample and rewrites its
   verification log in outputs/examples;
3. compiles standalone PDFs when Tectonic is installed; and
4. runs the checks: inventory vs. manuscript labels, fragments vs. inventory,
   every number in every manuscript table vs. the regenerated fragment, and
   every named example value vs. the manuscript and its source.

It deliberately does not launch the long solver panels that produce the raw
CSVs; those commands are listed per family in README.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
EXAMPLES = OUTPUTS / "examples"

TABLE_BUILDERS = {
    "bilinear": ROOT / "make_bilinear_tables.py",
    "maxdist": ROOT / "make_maxdist_tables.py",
    "ttrs": ROOT / "make_ttrs_tables.py",
    "large": ROOT / "make_large_instance_tables.py",
    "noxious": ROOT / "make_noxious_tables.py",
}

TABLE_TEX_FILES = [
    TABLES / "bilinear_results_fragment.tex",
    TABLES / "maxdist_results_fragment.tex",
    TABLES / "ttrs_results_fragment.tex",
    TABLES / "large_instance_results_fragment.tex",
    TABLES / "noxious_results_fragment.tex",
    TABLES / "current_comp_results_note.tex",
]

EXAMPLE_TEX_FILES = [
    EXAMPLES / "ttrs_sep_beta_counterexample_note.tex",
]


def _python_env() -> dict[str, str]:
    """Return an environment that can import the local paper_code package."""

    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    paths = [str(ROOT)]
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_python(script: Path, *args: str, output: Path | None = None) -> None:
    """Run a paper-code Python script with the current interpreter."""

    command = [sys.executable, str(script), *args]
    print("running:", " ".join(command), flush=True)
    if output is None:
        subprocess.run(command, cwd=ROOT, env=_python_env(), check=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        subprocess.run(
            command,
            cwd=ROOT,
            env=_python_env(),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def compile_tex(tex_path: Path) -> None:
    """Compile a standalone LaTeX file with Tectonic."""

    tectonic = shutil.which("tectonic")
    if tectonic is None:
        print(f"skipping PDF for {tex_path}: tectonic is not installed", flush=True)
        return
    print(f"compiling: {tex_path}", flush=True)
    subprocess.run([tectonic, tex_path.name], cwd=tex_path.parent, check=True)


def regenerate_tables(panels: list[str], *, skip_pdf: bool) -> None:
    """Rebuild selected table fragments from existing raw CSV files."""

    for panel in panels:
        run_python(TABLE_BUILDERS[panel])

    # The combined note depends on the individual fragments, so it is always
    # built after any selected table panel.
    run_python(ROOT / "make_current_comp_note.py")

    if not skip_pdf:
        for tex_path in TABLE_TEX_FILES:
            if tex_path.exists():
                compile_tex(tex_path)


def regenerate_examples(*, skip_pdf: bool) -> None:
    """Verify the named examples and refresh their shareable artifacts."""

    run_python(
        ROOT / "verify_ttrs_sep_beta_counterexample.py",
        output=EXAMPLES / "ttrs_sep_beta_counterexample_verification.txt",
    )
    if not skip_pdf:
        for tex_path in EXAMPLE_TEX_FILES:
            if tex_path.exists():
                compile_tex(tex_path)


CHECKS = [
    "check_paper_inventory.py",
    "check_table_fragments.py",
    "check_raw_provenance.py",
    "check_paper_tables.py",
    "check_paper_examples.py",
]

# The checks that read Lorentz.tex; they accept --manuscript and exit with
# CHECK_SKIPPED when the file is absent.
MANUSCRIPT_CHECKS = {
    "check_paper_inventory.py",
    "check_paper_tables.py",
    "check_paper_examples.py",
}


# Exit status the manuscript-facing checks use when Lorentz.tex is not
# available (for instance, when this directory is checked out on its own).
CHECK_SKIPPED = 3


def run_checks(manuscript: str | None) -> bool:
    """Run every consistency check; report all failures, return overall pass."""

    failed: list[str] = []
    skipped: list[str] = []
    for name in CHECKS:
        extra = ["--manuscript", manuscript] if manuscript and name in MANUSCRIPT_CHECKS else []
        try:
            run_python(ROOT / name, *extra)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == CHECK_SKIPPED:
                skipped.append(name)
            else:
                failed.append(name)
    print()
    if failed:
        print("CHECKS FAILED: " + ", ".join(failed))
        print("See [[known_data_issues]] in paper_experiments.toml for known causes.")
        return False
    if skipped:
        print("Data checks passed; manuscript checks skipped: " + ", ".join(skipped))
        print("Pass --manuscript /path/to/Lorentz.tex to run them.")
        return True
    print("All checks passed.")
    return True


def check_submodule() -> None:
    """Fail early with a clone hint if the ballconstraints submodule is absent."""

    marker = ROOT / "external" / "ballconstraints" / "src" / "define_functions.py"
    if marker.exists() or os.environ.get("BALLCONSTRAINTS_DIR"):
        return
    sys.exit(
        f"{ROOT / 'external' / 'ballconstraints'} is empty.  Run\n"
        "    git submodule update --init\n"
        "from the repository root (or set BALLCONSTRAINTS_DIR) and retry."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tables-only", action="store_true")
    mode.add_argument("--examples-only", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Run only the manuscript/inventory/table/example checks.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        choices=sorted(TABLE_BUILDERS),
        help="Regenerate only this table panel. May be repeated.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write TEX/CSV artifacts but do not compile standalone PDFs.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Do not run inventory and table-fragment checks at the end.",
    )
    parser.add_argument(
        "--manuscript",
        default=None,
        help="Path to Lorentz.tex for the manuscript checks "
        "(default: the manuscript entry in paper_experiments.toml).",
    )
    args = parser.parse_args(argv)

    panels = args.panel or list(TABLE_BUILDERS)
    do_tables = not args.check_only and (args.tables_only or args.all or not args.examples_only)
    do_examples = not args.check_only and (args.examples_only or args.all or not args.tables_only)

    if do_examples:
        check_submodule()
    if do_tables:
        regenerate_tables(panels, skip_pdf=args.skip_pdf)
    if do_examples:
        regenerate_examples(skip_pdf=args.skip_pdf)
    if not args.skip_checks and not run_checks(args.manuscript):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
