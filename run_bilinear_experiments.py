#!/usr/bin/env python3
"""Run the Section 3.1 pure bilinear computational panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper_code.dual_lop import solve_dual_lop
from paper_code.full_sep import solve_full_sep
from paper_code.gurobi_qp import solve_gurobi_qp
from paper_code.instances import generate_named_instance, paper_panel
from paper_code.lazy_sep import solve_lazy_sep
from paper_code.metrics import relative_error, relative_gap, value_scale
from paper_code.shor import solve_shor
from paper_code.sep_utils import MethodResult


OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT = OUTPUTS / "raw" / "bilinear_raw.csv"
DEFAULT_SELECTED_PANEL = OUTPUTS / "raw" / "bilinear_selected_seeds.csv"

FIELDNAMES = [
    "panel",
    "n",
    "m",
    "seed",
    "generator",
    "generator_alpha",
    "generator_sigma",
    "data_scale",
    "prefilter_relative_gap",
    "screen_relative_gap",
    "screen_absolute_gap",
    "method",
    "status",
    "value",
    "lower_bound",
    "upper_bound",
    "exact_value",
    "upper_gap",
    "lower_gap",
    "relative_upper_gap",
    "relative_lower_gap",
    "runtime",
    "rank1_residual",
    "relative_rank1_residual",
    "pd_gap",
    "relative_pd_gap",
    "shor_gap",
    "relative_shor_gap",
    "x_trace",
    "y_trace",
    "max_skew_violation",
    "fro_skew_violation",
    "relative_max_skew_violation",
    "relative_fro_skew_violation",
    "lazy_cut_strategy",
    "n_cuts",
    "n_iters",
    "n_rounds",
    "full_constraint_count",
    "oracle_solves",
    "dual_bracket_gap",
    "relative_dual_bracket_gap",
    "oracle_margin",
    "gurobi_time_limit",
    "gurobi_mip_gap",
    "gurobi_sol_count",
    "rank1_repairs",
    "rank1_residual_initial_max",
]

METHOD_ALIASES = {
    "shor": "Shor",
    "full": "Full SEP",
    "lazy": "Lazy SEP",
    "dual": "Dual LOP/TRS",
    "gurobi": "Gurobi",
}
DEFAULT_METHOD_ALIASES = ("shor", "full", "lazy", "dual", "gurobi")
CONIC_METHODS = ("Full SEP", "Lazy SEP", "Dual LOP/TRS")


def result_row(
    panel: str,
    n: int,
    m: int,
    seed: int,
    result: MethodResult,
    *,
    exact_value: float | None,
    args: argparse.Namespace,
    screen_relative_gap: float | None,
    screen_absolute_gap: float | None,
) -> dict[str, object]:
    """Convert one method result to a flat CSV row."""

    extra = result.extra
    upper_gap = (
        None
        if exact_value is None or result.upper_bound is None
        else float(result.upper_bound - exact_value)
    )
    lower_gap = (
        None
        if exact_value is None or result.lower_bound is None
        else float(exact_value - result.lower_bound)
    )
    relative_upper_gap = relative_error(result.upper_bound, exact_value)
    relative_lower_gap = relative_gap(exact_value, result.lower_bound)
    pd_gap = extra.get("pd_gap")
    return {
        "panel": panel,
        "n": n,
        "m": m,
        "seed": seed,
        "generator": args.generator,
        "generator_alpha": args.generator_alpha,
        "generator_sigma": args.generator_sigma,
        "data_scale": extra.get("data_scale"),
        "prefilter_relative_gap": args.prefilter_relative_gap,
        "screen_relative_gap": screen_relative_gap,
        "screen_absolute_gap": screen_absolute_gap,
        "method": result.method,
        "status": result.status,
        "value": result.value,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "exact_value": exact_value,
        "upper_gap": upper_gap,
        "lower_gap": lower_gap,
        "relative_upper_gap": relative_upper_gap,
        "relative_lower_gap": relative_lower_gap,
        "runtime": result.runtime,
        "rank1_residual": extra.get("rank1_residual"),
        "relative_rank1_residual": extra.get("relative_rank1_residual"),
        "pd_gap": pd_gap,
        "relative_pd_gap": None
        if pd_gap is None or result.value is None or result.lower_bound is None
        else abs(float(pd_gap)) / value_scale(result.value, result.lower_bound),
        "shor_gap": extra.get("shor_gap"),
        "relative_shor_gap": extra.get("relative_shor_gap"),
        "x_trace": extra.get("x_trace"),
        "y_trace": extra.get("y_trace"),
        "max_skew_violation": extra.get("max_skew_violation"),
        "fro_skew_violation": extra.get("fro_skew_violation"),
        "relative_max_skew_violation": extra.get("relative_max_skew_violation"),
        "relative_fro_skew_violation": extra.get("relative_fro_skew_violation"),
        "lazy_cut_strategy": extra.get("lazy_cut_strategy"),
        "n_cuts": extra.get("n_cuts"),
        "n_iters": extra.get("n_iters"),
        "n_rounds": extra.get("n_rounds"),
        "full_constraint_count": extra.get("full_constraint_count"),
        "oracle_solves": extra.get("oracle_solves"),
        "dual_bracket_gap": extra.get("bracket_gap"),
        "relative_dual_bracket_gap": extra.get("relative_bracket_gap"),
        "oracle_margin": extra.get("oracle_margin"),
        "gurobi_time_limit": extra.get("time_limit"),
        "gurobi_mip_gap": extra.get("mip_gap"),
        "gurobi_sol_count": extra.get("sol_count"),
        "rank1_repairs": extra.get("rank1_repairs"),
        "rank1_residual_initial_max": extra.get("rank1_residual_initial_max"),
    }


def run_one(
    panel: str,
    n: int,
    m: int,
    seed: int,
    args: argparse.Namespace,
    methods: tuple[str, ...],
    existing: dict[tuple[str, int, int, int], dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    """Run selected paper methods on one generated instance."""

    instance = generate_named_instance(
        n,
        m,
        seed,
        generator=args.generator,
        alpha=args.generator_alpha,
        sigma=args.generator_sigma,
    )

    def attach_instance_metadata(result: MethodResult) -> MethodResult:
        result.extra.setdefault("data_scale", instance.data_scale)
        return result

    key = (panel, n, m, seed)
    instance_rows = existing.get(key, {})
    screen_relative_gap = _existing_any_float(instance_rows, "screen_relative_gap")
    screen_absolute_gap = _existing_any_float(instance_rows, "screen_absolute_gap")
    if args.prefilter_relative_gap is not None and (
        screen_relative_gap is None or screen_absolute_gap is None
    ):
        shor_screen = solve_shor(instance, verbose=False)
        dual_screen = solve_dual_lop(
            instance,
            tol=args.dual_tol,
            oracle_tol=args.dual_oracle_tol,
            max_iters=args.dual_max_iters,
            verbose=False,
        )
        if dual_screen.upper_bound is not None and shor_screen.lower_bound is not None:
            screen_absolute_gap = max(0.0, dual_screen.upper_bound - shor_screen.lower_bound)
            screen_relative_gap = relative_gap(dual_screen.upper_bound, shor_screen.lower_bound)
    results: list[MethodResult] = []
    exact_value = None
    for method in CONIC_METHODS:
        exact_value = _existing_float(instance_rows, method, "upper_bound")
        if exact_value is not None:
            break

    if "Shor" in methods:
        results.append(
            attach_instance_metadata(solve_shor(instance, verbose=args.verbose_solver))
        )
    if "Full SEP" in methods:
        full = solve_full_sep(instance, verbose=args.verbose_solver)
        results.append(attach_instance_metadata(full))
        exact_value = full.upper_bound
    if "Lazy SEP" in methods:
        results.append(
            attach_instance_metadata(
                solve_lazy_sep(
                    instance,
                    tol=args.lazy_tol,
                    max_iters=args.lazy_max_iters,
                    cuts_per_iter=args.lazy_cuts_per_iter,
                    cut_strategy=args.lazy_cut_strategy,
                    verbose=args.verbose_solver,
                )
            )
        )
    if "Dual LOP/TRS" in methods:
        results.append(
            attach_instance_metadata(
                solve_dual_lop(
                    instance,
                    tol=args.dual_tol,
                    oracle_tol=args.dual_oracle_tol,
                    max_iters=args.dual_max_iters,
                    verbose=args.verbose_solver,
                )
            )
        )
    if "Gurobi" in methods:
        conic_times = [
            _existing_float(instance_rows, method, "runtime")
            for method in CONIC_METHODS
        ]
        conic_times.extend(result.runtime for result in results if result.method in CONIC_METHODS)
        conic_times = [time for time in conic_times if time is not None]
        if not conic_times:
            raise RuntimeError(f"Cannot set Gurobi time limit for n={n}, m={m}, seed={seed}")
        gurobi_limit = max(args.gurobi_floor, args.gurobi_mult * max(conic_times))
        results.append(
            attach_instance_metadata(
                solve_gurobi_qp(
                    instance,
                    time_limit=gurobi_limit,
                    mip_gap=args.gurobi_mip_gap,
                    verbose=args.verbose_solver,
                )
            )
        )

    for result in results:
        if result.method == "Shor" and exact_value is not None and result.lower_bound is not None:
            result.extra["shor_gap"] = max(0.0, exact_value - result.lower_bound)
            result.extra["relative_shor_gap"] = relative_gap(exact_value, result.lower_bound)
    return [
        result_row(
            panel,
            n,
            m,
            seed,
            result,
            exact_value=exact_value,
            args=args,
            screen_relative_gap=screen_relative_gap,
            screen_absolute_gap=screen_absolute_gap,
        )
        for result in results
    ]


def _existing_float(
    existing: dict[tuple[str, int, int, int], dict[str, dict[str, str]]] | dict[str, dict[str, str]],
    method: str,
    field: str,
) -> float | None:
    row = existing.get(method)
    if not row:
        return None
    value = row.get(field)
    if value in (None, ""):
        return None
    return float(value)


def _existing_any_float(existing: dict[str, dict[str, str]], field: str) -> float | None:
    """Return a metadata field from any existing method row for an instance."""

    for row in existing.values():
        value = row.get(field)
        if value not in (None, ""):
            return float(value)
    return None


def existing_rows(path: Path) -> dict[tuple[str, int, int, int], dict[str, dict[str, str]]]:
    """Return existing rows indexed by instance and method."""

    if not path.exists():
        return {}
    rows: dict[tuple[str, int, int, int], dict[str, dict[str, str]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["panel"], int(row["n"]), int(row["m"]), int(row["seed"]))
            rows.setdefault(key, {})[row["method"]] = row
    return rows


def check_header(path: Path) -> None:
    """Fail loudly rather than appending rows that do not match the header."""

    with path.open(newline="") as stream:
        header = next(csv.reader(stream), None)
    if header is not None and header != FIELDNAMES:
        missing = [name for name in FIELDNAMES if name not in header]
        raise SystemExit(
            f"{path} was written with a different column set and cannot be "
            f"appended to (missing: {missing or 'none'}; extra: "
            f"{[name for name in header if name not in FIELDNAMES] or 'none'}). "
            "Re-run without --resume, or write to a new --output."
        )


def parse_methods(text: str) -> tuple[str, ...]:
    """Parse comma-separated method aliases from the CLI."""

    methods = []
    for alias in text.split(","):
        alias = alias.strip().lower()
        if not alias:
            continue
        if alias not in METHOD_ALIASES:
            raise ValueError(f"unknown method alias {alias!r}; choose from {sorted(METHOD_ALIASES)}")
        methods.append(METHOD_ALIASES[alias])
    if not methods:
        raise ValueError("at least one method is required")
    return tuple(methods)


def parse_size(text: str) -> tuple[int, int]:
    """Parse a size written as n,m or n x m."""

    normalized = text.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"size must look like n,m or nxm, got {text!r}")
    return int(parts[0]), int(parts[1])


def prefiltered_panel(args: argparse.Namespace, panel: list[tuple[str, int, int, int]]) -> list[tuple[str, int, int, int]]:
    """Select the first seeds per size whose Shor relative gap passes threshold."""

    if args.prefilter_relative_gap is None:
        return panel

    selected: list[tuple[str, int, int, int]] = []
    seen_sizes: list[tuple[str, int, int]] = []
    for panel_name, n, m, _ in panel:
        key = (panel_name, n, m)
        if key not in seen_sizes:
            seen_sizes.append(key)

    for panel_name, n, m in seen_sizes:
        found = 0
        seed = 0
        while found < args.cases_per_size and seed < args.prefilter_max_seed:
            instance = generate_named_instance(
                n,
                m,
                seed,
                generator=args.generator,
                alpha=args.generator_alpha,
                sigma=args.generator_sigma,
            )
            shor = solve_shor(instance, verbose=False)
            dual = solve_dual_lop(
                instance,
                tol=args.screen_dual_tol,
                oracle_tol=args.screen_dual_oracle_tol,
                max_iters=args.screen_dual_max_iters,
                verbose=False,
            )
            if dual.upper_bound is not None and shor.lower_bound is not None:
                abs_gap = max(0.0, dual.upper_bound - shor.lower_bound)
                rel_gap = abs_gap / max(1.0, abs(dual.upper_bound))
                if rel_gap > args.prefilter_relative_gap:
                    selected.append((panel_name, n, m, seed))
                    found += 1
                    if args.verbose:
                        print(
                            f"[prefilter] n={n} m={m} seed={seed} "
                            f"rel_gap={rel_gap:.3e} abs_gap={abs_gap:.3e}",
                            flush=True,
                        )
            if args.verbose and seed > 0 and seed % args.prefilter_progress == 0:
                print(
                    f"[prefilter] n={n} m={m} scanned {seed} seeds; "
                    f"found {found}/{args.cases_per_size}",
                    flush=True,
                )
            seed += 1
        if found < args.cases_per_size:
            raise RuntimeError(
                f"Only found {found} instances for n={n}, m={m} after "
                f"{args.prefilter_max_seed} seeds."
            )
    return selected


def write_selected_panel(path: Path, args: argparse.Namespace, panel: list[tuple[str, int, int, int]]) -> None:
    """Save the selected size/seed panel so the experiment is reproducible."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "panel",
                "n",
                "m",
                "seed",
                "generator",
                "generator_alpha",
                "generator_sigma",
                "prefilter_relative_gap",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for panel_name, n, m, seed in panel:
            writer.writerow(
                {
                    "panel": panel_name,
                    "n": n,
                    "m": m,
                    "seed": seed,
                    "generator": args.generator,
                    "generator_alpha": args.generator_alpha,
                    "generator_sigma": args.generator_sigma,
                    "prefilter_relative_gap": args.prefilter_relative_gap,
                }
            )


def read_selected_panel(path: Path) -> list[tuple[str, int, int, int]]:
    """Load a previously saved selected size/seed panel."""

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        (row["panel"], int(row["n"]), int(row["m"]), int(row["seed"]))
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"raw CSV to write (default {DEFAULT_OUTPUT.name}; --quick defaults to "
        "bilinear_smoke.csv so a smoke test never overwrites the committed panel)",
    )
    parser.add_argument("--selected-panel-input", type=Path, default=None)
    parser.add_argument("--selected-panel-output", type=Path, default=DEFAULT_SELECTED_PANEL)
    parser.add_argument("--select-only", action="store_true", help="save selected seeds and exit before method solves")
    parser.add_argument("--quick", action="store_true", help="run only n=m=2, seed=0")
    parser.add_argument("--resume", action="store_true", help="append to output and skip completed instances")
    parser.add_argument("--generator", choices=("iid", "correlated-top"), default="iid")
    parser.add_argument("--generator-alpha", type=float, default=3.0)
    parser.add_argument("--generator-sigma", type=float, default=0.1)
    parser.add_argument(
        "--prefilter-relative-gap",
        type=float,
        default=None,
        help="select first cases per size with Shor relative gap above this threshold",
    )
    parser.add_argument("--cases-per-size", type=int, default=10)
    parser.add_argument("--prefilter-max-seed", type=int, default=10000)
    parser.add_argument("--prefilter-progress", type=int, default=500)
    parser.add_argument("--screen-dual-tol", type=float, default=1e-3)
    parser.add_argument("--screen-dual-oracle-tol", type=float, default=1e-7)
    parser.add_argument("--screen-dual-max-iters", type=int, default=50)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHOD_ALIASES),
        help="comma-separated methods: shor,full,lazy,dual,gurobi",
    )
    parser.add_argument(
        "--only-size",
        action="append",
        default=[],
        help="restrict to a size n,m or nxm; may be repeated",
    )
    parser.add_argument(
        "--skip-size",
        action="append",
        default=[],
        help="skip a size n,m or nxm; may be repeated",
    )
    parser.add_argument(
        "--only-seed",
        action="append",
        type=int,
        default=[],
        help="restrict to a seed value; may be repeated",
    )
    parser.add_argument("--verbose", action="store_true", help="print progress rows")
    parser.add_argument("--verbose-solver", action="store_true", help="show Mosek/Gurobi logs")
    parser.add_argument("--lazy-tol", type=float, default=1e-7)
    parser.add_argument("--lazy-max-iters", type=int, default=100)
    parser.add_argument("--lazy-cuts-per-iter", type=int, default=50)
    parser.add_argument(
        "--lazy-cut-strategy",
        choices=("coordinate", "svd"),
        default="coordinate",
        help="skew-skew separator for Lazy SEP; coordinate adds sparse "
        "individual equality cuts",
    )
    parser.add_argument("--dual-tol", type=float, default=1e-6)
    parser.add_argument("--dual-oracle-tol", type=float, default=1e-8)
    parser.add_argument("--dual-max-iters", type=int, default=80)
    parser.add_argument("--gurobi-floor", type=float, default=5.0)
    parser.add_argument("--gurobi-mult", type=float, default=3.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT.with_name("bilinear_smoke.csv") if args.quick else DEFAULT_OUTPUT
    methods = parse_methods(args.methods)

    if args.selected_panel_input is not None:
        panel = read_selected_panel(args.selected_panel_input)
    else:
        panel = [("square", 2, 2, 0)] if args.quick else paper_panel()
    only_sizes = {parse_size(size) for size in args.only_size}
    skip_sizes = {parse_size(size) for size in args.skip_size}
    only_seeds = set(args.only_seed)
    if only_sizes:
        panel = [row for row in panel if (row[1], row[2]) in only_sizes]
    if skip_sizes:
        panel = [row for row in panel if (row[1], row[2]) not in skip_sizes]
    if only_seeds:
        panel = [row for row in panel if row[3] in only_seeds]
    if args.selected_panel_input is None:
        panel = prefiltered_panel(args, panel)
    if args.prefilter_relative_gap is not None and args.selected_panel_input is None:
        write_selected_panel(args.selected_panel_output, args, panel)
        if args.verbose:
            print(f"wrote selected panel to {args.selected_panel_output}", flush=True)
    if args.select_only:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = existing_rows(args.output) if args.resume else {}
    mode = "a" if args.resume and args.output.exists() else "w"
    if mode == "a":
        check_header(args.output)
    written = 0
    skipped = 0
    with args.output.open(mode, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for index, (panel_name, n, m, seed) in enumerate(panel, 1):
            key = (panel_name, n, m, seed)
            have = set(existing.get(key, {}))
            if set(methods).issubset(have):
                skipped += 1
                if args.verbose:
                    print(f"[{index}/{len(panel)}] skip n={n} m={m} seed={seed}", flush=True)
                continue
            if args.verbose:
                print(f"[{index}/{len(panel)}] run n={n} m={m} seed={seed}", flush=True)
            needed = tuple(method for method in methods if method not in have)
            rows = run_one(panel_name, n, m, seed, args, needed, existing)
            writer.writerows(rows)
            stream.flush()
            existing.setdefault(key, {}).update({str(row["method"]): {k: str(v) for k, v in row.items()} for row in rows})
            written += len(rows)
    if args.verbose:
        print(f"wrote {written} rows to {args.output} ({skipped} instances skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
