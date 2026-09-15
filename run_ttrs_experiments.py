#!/usr/bin/env python3
"""Run Shor, KRON, Lazy KRON, Full SEP, and Lazy SEP on TTRS instances."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from paper_code.metrics import relative_error, relative_gap, value_scale
from paper_code.ttrs_instances import (
    DEFAULT_REFERENCE_CSV,
    TtrsInstance,
    ballconstraints_panel,
)
from paper_code.ttrs_gurobi import solve_ttrs_gurobi
from paper_code.ttrs_models import METHODS as CONIC_METHODS, solve_ttrs_relax


OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT = OUTPUTS / "raw" / "ttrs_raw.csv"

FIELDNAMES = [
    "file",
    "n",
    "method",
    "status",
    "value",
    "lower_bound",
    "upper_bound",
    "reference_value",
    "bound_gap",
    "relative_bound_gap",
    "reference_error",
    "relative_reference_error",
    "runtime",
    "pd_gap",
    "relative_pd_gap",
    "rank1_residual",
    "relative_rank1_residual",
    "rank1_eig_largest",
    "rank1_eig_second",
    "rank1_eig_ratio",
    "x_trace",
    "y_trace",
    "x_ball",
    "y_ball",
    "primal_feasibility_violation",
    "certification_gap",
    "relative_certification_gap",
    "max_skew_violation",
    "fro_skew_violation",
    "relative_max_skew_violation",
    "relative_fro_skew_violation",
    "lazy_cut_strategy",
    "max_kron_violation",
    "fro_kron_violation",
    "relative_max_kron_violation",
    "relative_fro_kron_violation",
    "n_cuts",
    "n_iters",
    "n_rounds",
    "full_constraint_count",
    "gurobi_time_limit",
    "gurobi_mip_gap",
    "gurobi_sol_count",
]

METHOD_ALIASES = {
    "shor": "Shor",
    "kron": "KRON",
    "lazy-kron": "Lazy KRON",
    "lkron": "Lazy KRON",
    "full": "Full SEP",
    "lazy": "Lazy SEP",
    "gurobi": "Gurobi",
}


def parse_methods(text: str) -> tuple[str, ...]:
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
    return tuple(methods)


def parse_dimensions(text: str) -> set[int]:
    return {int(part) for part in text.split(",") if part.strip()}


def result_row(instance: TtrsInstance, result) -> dict[str, object]:
    extra = result.extra
    reference = instance.reference_value
    bound_gap = (
        None
        if reference is None or result.lower_bound is None
        else max(0.0, float(reference) - float(result.lower_bound))
    )
    reference_error = (
        None
        if reference is None or result.lower_bound is None
        else float(result.lower_bound) - float(reference)
    )
    pd_gap = extra.get("pd_gap")
    certification_gap = extra.get("certification_gap")
    return {
        "file": instance.name,
        "n": instance.n,
        "method": result.method,
        "status": result.status,
        "value": result.value,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "reference_value": reference,
        "bound_gap": bound_gap,
        "relative_bound_gap": relative_gap(reference, result.lower_bound),
        "reference_error": reference_error,
        "relative_reference_error": relative_error(reference, result.lower_bound),
        "runtime": result.runtime,
        "pd_gap": pd_gap,
        "relative_pd_gap": None
        if pd_gap is None
        else abs(float(pd_gap)) / value_scale(result.value, result.lower_bound),
        "rank1_residual": extra.get("rank1_residual"),
        "relative_rank1_residual": extra.get("relative_rank1_residual"),
        "rank1_eig_largest": extra.get("rank1_eig_largest"),
        "rank1_eig_second": extra.get("rank1_eig_second"),
        "rank1_eig_ratio": extra.get("rank1_eig_ratio"),
        "x_trace": extra.get("x_trace"),
        "y_trace": extra.get("y_trace"),
        "x_ball": extra.get("x_ball"),
        "y_ball": extra.get("y_ball"),
        "primal_feasibility_violation": extra.get("primal_feasibility_violation"),
        "certification_gap": certification_gap,
        "relative_certification_gap": relative_error(
            result.upper_bound, result.lower_bound
        ),
        "max_skew_violation": extra.get("max_skew_violation"),
        "fro_skew_violation": extra.get("fro_skew_violation"),
        "relative_max_skew_violation": extra.get("relative_max_skew_violation"),
        "relative_fro_skew_violation": extra.get("relative_fro_skew_violation"),
        "lazy_cut_strategy": extra.get("lazy_cut_strategy"),
        "max_kron_violation": extra.get("max_kron_violation"),
        "fro_kron_violation": extra.get("fro_kron_violation"),
        "relative_max_kron_violation": extra.get("relative_max_kron_violation"),
        "relative_fro_kron_violation": extra.get("relative_fro_kron_violation"),
        "n_cuts": extra.get("n_cuts"),
        "n_iters": extra.get("n_iters"),
        "n_rounds": extra.get("n_rounds"),
        "full_constraint_count": extra.get("full_constraint_count"),
        "gurobi_time_limit": extra.get("time_limit"),
        "gurobi_mip_gap": extra.get("mip_gap"),
        "gurobi_sol_count": extra.get("sol_count"),
    }


def existing_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        return {
            (row["file"], row["method"]): row
            for row in csv.DictReader(stream)
        }


def check_header(path: Path) -> None:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
    if header != FIELDNAMES:
        missing = [name for name in FIELDNAMES if name not in header]
        unknown = [name for name in header if name not in FIELDNAMES]
        raise RuntimeError(
            "existing output header does not match current schema "
            f"(missing: {missing or 'none'}, unknown: {unknown or 'none'})"
        )


def migrate_header(path: Path) -> None:
    """Upgrade older TTRS raw CSV files by adding newly introduced columns."""

    if not path.exists():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no CSV header")
        old_fields = list(reader.fieldnames)
        if old_fields == FIELDNAMES:
            return
        unknown = [name for name in old_fields if name not in FIELDNAMES]
        if unknown:
            raise RuntimeError(
                f"{path} contains unrecognized columns and cannot be migrated: {unknown}"
            )
        rows = list(reader)

    backup = path.with_name(path.name + ".before_schema_migration")
    if not backup.exists():
        backup.write_text(path.read_text())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
    check_header(path)


def _existing_float(existing: dict[tuple[str, str], dict[str, str]], file: str, method: str, field: str) -> float | None:
    row = existing.get((file, method))
    if not row:
        return None
    value = row.get(field)
    if value in (None, ""):
        return None
    return float(value)


def _gurobi_time_limit(instance: TtrsInstance, existing: dict[tuple[str, str], dict[str, str]], args: argparse.Namespace) -> float:
    """Set Gurobi's limit from the largest conic time already available."""

    conic_times = [
        _existing_float(existing, instance.name, method, "runtime")
        for method in CONIC_METHODS
    ]
    conic_times = [time for time in conic_times if time is not None]
    if not conic_times:
        raise RuntimeError(
            f"Cannot set Gurobi time limit for {instance.name}: no conic runtimes found"
        )
    return max(args.gurobi_floor, args.gurobi_mult * max(conic_times))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory of instance_*.mat files (default: the ballconstraints submodule).",
    )
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods",
        default="shor,kron,full,lazy",
        help="comma-separated methods: shor,kron,lazy-kron,full,lazy,gurobi",
    )
    parser.add_argument("--n", type=parse_dimensions, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lazy-tol", type=float, default=1e-7)
    parser.add_argument("--lazy-max-iters", type=int, default=100)
    parser.add_argument("--lazy-cuts-per-iter", type=int, default=100)
    parser.add_argument(
        "--lazy-cut-strategy",
        choices=("svd", "coordinate"),
        default="coordinate",
        help="skew-skew separator for Lazy SEP; coordinate adds sparse "
        "individual equality cuts and can be more stable when many cuts are needed",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--gurobi-floor", type=float, default=5.0)
    parser.add_argument("--gurobi-mult", type=float, default=3.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=1e-6)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="optional wall-clock budget in hours; completed rows are flushed "
        "as they finish, so a later --resume safely continues the panel",
    )
    args = parser.parse_args()
    deadline = (
        None if args.max_hours is None else time.perf_counter() + 3600.0 * args.max_hours
    )

    methods = parse_methods(args.methods)
    panel = ballconstraints_panel(
        data_dir=args.data_dir,
        reference_csv=args.reference_csv,
        dimensions=args.n,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        migrate_header(args.output)
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
        for index, instance in enumerate(panel, start=1):
            for method in methods:
                key = (instance.name, method)
                if key in existing:
                    skipped += 1
                    if args.verbose:
                        print(f"[{index}/{len(panel)}] skip {instance.name} {method}", flush=True)
                    continue
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0.0:
                    if args.verbose:
                        print(
                            f"wall-clock budget reached after writing {written} rows",
                            flush=True,
                        )
                    break
                if args.verbose:
                    print(f"[{index}/{len(panel)}] run {instance.name} {method}", flush=True)
                if method == "Gurobi":
                    time_limit = _gurobi_time_limit(instance, existing, args)
                    if remaining is not None:
                        time_limit = min(time_limit, max(1.0, remaining))
                    result = solve_ttrs_gurobi(
                        instance,
                        time_limit=time_limit,
                        mip_gap=args.gurobi_mip_gap,
                        verbose=args.verbose_solver,
                    )
                else:
                    result = solve_ttrs_relax(
                        instance,
                        method=method,
                        lazy_tol=args.lazy_tol,
                        lazy_max_iters=args.lazy_max_iters,
                        lazy_cuts_per_iter=args.lazy_cuts_per_iter,
                        lazy_cut_strategy=args.lazy_cut_strategy,
                        verbose=args.verbose_solver,
                    )
                row = result_row(instance, result)
                writer.writerow(row)
                stream.flush()
                existing[key] = {k: str(v) for k, v in row.items()}
                written += 1
            if deadline is not None and time.perf_counter() >= deadline:
                break
    if args.verbose:
        print(f"wrote {written} rows to {args.output} ({skipped} rows skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
