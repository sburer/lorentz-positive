#!/usr/bin/env python3
"""Run the Section 4.1 max-distance panel (the first quadratic problem).

Each instance is a max-distance-between-two-ellipsoids problem written in the
manuscript's form (12) over two unit balls; see ``paper_code.quad_instances``.
The methods are the Shor relaxation (14), the KRON strengthening, Lazy KRON,
Full SEP (15), Lazy SEP, and a Gurobi global solve.

Unlike Section 3, none of the conic methods is exact for (12).  Each conic
method is therefore evaluated against the objective value of its own recovered
first-column point.  Gurobi remains a separate comparison method and is never
used to compute an SDP method's gap.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper_code.quad_gurobi import solve_quad_gurobi
from paper_code.quad_instances import GENERATORS, generate_named_instance, maxdist_panel
from paper_code.metrics import relative_gap, value_scale
from paper_code.quad_models import solve_quad, solve_shor_random_face
from paper_code.sep_utils import MethodResult


OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT = OUTPUTS / "raw" / "maxdist_raw.csv"
DEFAULT_SELECTED_PANEL = OUTPUTS / "raw" / "maxdist_selected_seeds.csv"

FIELDNAMES = [
    "panel",
    "n",
    "m",
    "seed",
    "generator",
    "condition",
    "center_scale",
    "prefiltered",
    "screen_relative_gap",
    "screen_absolute_gap",
    "method",
    "status",
    "value",
    "lower_bound",
    "upper_bound",
    "reference_upper_bound",
    "max_distance_squared",
    "bound_gap",
    "relative_bound_gap",
    "runtime",
    "rank1_residual",
    "relative_rank1_residual",
    "rank1_eig_largest",
    "rank1_eig_second",
    "rank1_eig_ratio",
    "sep_rank1_residual",
    "relative_sep_rank1_residual",
    "pd_gap",
    "relative_pd_gap",
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
DEFAULT_METHOD_ALIASES = ("shor", "kron", "full", "lazy", "gurobi")
CONIC_METHODS = ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP")


def result_row(
    panel: str,
    n: int,
    m: int,
    seed: int,
    result: MethodResult,
    *,
    reference: float | None,
    instance,
    args: argparse.Namespace,
    screen_relative_gap: float | None,
    screen_absolute_gap: float | None,
) -> dict[str, object]:
    """Convert one method result to a flat CSV row."""

    extra = result.extra
    bound_gap = (
        None
        if result.upper_bound is None or result.lower_bound is None
        else max(0.0, float(result.upper_bound - result.lower_bound))
    )
    relative_bound_gap = relative_gap(result.upper_bound, result.lower_bound)
    pd_gap = extra.get("pd_gap")
    return {
        "panel": panel,
        "n": n,
        "m": m,
        "seed": seed,
        "generator": args.generator,
        "condition": args.condition,
        "center_scale": args.center_scale,
        "prefiltered": args.prefilter,
        "screen_relative_gap": screen_relative_gap,
        "screen_absolute_gap": screen_absolute_gap,
        "method": result.method,
        "status": result.status,
        "value": result.value,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "reference_upper_bound": reference,
        # The max-distance value implied by this method's lower bound on (12).
        "max_distance_squared": (
            None
            if result.lower_bound is None
            else instance.max_distance_squared(result.lower_bound)
        ),
        "bound_gap": bound_gap,
        "relative_bound_gap": relative_bound_gap,
        "runtime": result.runtime,
        "rank1_residual": extra.get("rank1_residual"),
        "relative_rank1_residual": extra.get("relative_rank1_residual"),
        "rank1_eig_largest": extra.get("rank1_eig_largest"),
        "rank1_eig_second": extra.get("rank1_eig_second"),
        "rank1_eig_ratio": extra.get("rank1_eig_ratio"),
        "sep_rank1_residual": extra.get("sep_rank1_residual"),
        "relative_sep_rank1_residual": extra.get("relative_sep_rank1_residual"),
        "pd_gap": pd_gap,
        "relative_pd_gap": None
        if pd_gap is None or result.value is None or result.lower_bound is None
        else abs(float(pd_gap)) / value_scale(result.value, result.lower_bound),
        "x_trace": extra.get("x_trace"),
        "y_trace": extra.get("y_trace"),
        "x_ball": extra.get("x_ball"),
        "y_ball": extra.get("y_ball"),
        "primal_feasibility_violation": extra.get("primal_feasibility_violation"),
        "certification_gap": bound_gap,
        "relative_certification_gap": relative_bound_gap,
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


def gurobi_upper_bound(
    instance,
    args: argparse.Namespace,
    *,
    time_limit: float,
    emphasize_feasibility: bool = False,
) -> float | None:
    """Best feasible value Gurobi finds within ``time_limit``, or None."""

    result = solve_quad_gurobi(
        instance,
        time_limit=time_limit,
        mip_gap=args.gurobi_mip_gap,
        emphasize_feasibility=emphasize_feasibility,
        verbose=args.verbose_solver,
    )
    return result.upper_bound


def screen_instance(
    instance, args: argparse.Namespace
) -> tuple[float, float | None, float, float | None]:
    """Return (Shor lower bound, reference, screen score, rank1 residual).

    The clean screen is rank-only: return score 0 when the instance is rejected
    as effectively rank one, and score 1 when it survives both rank tests.
    """

    shor = solve_quad(instance, method="Shor")
    lower = shor.lower_bound if shor.lower_bound is not None else float("-inf")
    rank1 = shor.extra.get("rank1_residual")
    eig_ratio = shor.extra.get("rank1_eig_ratio")

    # Stage 1: reject numerically rank-one Shor solutions.  We now use the
    # spectral dominance ratio lambda_1/lambda_2 rather than the Frobenius
    # residual; very large ratios mean the moment matrix is effectively rank
    # one and the instance is not useful for the Shor-loose panel.
    if eig_ratio is not None and eig_ratio > args.screen_eig_ratio_threshold:
        return lower, None, 0.0, rank1

    # Stage 2: rank above tolerance does NOT imply a gap.  Instead of calling
    # Gurobi during screening, re-solve the Shor optimal face with a random
    # objective.  If the face contains a rank-one optimum, the random objective
    # should expose it except in degenerate cases.
    face = solve_shor_random_face(
        instance,
        lower,
        seed=instance.seed,
        face_tol=args.screen_face_tol,
        verbose=False,
    )
    face_ratio = face.extra.get("rank1_eig_ratio")
    if face_ratio is not None and face_ratio > args.screen_eig_ratio_threshold:
        return lower, None, 0.0, rank1

    return lower, None, 1.0, rank1


def run_one(
    panel: str,
    n: int,
    m: int,
    seed: int,
    args: argparse.Namespace,
    methods: tuple[str, ...],
    existing: dict,
) -> list[dict[str, object]]:
    """Run the selected methods on one generated max-distance instance."""

    if n != m:
        raise ValueError("the max-distance panel is square: n must equal m")
    instance = generate_named_instance(
        n,
        seed,
        generator=args.generator,
        condition=args.condition,
        center_scale=args.center_scale,
    )
    key = (panel, n, m, seed)
    instance_rows = existing.get(key, {})
    screen_relative_gap = _existing_any_float(instance_rows, "screen_relative_gap")
    screen_absolute_gap = _existing_any_float(instance_rows, "screen_absolute_gap")
    reference = _existing_any_float(instance_rows, "reference_upper_bound")

    results: list[MethodResult] = []
    for method in CONIC_METHODS:
        if method in methods:
            result = solve_quad(
                instance,
                method=method,
                solver_threads=args.mosek_threads,
                lazy_tol=args.lazy_tol,
                lazy_max_iters=args.lazy_max_iters,
                lazy_cuts_per_iter=args.lazy_cuts_per_iter,
                lazy_cut_strategy=args.lazy_cut_strategy,
                verbose=args.verbose_solver,
            )
            if result.status == "Optimal":
                violation = result.extra.get("primal_feasibility_violation")
                if violation is None or float(violation) > args.recovery_feas_tol:
                    raise RuntimeError(
                        f"infeasible recovered point for n={n}, seed={seed}, "
                        f"method={method}: violation={violation}"
                    )
                if result.upper_bound is None or result.lower_bound is None:
                    raise RuntimeError(
                        f"missing recovered bracket for n={n}, seed={seed}, method={method}"
                    )
                scale = value_scale(result.upper_bound, result.lower_bound)
                if result.upper_bound - result.lower_bound < -args.recovery_feas_tol * scale:
                    raise RuntimeError(
                        f"recovered upper bound below SDP bound for n={n}, seed={seed}, "
                        f"method={method}: [{result.lower_bound}, {result.upper_bound}]"
                    )
            results.append(result)

    ran_gurobi = "Gurobi" in methods
    if ran_gurobi:
        conic_times = [
            _existing_float(instance_rows, method, "runtime") for method in CONIC_METHODS
        ]
        conic_times.extend(
            result.runtime for result in results if result.method in CONIC_METHODS
        )
        conic_times = [t for t in conic_times if t is not None]
        if not conic_times:
            raise RuntimeError(
                f"Cannot set Gurobi time limit for n={n}, m={m}, seed={seed}"
            )
        gurobi_limit = max(args.gurobi_floor, args.gurobi_mult * max(conic_times))
        results.append(
            solve_quad_gurobi(
                instance,
                time_limit=gurobi_limit,
                mip_gap=args.gurobi_mip_gap,
                verbose=args.verbose_solver,
            )
        )

    # This legacy shared reference is retained only for backward-compatible CSV
    # and screening metadata.  Paper tables compute every SDP gap from that
    # method's own recovered upper bound.
    candidates = [
        result.upper_bound for result in results if result.upper_bound is not None
    ]
    if not ran_gurobi and reference is None and not args.rank_validation_only:
        bound = gurobi_upper_bound(
            instance, args, time_limit=args.reference_gurobi_time_limit
        )
        if bound is not None:
            candidates.append(bound)
    if reference is not None:
        candidates.append(reference)
    if not candidates and args.rank_validation_only:
        reference = None
    elif not candidates:
        raise RuntimeError(
            f"No feasible reference value for n={n}, m={m}, seed={seed}"
        )
    else:
        reference = min(candidates)

    if screen_relative_gap is None or screen_absolute_gap is None:
        shor_rows = [r for r in results if r.method == "Shor"]
        shor_lower = (
            shor_rows[0].lower_bound
            if shor_rows and shor_rows[0].lower_bound is not None
            else _existing_float(instance_rows, "Shor", "lower_bound")
        )
        if shor_lower is not None:
            screen_absolute_gap = max(0.0, reference - shor_lower)
            screen_relative_gap = relative_gap(reference, shor_lower)

    return [
        result_row(
            panel,
            n,
            m,
            seed,
            result,
            reference=reference if result.method == "Gurobi" else None,
            instance=instance,
            args=args,
            screen_relative_gap=screen_relative_gap,
            screen_absolute_gap=screen_absolute_gap,
        )
        for result in results
    ]


def _existing_float(existing: dict, method: str, field: str) -> float | None:
    row = existing.get(method)
    if not row:
        return None
    value = row.get(field)
    if value in (None, ""):
        return None
    return float(value)


def _existing_any_float(existing: dict, field: str) -> float | None:
    for row in existing.values():
        value = row.get(field)
        if value not in (None, ""):
            return float(value)
    return None


def existing_rows(path: Path) -> dict:
    """Return existing rows indexed by instance and method."""

    if not path.exists():
        return {}
    rows: dict = {}
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
            f"appended to (missing: {missing or 'none'}). Re-run without "
            "--resume, or write to a new --output."
        )


def parse_methods(text: str) -> tuple[str, ...]:
    methods = []
    for alias in text.split(","):
        alias = alias.strip().lower()
        if not alias:
            continue
        if alias not in METHOD_ALIASES:
            raise ValueError(
                f"unknown method alias {alias!r}; choose from {sorted(METHOD_ALIASES)}"
            )
        methods.append(METHOD_ALIASES[alias])
    if not methods:
        raise ValueError("at least one method is required")
    return tuple(methods)


def prefiltered_panel(args: argparse.Namespace, panel: list) -> list:
    """Select the first seeds per size that survive the rank-only Shor screen."""

    if not args.prefilter:
        return panel

    selected: list = []
    sizes: list[tuple[str, int, int]] = []
    for panel_name, n, m, _ in panel:
        if (panel_name, n, m) not in sizes:
            sizes.append((panel_name, n, m))

    for panel_name, n, m in sizes:
        found = 0
        seed = 0
        eig_filtered = 0
        while found < args.cases_per_size and seed < args.prefilter_max_seed:
            instance = generate_named_instance(
                n,
                seed,
                generator=args.generator,
                condition=args.condition,
                center_scale=args.center_scale,
            )
            _, _, screen_score, rank1 = screen_instance(instance, args)
            eig_filtered += screen_score == 0.0
            if screen_score > 0.0:
                selected.append((panel_name, n, m, seed))
                found += 1
                if args.verbose:
                    print(
                        f"[prefilter] n=m={n} seed={seed} survived rank screen "
                        f"rank1_resid={rank1:.2e}",
                        flush=True,
                    )
            if args.verbose and seed > 0 and seed % args.prefilter_progress == 0:
                print(
                    f"[prefilter] n=m={n} scanned {seed} seeds; "
                    f"found {found}/{args.cases_per_size} "
                    f"({eig_filtered} settled by the eigenvalue-ratio test)",
                    flush=True,
                )
            seed += 1
        if found < args.cases_per_size:
            raise RuntimeError(
                f"Only found {found} instances for n=m={n} after "
                f"{args.prefilter_max_seed} seeds."
            )
    return selected


def write_selected_panel(path: Path, args: argparse.Namespace, panel: list) -> None:
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
                "condition",
                "center_scale",
                "prefiltered",
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
                    "condition": args.condition,
                    "center_scale": args.center_scale,
                    "prefiltered": args.prefilter,
                }
            )


def read_selected_panel(path: Path) -> tuple[list, dict[str, str]]:
    """Return the saved panel and the generator/prefilter metadata beside it.

    The metadata matters for the write-up: a panel built by --select-only was
    filtered, and the fragment has to say so, but the run that consumes the
    panel does not otherwise know that.
    """

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    panel = [
        (row["panel"], int(row["n"]), int(row["m"]), int(row["seed"])) for row in rows
    ]
    metadata = dict(rows[0]) if rows else {}
    return panel, metadata


def parse_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.replace("x", ",").split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"raw CSV to write (default {DEFAULT_OUTPUT.name}; --quick defaults to "
        "maxdist_smoke.csv so a smoke test never overwrites the committed panel)",
    )
    parser.add_argument("--selected-panel-input", type=Path, default=None)
    parser.add_argument(
        "--selected-panel-output", type=Path, default=DEFAULT_SELECTED_PANEL
    )
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="run only n=m=3, seed=0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=(2, 4, 6, 8, 10),
        help="comma-separated square sizes n=m (default 2,4,6,8,10)",
    )
    parser.add_argument("--cases-per-size", type=int, default=10)
    parser.add_argument("--generator", choices=GENERATORS, default="ellipsoids")
    parser.add_argument("--condition", type=float, default=5.0)
    parser.add_argument("--center-scale", type=float, default=1.0)
    parser.add_argument(
        "--prefilter",
        action="store_true",
        help="select the first cases per size that survive the rank/face Shor screen",
    )
    parser.add_argument(
        "--screen-eig-ratio-threshold",
        type=float,
        default=1.0e4,
        help="stage-1 screen: reject a Shor solution when lambda_1/lambda_2 "
        "for the moment matrix exceeds this threshold",
    )
    parser.add_argument("--prefilter-max-seed", type=int, default=10000)
    parser.add_argument("--prefilter-progress", type=int, default=100)
    parser.add_argument(
        "--screen-face-tol",
        type=float,
        default=1e-7,
        help="stage-2 screen: relative width used when re-solving the Shor "
        "optimal face with a random objective",
    )
    parser.add_argument(
        "--reference-gurobi-time-limit",
        type=float,
        default=10.0,
        help="seconds for the dedicated reference solve, used only when Gurobi "
        "is not among --methods",
    )
    parser.add_argument(
        "--rank-validation-only",
        action="store_true",
        help="do not run the short Gurobi reference solve when Gurobi is absent; "
        "large-panel tables use each method's recovered-point gap without a "
        "Gurobi reference",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHOD_ALIASES),
        help="comma-separated methods: shor,kron,lazy-kron,full,lazy,gurobi",
    )
    parser.add_argument("--only-size", action="append", type=int, default=[])
    parser.add_argument("--skip-size", action="append", type=int, default=[])
    parser.add_argument("--only-seed", action="append", type=int, default=[])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument(
        "--mosek-threads",
        type=int,
        default=None,
        help="optional MOSEK thread limit; useful when Full KRON factorization "
        "would otherwise exceed available memory",
    )
    parser.add_argument("--lazy-tol", type=float, default=1e-7)
    parser.add_argument("--lazy-max-iters", type=int, default=100)
    parser.add_argument("--lazy-cuts-per-iter", type=int, default=50)
    parser.add_argument(
        "--lazy-cut-strategy",
        choices=("svd", "coordinate"),
        default="coordinate",
        help="skew-skew separator for Lazy SEP; coordinate adds sparse "
        "individual equality cuts",
    )
    parser.add_argument("--gurobi-floor", type=float, default=5.0)
    parser.add_argument("--gurobi-mult", type=float, default=3.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=1e-6)
    parser.add_argument(
        "--recovery-feas-tol",
        type=float,
        default=1e-7,
        help="abort before writing an instance when a conic first-column point "
        "violates either unit ball by more than this amount",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT.with_name("maxdist_smoke.csv") if args.quick else DEFAULT_OUTPUT
    methods = parse_methods(args.methods)

    if args.selected_panel_input is not None:
        panel, panel_metadata = read_selected_panel(args.selected_panel_input)
        # Carry the selection flag into this run's rows so the generated
        # fragment reports that the panel was filtered.
        saved = panel_metadata.get("prefiltered")
        legacy = panel_metadata.get("prefilter_relative_gap")
        if saved in ("True", "true", "1") or legacy not in (None, "", "None"):
            args.prefilter = True
    elif args.quick:
        panel = [("square", 3, 3, 0)]
    else:
        panel = maxdist_panel(args.sizes, args.cases_per_size)
    if args.only_size:
        panel = [row for row in panel if row[1] in set(args.only_size)]
    if args.skip_size:
        panel = [row for row in panel if row[1] not in set(args.skip_size)]
    if args.only_seed:
        panel = [row for row in panel if row[3] in set(args.only_seed)]
    if args.selected_panel_input is None:
        panel = prefiltered_panel(args, panel)
    if args.prefilter and args.selected_panel_input is None:
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
                    print(f"[{index}/{len(panel)}] skip n=m={n} seed={seed}", flush=True)
                continue
            if args.verbose:
                print(f"[{index}/{len(panel)}] run n=m={n} seed={seed}", flush=True)
            needed = tuple(method for method in methods if method not in have)
            rows = run_one(panel_name, n, m, seed, args, needed, existing)
            writer.writerows(rows)
            stream.flush()
            existing.setdefault(key, {}).update(
                {str(row["method"]): {k: str(v) for k, v in row.items()} for row in rows}
            )
            written += len(rows)
    if args.verbose:
        print(
            f"wrote {written} rows to {args.output} ({skipped} instances skipped)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
