#!/usr/bin/env python3
"""Run the Section 4.3 noxious-facility relaxations and write a raw CSV.

Every row records one (instance, method) pair: the relaxation's distance
upper bound ``theta_bound``, the exact planar reference value from the
hull/Voronoi arrangement, their difference, timing, and model-size and
Lazy SEP diagnostics.  The method names in the CSV are ``Shor``,
``RLT``, ``KRON``, ``Full SEP`` and ``Lazy SEP``.

The paper panels are listed in paper-code/README.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper_code.noxious_instances import (
    make_instance,
    normalize_by_minimum_enclosing_disk,
    random_disk_points,
    regular_polygon,
    targeted_seed_410_points,
    true_value_arrangement,
    true_value_grid,
)
from paper_code.noxious_models import (
    METHODS,
    solve_facility_gurobi,
    solve_facility_relaxation,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "raw" / "noxious_smoke.csv"

METHOD_SUFFIXES = {
    "shor": "Shor",
    "rlt": "RLT",
    "kron": "KRON",
    "full": "Full SEP",
    "sep": "Full SEP",
    "lazy": "Lazy SEP",
}

FIELDNAMES = [
    "instance",
    "formulation",
    "theta_semantics",
    "generator",
    "normalization",
    "normalization_center_x1",
    "normalization_center_x2",
    "normalization_radius",
    "num_sites",
    "seed",
    "method",
    "status",
    "theta_bound",
    "distance_bound",
    "squared_distance_bound",
    "theta_reference",
    "reference_method",
    "theta_feasible_grid",
    "theta_gurobi",
    "gurobi_status",
    "gurobi_gap",
    "gurobi_runtime",
    "grid_x1",
    "grid_x2",
    "runtime",
    "relaxation_gap",
    "rel_gap",
    "eval_ratio",
    "soc_pair_blocks",
    "sep_pair_blocks",
    "num_distance_blocks",
    "num_hull_facets",
    "num_cone_blocks",
    "num_nonnegative_rows",
    "radius_squared",
    "lazy_cut_strategy",
    "n_cuts",
    "n_rounds",
    "max_skew_violation",
]


def parse_ints(text: str) -> tuple[int, ...]:
    """Parse comma-separated integers."""

    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("at least one value is required")
    return values


def parse_methods(text: str) -> tuple[str, ...]:
    """Parse method aliases (``all``, ``shor``, ``rlt``, ``kron``, ``full``, ``lazy``)."""

    methods: list[str] = []
    for part in text.split(","):
        alias = part.strip().lower()
        if not alias:
            continue
        if alias == "all":
            methods.extend(METHODS)
            continue
        if alias not in METHOD_SUFFIXES:
            raise ValueError(f"unknown method alias {alias!r}")
        methods.append(METHOD_SUFFIXES[alias])
    if not methods:
        raise ValueError("at least one method is required")
    return tuple(dict.fromkeys(methods))


def build_instances(args: argparse.Namespace):
    """Yield deterministic test instances requested on the command line."""

    sizes = parse_ints(args.sizes)
    if args.generator == "regular":
        for num_sites in sizes:
            name = f"regular_{num_sites}"
            yield name, "regular", num_sites, "", "none", (0.0, 0.0), 1.0, make_instance(
                regular_polygon(num_sites), name=name
            )
    elif args.generator == "random-disk":
        seeds = parse_ints(args.seeds)
        for num_sites in sizes:
            for seed in seeds:
                points = random_disk_points(num_sites, seed=seed)
                if args.normalization == "minimum-disk":
                    points, center, radius = normalize_by_minimum_enclosing_disk(points)
                    name = f"random_disk_normalized_{num_sites}_seed_{seed}"
                else:
                    center, radius = (0.0, 0.0), 1.0
                    name = f"random_disk_{num_sites}_seed_{seed}"
                yield name, "random-disk", num_sites, seed, args.normalization, center, radius, make_instance(
                    points, name=name
                )
    elif args.generator == "targeted-410":
        name = "targeted_m4_seed_410"
        yield name, "targeted-410", 4, 410, "none", (0.0, 0.0), 1.0, make_instance(
            targeted_seed_410_points(), name=name
        )
    else:
        raise ValueError(f"unknown generator {args.generator!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--generator",
        choices=("regular", "random-disk", "targeted-410"),
        default="regular",
    )
    parser.add_argument("--sizes", default="3,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--normalization",
        choices=("minimum-disk", "none"),
        default="minimum-disk",
        help="normalization used for random-disk instances",
    )
    parser.add_argument("--methods", default="all")
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument(
        "--with-gurobi-reference",
        action="store_true",
        help="cross-check the planar reference with the original nonconvex QCP",
    )
    parser.add_argument("--gurobi-time-limit", type=float, default=30.0)
    parser.add_argument(
        "--mosek-time-limit",
        type=float,
        default=None,
        help="optional per-solve Mosek time limit in seconds",
    )
    parser.add_argument("--lazy-tol", type=float, default=1e-7)
    parser.add_argument("--lazy-max-iters", type=int, default=100)
    parser.add_argument("--lazy-cuts-per-iter", type=int, default=50)
    parser.add_argument(
        "--lazy-cut-strategy", choices=("coordinate", "svd"), default="coordinate"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    methods = parse_methods(args.methods)

    rows = []
    for name, generator, num_sites, seed, normalization, center, radius, instance in build_instances(args):
        grid_theta_squared, _ = true_value_grid(instance, grid=args.grid)
        exact_distance, exact_x = true_value_arrangement(instance)
        if args.with_gurobi_reference:
            gurobi = solve_facility_gurobi(instance, time_limit=args.gurobi_time_limit)
        else:
            gurobi = None

        for method in methods:
            result = solve_facility_relaxation(
                instance,
                method=method,
                mosek_time_limit=args.mosek_time_limit,
                lazy_tol=args.lazy_tol,
                lazy_max_iters=args.lazy_max_iters,
                lazy_cuts_per_iter=args.lazy_cuts_per_iter,
                lazy_cut_strategy=args.lazy_cut_strategy,
                verbose=args.verbose,
            )
            reference = exact_distance
            gurobi_theta = (
                None
                if gurobi is None or gurobi.theta is None
                else max(0.0, gurobi.theta) ** 0.5
            )

            row = {
                "instance": name,
                "formulation": "unit-sphere",
                "theta_semantics": "distance",
                "generator": generator,
                "normalization": normalization,
                "normalization_center_x1": center[0],
                "normalization_center_x2": center[1],
                "normalization_radius": radius,
                "num_sites": num_sites,
                "seed": seed,
                "method": method,
                "status": result.status,
                "theta_bound": result.theta_bound,
                "distance_bound": result.distance_bound,
                "squared_distance_bound": result.extra.get("squared_distance_bound", ""),
                "theta_reference": reference,
                "reference_method": "planar-arrangement",
                "theta_feasible_grid": grid_theta_squared,
                "theta_gurobi": "" if gurobi_theta is None else gurobi_theta,
                "gurobi_status": "" if gurobi is None else gurobi.status,
                "gurobi_gap": "" if gurobi is None else gurobi.mip_gap,
                "gurobi_runtime": "" if gurobi is None else gurobi.runtime,
                "grid_x1": exact_x[0],
                "grid_x2": exact_x[1],
                "runtime": result.runtime,
                "relaxation_gap": (
                    ""
                    if result.theta_bound is None
                    else (result.theta_bound - reference) / max(1.0, abs(reference))
                ),
                **result.extra,
            }
            rows.append(row)
            if args.verbose:
                theta = "--" if result.theta_bound is None else f"{result.theta_bound:.12g}"
                print(
                    f"{name:24s} {method:14s} theta_ub={theta} "
                    f"exact={reference:.12g} pairs={row['soc_pair_blocks']} "
                    f"sep={row['sep_pair_blocks']} time={result.runtime:.3g}s "
                    f"status={result.status}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    print(f"rows {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
