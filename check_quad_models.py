#!/usr/bin/env python3
"""Correctness checks for the Section 4.1 max-distance models.

Three invariants are pinned here, all cheap:

1. Bilinear specialization.  With Q = 0 and P = 0, problem (12) is exactly the
   bilinear problem (7), and the Section 4.1 Full SEP model (15) must reproduce
   the exact optimal value produced by the Section 3 Full SEP model (11).
2. Relaxation ordering.  Shor <= KRON <= SEP <= the best feasible value found
   by Gurobi, the KRON <= SEP half being Lemma 7.
3. Lazy KRON agrees with KRON, and Lazy SEP agrees with Full SEP.

Run before any production sweep:

    PYTHONPATH=paper-code ~/miniconda3/bin/python paper-code/check_quad_models.py
"""

from __future__ import annotations

import argparse

import numpy as np

from paper_code.full_sep import solve_full_sep
from paper_code.instances import generate_instance
from paper_code.quad_gurobi import solve_quad_gurobi
from paper_code.quad_instances import QuadInstance, generate_named_instance
from paper_code.quad_models import solve_quad


def check_bilinear_specialization(sizes, tol: float) -> int:
    """Full SEP for (15) with Q=P=0 must match Full SEP for (11)."""

    failures = 0
    for n, m, seed in sizes:
        bilinear = generate_instance(n, m, seed)
        exact = solve_full_sep(bilinear)
        quad = QuadInstance(
            n=n,
            m=m,
            seed=seed,
            c=bilinear.c,
            d=bilinear.d,
            Q=np.zeros((m, m)),
            P=np.zeros((n, n)),
            R=bilinear.R,
            generator="bilinear-specialization",
        )
        got = solve_quad(quad, method="Full SEP")
        diff = abs(got.value - exact.value)
        flag = "ok" if diff <= tol else "FAIL"
        failures += diff > tol
        print(
            f"[bilinear] n={n} m={m} seed={seed} "
            f"sec3={exact.value:.10g} sec4={got.value:.10g} diff={diff:.2e} {flag}"
        )
    return failures


def check_ordering(sizes, generator: str, tol: float, time_limit: float) -> int:
    """Shor <= KRON = Lazy KRON <= Full SEP = Lazy SEP <= best feasible value."""

    failures = 0
    for k, seed in sizes:
        instance = generate_named_instance(k, seed, generator=generator)
        results = {
            method: solve_quad(instance, method=method)
            for method in ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP")
        }
        # Every relaxation's first-column point is feasible; Gurobi's incumbent
        # is what makes the bound tight enough for this test to have teeth.
        candidates = [
            result.upper_bound
            for result in results.values()
            if result.upper_bound is not None
        ]
        grb = solve_quad_gurobi(instance, time_limit=time_limit)
        if grb.upper_bound is not None:
            candidates.append(grb.upper_bound)
        upper = min(candidates)

        chain = [
            ("Shor", results["Shor"].value),
            ("KRON", results["KRON"].value),
            ("Full SEP", results["Full SEP"].value),
        ]
        for (left_name, left), (right_name, right) in zip(chain, chain[1:]):
            if left > right + tol:
                failures += 1
                print(f"[order] n=m={k} seed={seed} FAIL {left_name} > {right_name}")
        if results["Full SEP"].value > upper + tol:
            failures += 1
            print(f"[order] n=m={k} seed={seed} FAIL SEP bound exceeds feasible value")
        lazy_kron_diff = abs(results["Lazy KRON"].value - results["KRON"].value)
        if lazy_kron_diff > 1e-6:
            failures += 1
            print(f"[order] n=m={k} seed={seed} FAIL Lazy KRON != KRON")
        lazy_diff = abs(results["Lazy SEP"].value - results["Full SEP"].value)
        if lazy_diff > 1e-6:
            failures += 1
            print(f"[order] n=m={k} seed={seed} FAIL Lazy SEP != Full SEP")
        print(
            f"[order] n=m={k} seed={seed} "
            f"shor={results['Shor'].value:.10g} "
            f"kron={results['KRON'].value:.10g} "
            f"lazy_kron_diff={lazy_kron_diff:.2e} "
            f"sep={results['Full SEP'].value:.10g} "
            f"lazy_diff={lazy_diff:.2e} ub={upper:.10g} "
            f"maxdist2={instance.max_distance_squared(results['Full SEP'].value):.10g}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--generator", default="ellipsoids")
    parser.add_argument(
        "--gurobi-time-limit",
        type=float,
        default=10.0,
        help="seconds for the Gurobi solve supplying the feasible upper bound",
    )
    args = parser.parse_args()

    failures = check_bilinear_specialization(
        [(2, 2, 0), (3, 3, 1), (3, 5, 2), (4, 3, 3)], args.tol
    )
    failures += check_ordering(
        [(2, 0), (3, 0), (3, 1), (4, 0), (5, 0)],
        args.generator,
        args.tol,
        args.gurobi_time_limit,
    )
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
