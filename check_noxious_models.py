#!/usr/bin/env python3
"""Correctness checks for the Section 4.3 noxious-facility models.

Pins: the exact planar reference is 1 for every regular polygon with the
rank-one point satisfying the sphere/cone identities; minimum-disk
normalization reaches the unit circle; and with ``--solve`` the bound chain
``Full SEP <= KRON <= RLT <= Shor`` and ``Lazy SEP == Full SEP`` on a
triangle.

Run with:  PYTHONPATH=paper-code python paper-code/check_noxious_models.py --solve
"""

from __future__ import annotations

import argparse

import numpy as np

from paper_code.noxious_instances import (
    make_instance,
    normalize_by_minimum_enclosing_disk,
    random_disk_points,
    regular_polygon,
    true_value_arrangement,
)
from paper_code.noxious_models import (
    rank_one_diagnostics,
    solve_facility_relaxation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", action="store_true", help="also run the SDP smoke test")
    args = parser.parse_args()

    for num_sites in (3, 4, 6):
        instance = make_instance(regular_polygon(num_sites), name=f"regular_{num_sites}")
        theta, x = true_value_arrangement(instance)
        if abs(theta - 1.0) > 1e-10:
            raise AssertionError(f"regular {num_sites}: expected theta=1, got {theta}")
        diagnostics = rank_one_diagnostics(instance, x, theta)
        if max(
            diagnostics["sphere_residual"],
            diagnostics["max_distance_identity_error"],
            diagnostics["maximum_hull_violation"],
            max(0.0, -diagnostics["minimum_cone_margin"]),
        ) > 1e-9:
            raise AssertionError((num_sites, diagnostics))

    normalized, _, _ = normalize_by_minimum_enclosing_disk(
        random_disk_points(8, seed=2)
    )
    if abs(float(np.max(np.linalg.norm(normalized, axis=1))) - 1.0) > 1e-9:
        raise AssertionError("minimum-disk normalization did not reach the unit circle")
    random_instance = make_instance(normalized, name="random_8_seed_2")
    theta, x = true_value_arrangement(random_instance)
    diagnostics = rank_one_diagnostics(random_instance, x, theta)
    if diagnostics["max_distance_identity_error"] > 1e-9:
        raise AssertionError(diagnostics)

    if args.solve:
        instance = make_instance(regular_polygon(3), name="regular_3")
        results = {
            method: solve_facility_relaxation(instance, method=method)
            for method in (
                "Shor",
                "RLT",
                "KRON",
                "Full SEP",
                "Lazy SEP",
            )
        }
        values = {name: result.theta_bound for name, result in results.items()}
        if any(value is None for value in values.values()):
            raise AssertionError(values)
        shor, rlt, kron, full, lazy = (values[name] for name in results)
        if not (full <= kron + 1e-6 and kron <= rlt + 1e-6 and rlt <= shor + 1e-6):
            raise AssertionError(values)
        if abs(full - lazy) > 1e-6:
            raise AssertionError(values)
        print("SDP smoke bounds:", values)

    print("unit-sphere formulation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
