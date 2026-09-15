#!/usr/bin/env python3
"""Self-checks for the Dual LOP/TRS oracle.

The Section 3.1 panel never triggers the rank-one repair in the TRS separation
oracle: on random data the TRS SDP solution is rank one to about 1e-7, so the
first column of the SDP solution is already a valid separating point.  The cases
below are constructed to be degenerate, so that the TRS has a continuum of
optimal solutions and the first column is *not* a TRS solution.  Without the
repair described in Section 2.1 of the manuscript these return badly wrong
primal values (for the diagonal case, 0.0 instead of -2.0).

Run with:  PYTHONPATH=paper-code python paper-code/check_dual_lop.py
"""

from __future__ import annotations

import numpy as np

from paper_code.dual_lop import solve_dual_lop
from paper_code.instances import BilinearInstance


def degenerate_cases() -> list[tuple[str, BilinearInstance, float]]:
    """Instances with c=d=0, where the optimal value is -||R||_2.

    With no linear terms the TRS objective does not involve the first column of
    the moment matrix at all, so the SDP has optimal solutions whose first
    column is zero.  A repeated leading singular value makes the optimal face
    higher dimensional still.
    """

    cases: list[tuple[str, BilinearInstance, float]] = []
    for k in (3, 5):
        cases.append(
            (
                f"R = I_{k}",
                BilinearInstance(n=k, m=k, seed=0, c=np.zeros(k), d=np.zeros(k), R=np.eye(k)),
                -1.0,
            )
        )
    cases.append(
        (
            "R = diag(2,2,1,0.5), repeated top singular value",
            BilinearInstance(
                n=4, m=4, seed=1, c=np.zeros(4), d=np.zeros(4), R=np.diag([2.0, 2.0, 1.0, 0.5])
            ),
            -2.0,
        )
    )
    cases.append(
        (
            "rectangular, rank-deficient R",
            BilinearInstance(
                n=3,
                m=5,
                seed=2,
                c=np.zeros(5),
                d=np.zeros(3),
                R=np.hstack([1.5 * np.eye(3), np.zeros((3, 2))]),
            ),
            -1.5,
        )
    )
    return cases


def main() -> int:
    failures = 0
    for label, instance, expected in degenerate_cases():
        result = solve_dual_lop(instance)
        value = result.value
        bound = result.lower_bound
        repairs = result.extra["rank1_repairs"]
        ok = (
            value is not None
            and bound is not None
            and abs(value - expected) <= 1e-5
            and bound <= expected + 1e-5
            and repairs > 0
        )
        failures += not ok
        print(
            f"[{'ok ' if ok else 'FAIL'}] {label}: value={value:.9f} "
            f"bound={bound:.9f} expected={expected} repairs={repairs}/"
            f"{result.extra['oracle_solves']}"
        )

    print(f"\n{len(degenerate_cases()) - failures}/{len(degenerate_cases())} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
