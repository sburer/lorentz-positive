#!/usr/bin/env python3
"""Regression checks for max-distance table summaries."""

from __future__ import annotations

from make_maxdist_tables import summarize


def check_method_specific_recovery_wins() -> None:
    """Conic gaps must use that method's recovered point, not Gurobi."""

    rows = [
        {
            "panel": "square",
            "n": "2",
            "m": "2",
            "seed": "0",
            "method": "Shor",
            "lower_bound": "0.0",
            "upper_bound": "10.0",
            "primal_feasibility_violation": "0.0",
            "runtime": "1.0",
            "status": "Optimal",
        },
        {
            "panel": "square",
            "n": "2",
            "m": "2",
            "seed": "0",
            "method": "Full SEP",
            "lower_bound": "8.0",
            "upper_bound": "8.000000001",
            "primal_feasibility_violation": "0.0",
            "runtime": "2.0",
            "status": "Optimal",
        },
        {
            "panel": "square",
            "n": "2",
            "m": "2",
            "seed": "0",
            "method": "Gurobi",
            "lower_bound": "0.0",
            "reference_upper_bound": "8.0",
            "runtime": "3.0",
            "status": "TIME_LIMIT",
        },
    ]
    summary = summarize(rows, tight_tol=1e-9)[0]
    assert abs(float(summary["max_shor_gap"]) - 1.0) <= 1e-12
    assert int(summary["sep_tight"]) == 1


def main() -> int:
    check_method_specific_recovery_wins()
    print("max-distance table checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
