#!/usr/bin/env python3
"""Regression checks for large-panel tightness and diagnostics."""

from make_large_instance_tables import is_tight, recovered_feasibility
from paper_code.metrics import PAPER_TIGHT_REL_TOL


def main() -> int:
    tight_with_poor_diagnostics = {
        "relative_certification_gap": "1e-6",
        "relative_rank1_residual": "0.25",
        "primal_feasibility_violation": "0.1",
    }
    not_tight_with_good_diagnostics = {
        "relative_certification_gap": "1e-4",
        "relative_rank1_residual": "1e-12",
        "primal_feasibility_violation": "0",
    }

    assert is_tight(tight_with_poor_diagnostics, tight_tol=PAPER_TIGHT_REL_TOL)
    assert recovered_feasibility(tight_with_poor_diagnostics) == 0.1
    assert not is_tight(
        not_tight_with_good_diagnostics, tight_tol=PAPER_TIGHT_REL_TOL
    )
    print("large-panel tightness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
