"""Gurobi global solve for the paper's problem (12) over two unit balls.

Same conventions as ``gurobi_qp`` for the Section 3 bilinear problem: the
redundant box bounds -1 <= x_j, y_i <= 1 are added alongside the two ball
constraints, and the time limit is set by the caller so that it is comparable
to the conic methods on the same instance.
"""

from __future__ import annotations

import time

import numpy as np

from .quad_instances import QuadInstance
from .sep_utils import MethodResult


def solve_quad_gurobi(
    instance: QuadInstance,
    *,
    time_limit: float,
    mip_gap: float = 1e-6,
    emphasize_feasibility: bool = False,
    verbose: bool = False,
) -> MethodResult:
    """Solve min c'x+d'y+x'Qx+y'Py+y'Rx over two unit balls with NonConvex=2.

    ``emphasize_feasibility`` trades bound quality for incumbent quality:
    ``MIPFocus=1`` tells Gurobi to prioritize finding feasible solutions over
    proving optimality, and ``Heuristics`` raises the share of effort spent on
    primal heuristics from its 0.05 default.  This is for the prefilter screen,
    where only the incumbent is used and the time budget is ~1 second; the
    lower bound Gurobi returns under these settings is not worth reporting.
    """

    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    model = gp.Model("paper_quad_qp")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    # The incumbent is used as a certified upper bound, so keep the constraint
    # violation small before it is projected away below.
    model.Params.FeasibilityTol = 1e-9
    if emphasize_feasibility:
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.8
    else:
        model.Params.NumericFocus = 3

    x = model.addMVar(instance.m, lb=-1.0, ub=1.0, name="x")
    y = model.addMVar(instance.n, lb=-1.0, ub=1.0, name="y")
    model.addConstr(x @ x <= 1.0, name="x_ball")
    model.addConstr(y @ y <= 1.0, name="y_ball")

    model.setObjective(
        instance.c @ x
        + instance.d @ y
        + x @ instance.Q @ x
        + y @ instance.P @ y
        + y @ instance.R @ x,
        GRB.MINIMIZE,
    )
    model.optimize()

    status = model.Status
    has_solution = model.SolCount > 0
    # Gurobi's incumbent is feasible only to FeasibilityTol, so ObjVal can sit
    # slightly *below* the true optimum and is not a valid upper bound.  Project
    # the returned point onto the two balls and re-evaluate, which makes the
    # reported upper bound feasible by construction and independent of solver
    # tolerances.
    xval = _project(np.array(x.X)) if has_solution else None
    yval = _project(np.array(y.X)) if has_solution else None
    raw_upper = float(model.ObjVal) if has_solution else None
    upper = instance.objective(xval, yval) if has_solution else None
    lower = (
        _safe_float_attr(model, "ObjBound")
        if status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        else None
    )
    exact = upper if status == GRB.OPTIMAL else None
    return MethodResult(
        "Gurobi",
        _status_name(status),
        exact,
        lower,
        upper,
        xval,
        yval,
        time.perf_counter() - start,
        {
            "time_limit": time_limit,
            "mip_gap": None if not has_solution else float(model.MIPGap),
            "sol_count": int(model.SolCount),
            "gurobi_runtime": float(model.Runtime),
            "emphasize_feasibility": emphasize_feasibility,
            "raw_upper_bound": raw_upper,
            "projection_shift": (
                None if raw_upper is None else float(upper - raw_upper)
            ),
        },
    )


def _project(v: np.ndarray) -> np.ndarray:
    """Scale v onto the unit ball if the solver left it marginally outside."""

    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1.0 else v


def _safe_float_attr(model, name: str) -> float | None:
    """Return a numeric Gurobi attribute when it is available for this status."""

    try:
        return float(getattr(model, name))
    except (AttributeError, RuntimeError):
        return None


def _status_name(status: int) -> str:
    try:
        from gurobipy import GRB

        return {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(status, str(status))
    except Exception:
        return str(status)
