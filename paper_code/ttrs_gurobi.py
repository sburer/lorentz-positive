"""Gurobi global solve for the paper-code TTRS instances.

The TTRS panel uses the scaled single-variable form

    min x'Qx + c'x
    s.t. ||x|| <= 1, ||diag(a)x + b|| <= 1.

This module mirrors the bilinear and max-distance Gurobi wrappers: the caller
sets the time limit so it is comparable to the conic methods, and the returned
``MethodResult`` stores the incumbent, bound, and basic Gurobi diagnostics.
"""

from __future__ import annotations

import time

import numpy as np

from .sep_utils import MethodResult
from .ttrs_instances import TtrsInstance


def solve_ttrs_gurobi(
    instance: TtrsInstance,
    *,
    time_limit: float,
    mip_gap: float = 1e-6,
    verbose: bool = False,
) -> MethodResult:
    """Solve one TTRS instance with Gurobi NonConvex=2."""

    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    model = gp.Model("paper_ttrs_qp")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.FeasibilityTol = 1e-9
    model.Params.NumericFocus = 3

    n = instance.n
    a = np.asarray(instance.a, dtype=float).reshape(-1)
    b = np.asarray(instance.b, dtype=float).reshape(-1)

    # The two trust-region constraints imply these box bounds; adding them
    # makes the spatial branch-and-bound model materially easier for Gurobi.
    lower = -np.ones(n)
    upper = np.ones(n)
    positive = a > 0.0
    lower[positive] = np.maximum(lower[positive], (-1.0 - b[positive]) / a[positive])
    upper[positive] = np.minimum(upper[positive], (1.0 - b[positive]) / a[positive])

    x = model.addMVar(n, lb=lower, ub=upper, name="x")
    y = a * x + b
    model.addConstr(x @ x <= 1.0, name="x_ball")
    model.addConstr(y @ y <= 1.0, name="y_ball")
    model.setObjective(x @ instance.Q @ x + instance.c @ x, GRB.MINIMIZE)
    model.optimize()

    status = model.Status
    has_solution = model.SolCount > 0
    x_raw = np.array(x.X) if has_solution else None
    xval = _project_for_report(instance, x_raw) if has_solution else None
    raw_upper = float(model.ObjVal) if has_solution else None
    upper_bound = instance.objective(xval) if has_solution else None
    lower_bound = (
        _safe_float_attr(model, "ObjBound")
        if status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        else None
    )
    exact = upper_bound if status == GRB.OPTIMAL else None
    yval = None if xval is None else a * xval + b
    projection_shift = (
        None if raw_upper is None or upper_bound is None else upper_bound - raw_upper
    )

    x_ball = None if xval is None else instance.x_ball_value(xval)
    y_ball = None if xval is None else instance.y_ball_value(xval)
    feasibility = (
        None if x_ball is None or y_ball is None else max(0.0, x_ball - 1.0, y_ball - 1.0)
    )
    return MethodResult(
        "Gurobi",
        _status_name(status),
        exact,
        lower_bound,
        upper_bound,
        xval,
        yval,
        time.perf_counter() - start,
        {
            "time_limit": time_limit,
            "mip_gap": None if not has_solution else float(model.MIPGap),
            "sol_count": int(model.SolCount),
            "gurobi_runtime": float(model.Runtime),
            "raw_upper_bound": raw_upper,
            "projection_shift": projection_shift,
            "x_ball": x_ball,
            "y_ball": y_ball,
            "primal_feasibility_violation": feasibility,
        },
    )


def _project_for_report(instance: TtrsInstance, x: np.ndarray) -> np.ndarray:
    """Return a feasible reporting point for centered diagonal instances."""

    x = np.asarray(x, dtype=float).reshape(-1)
    if np.linalg.norm(instance.b) > 1e-10:
        return x
    scale = max(
        1.0,
        float(np.linalg.norm(x)),
        float(np.linalg.norm(instance.a * x)),
    )
    return x / scale


def _safe_float_attr(model, name: str) -> float | None:
    """Return a numeric Gurobi attribute when available for this status."""

    try:
        return float(getattr(model, name))
    except Exception:
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
