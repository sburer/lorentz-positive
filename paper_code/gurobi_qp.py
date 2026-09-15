"""Gurobi global solve for the paper bilinear problem."""

from __future__ import annotations

import time

import numpy as np

from .instances import BilinearInstance
from .sep_utils import MethodResult


def solve_gurobi_qp(
    instance: BilinearInstance,
    *,
    time_limit: float,
    mip_gap: float = 1e-6,
    verbose: bool = False,
) -> MethodResult:
    """Solve min c'x+d'y+y'Rx over two balls using Gurobi NonConvex=2."""

    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    model = gp.Model("paper_bilinear_qp")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    x = model.addMVar(instance.m, lb=-1.0, ub=1.0, name="x")
    y = model.addMVar(instance.n, lb=-1.0, ub=1.0, name="y")
    model.addConstr(x @ x <= 1.0, name="x_ball")
    model.addConstr(y @ y <= 1.0, name="y_ball")

    expr = instance.c @ x + instance.d @ y
    for j in range(instance.m):
        for i in range(instance.n):
            if instance.R[i, j] != 0.0:
                expr += float(instance.R[i, j]) * x[j] * y[i]
    model.setObjective(expr, GRB.MINIMIZE)
    model.optimize()

    status = model.Status
    status_name = _status_name(status)
    has_solution = model.SolCount > 0
    xval = np.array(x.X) if has_solution else None
    yval = np.array(y.X) if has_solution else None
    upper = float(model.ObjVal) if has_solution else None
    lower = (
        _safe_float_attr(model, "ObjBound")
        if status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        else None
    )
    exact = upper if status == GRB.OPTIMAL else None
    return MethodResult(
        "Gurobi",
        status_name,
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
        },
    )


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


def _safe_float_attr(model: object, name: str) -> float | None:
    """Read optional Gurobi attributes that may be unavailable after limits."""

    try:
        return float(getattr(model, name))
    except Exception:
        return None
