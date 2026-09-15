"""Shor SDP relaxation for the paper bilinear experiments."""

from __future__ import annotations

import time

import numpy as np

from .instances import BilinearInstance
from .metrics import relative_residual
from .sep_utils import MethodResult, fusion_sparse


def solve_shor(instance: BilinearInstance, *, verbose: bool = False) -> MethodResult:
    """Solve the standard Shor relaxation over moments of (1,x,y).

    The moment matrix H represents [1; x; y][1; x; y]'. The two unit-ball
    constraints are relaxed to trace(X) <= 1 and trace(Y) <= 1.
    """

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    start = time.perf_counter()
    dim = 1 + instance.m + instance.n
    x_slice = slice(1, 1 + instance.m)
    y_slice = slice(1 + instance.m, dim)

    cost = np.zeros((dim, dim))
    cost[0, x_slice] = 0.5 * instance.c
    cost[x_slice, 0] = 0.5 * instance.c
    cost[0, y_slice] = 0.5 * instance.d
    cost[y_slice, 0] = 0.5 * instance.d
    cost[x_slice, y_slice] = 0.5 * instance.R.T
    cost[y_slice, x_slice] = 0.5 * instance.R

    x_trace = np.zeros((dim, dim))
    x_trace[x_slice, x_slice] = np.eye(instance.m)
    y_trace = np.zeros((dim, dim))
    y_trace[y_slice, y_slice] = np.eye(instance.n)

    with Model("paper_shor") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)
        H = model.variable("H", Domain.inPSDCone(dim))
        model.constraint("homogeneous", H.index(0, 0), Domain.equalsTo(1.0))
        model.constraint("x_ball", Expr.dot(fusion_sparse(x_trace, Matrix), H), Domain.lessThan(1.0))
        model.constraint("y_ball", Expr.dot(fusion_sparse(y_trace, Matrix), H), Domain.lessThan(1.0))
        model.objective(ObjectiveSense.Minimize, Expr.dot(fusion_sparse(cost, Matrix), H))
        model.solve()

        status = str(model.getPrimalSolutionStatus())
        if model.getPrimalSolutionStatus().name != "Optimal":
            return MethodResult("Shor", status, None, None, None, None, None, time.perf_counter() - start)
        Hstar = np.array(H.level()).reshape(dim, dim)
        lower = float(model.primalObjValue())
        dual = float(model.dualObjValue())

    x = Hstar[0, x_slice].copy()
    y = Hstar[0, y_slice].copy()
    mean_value = instance.objective(x, y)
    rank1 = float(np.linalg.norm(Hstar - np.outer(Hstar[:, 0], Hstar[:, 0])))
    rank1_rel = relative_residual(rank1, Hstar, np.outer(Hstar[:, 0], Hstar[:, 0]))
    return MethodResult(
        "Shor",
        "Optimal",
        lower,
        lower,
        mean_value,
        x,
        y,
        time.perf_counter() - start,
        {
            "pd_gap": lower - dual,
            "rank1_residual": rank1,
            "relative_rank1_residual": rank1_rel,
            "x_trace": float(np.trace(Hstar[x_slice, x_slice])),
            "y_trace": float(np.trace(Hstar[y_slice, y_slice])),
            "mean_upper_bound": mean_value,
        },
    )
