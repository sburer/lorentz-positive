"""Full SEP solver for the paper bilinear experiments.

This is Hildebrand II's polynomial-size formulation with all skew-skew
equalities present from the start. It optimizes over S >= 0 and recovers the
paper lift Z = [[1, x'], [y, V]] in SEP(n+1,m+1).
"""

from __future__ import annotations

import time

import numpy as np

from .instances import BilinearInstance
from .metrics import relative_residual
from .sep_utils import MethodResult, W_basis, fusion_sparse, rank1_residual, skew_kron_rows


def solve_full_sep(instance: BilinearInstance, *, verbose: bool = False) -> MethodResult:
    """Solve the exact Full SEP formulation."""

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    start = time.perf_counter()
    p = instance.n + 1
    q = instance.m + 1
    if min(instance.n, instance.m) < 2:
        raise NotImplementedError("Full SEP requires min(n,m) >= 2.")

    C = instance.sep_cost_matrix()
    Wp = W_basis(p)
    Wq = W_basis(q)
    block_size = instance.n * instance.m

    def G(i: int, j: int) -> np.ndarray:
        return np.kron(Wp[i], Wq[j])

    WC = np.zeros((block_size, block_size))
    for i in range(p):
        for j in range(q):
            if C[i, j] != 0.0:
                WC += C[i, j] * G(i, j)
    T = skew_kron_rows(instance.n, instance.m)

    with Model("paper_full_sep") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)
        S = model.variable("S", Domain.inPSDCone(block_size))
        model.constraint("trace", Expr.sum(S.diag()), Domain.equalsTo(1.0))
        if T.nnz:
            model.constraint(
                "skew",
                Expr.mul(fusion_sparse(T, Matrix), Expr.flatten(S)),
                Domain.equalsTo(0.0),
            )
        model.objective(ObjectiveSense.Minimize, Expr.dot(fusion_sparse(WC, Matrix), S))
        model.solve()

        status = str(model.getPrimalSolutionStatus())
        if model.getPrimalSolutionStatus().name != "Optimal":
            return MethodResult(
                "Full SEP", status, None, None, None, None, None, time.perf_counter() - start
            )
        Sstar = np.array(S.level()).reshape(block_size, block_size)
        value = float(model.primalObjValue())
        lower = float(model.dualObjValue())

    Z = np.array([[float(np.vdot(G(i, j), Sstar)) for j in range(q)] for i in range(p)])
    x = Z[0, 1:].copy()
    y = Z[1:, 0].copy()
    rank1 = rank1_residual(Z)
    return MethodResult(
        "Full SEP",
        status,
        value,
        lower,
        value,
        x,
        y,
        time.perf_counter() - start,
        {
            "rank1_residual": rank1,
            "relative_rank1_residual": relative_residual(
                rank1, Z[1:, 1:], np.outer(y, x)
            ),
            "pd_gap": value - lower,
            "skew_equalities": int(T.shape[0] if T.nnz else 0),
        },
    )
