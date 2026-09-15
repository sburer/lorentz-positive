"""Lazy SEP solver for the paper bilinear experiments."""

from __future__ import annotations

import time

import numpy as np

from .instances import BilinearInstance
from .metrics import relative_residual
from .sep_utils import (
    MethodResult,
    W_basis,
    cut_batch_matrix,
    fusion_sparse,
    rank1_residual,
    separate_skew_skew,
    skew_pairs,
)


def solve_lazy_sep(
    instance: BilinearInstance,
    *,
    tol: float = 1e-7,
    max_iters: int = 100,
    cuts_per_iter: int = 50,
    cut_strategy: str = "coordinate",
    verbose: bool = False,
) -> MethodResult:
    """Solve Full SEP by adding violated skew-skew equalities as cuts."""

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    start = time.perf_counter()
    p = instance.n + 1
    q = instance.m + 1
    if min(instance.n, instance.m) < 2:
        raise NotImplementedError("Lazy SEP requires min(n,m) >= 2.")

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

    active_cuts: list[tuple[np.ndarray, np.ndarray]] = []
    history: list[dict[str, float | int]] = []
    Sstar = None
    value = lower = None
    max_violation = fro_violation = None

    with Model("paper_lazy_sep") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)
        S = model.variable("S", Domain.inPSDCone(block_size))
        svec = Expr.flatten(S)
        model.constraint("trace", Expr.sum(S.diag()), Domain.equalsTo(1.0))
        model.objective(ObjectiveSense.Minimize, Expr.dot(fusion_sparse(WC, Matrix), S))

        for iteration in range(max_iters + 1):
            solve_start = time.perf_counter()
            model.solve()
            solve_seconds = time.perf_counter() - solve_start
            status = str(model.getPrimalSolutionStatus())
            if model.getPrimalSolutionStatus().name != "Optimal":
                return MethodResult(
                    "Lazy SEP",
                    status,
                    None,
                    None,
                    None,
                    None,
                    None,
                    time.perf_counter() - start,
                    {"n_cuts": len(active_cuts), "n_iters": iteration},
                )

            Sstar = np.array(S.level()).reshape(block_size, block_size)
            value = float(model.primalObjValue())
            lower = float(model.dualObjValue())
            sep_start = time.perf_counter()
            max_violation, fro_violation, cuts = separate_skew_skew(
                Sstar,
                instance.n,
                instance.m,
                ncuts=cuts_per_iter,
                strategy=cut_strategy,
            )
            relative_max_violation = relative_residual(max_violation, Sstar)
            relative_fro_violation = relative_residual(fro_violation, Sstar)
            sep_seconds = time.perf_counter() - sep_start
            row = {
                "iteration": iteration,
                "n_cuts": len(active_cuts),
                "value": value,
                "lower_bound": lower,
                "max_skew_violation": max_violation,
                "fro_skew_violation": fro_violation,
                "relative_max_skew_violation": relative_max_violation,
                "relative_fro_skew_violation": relative_fro_violation,
                "solve_seconds": solve_seconds,
                "separation_seconds": sep_seconds,
            }
            history.append(row)
            if relative_max_violation <= tol or iteration == max_iters:
                break
            scale = max(1.0, float(np.linalg.norm(Sstar)))
            new_cuts = [(A, B) for sigma, A, B in cuts if sigma / scale > tol]
            if not new_cuts:
                break
            model.constraint(
                f"cuts_{iteration}",
                Expr.mul(cut_batch_matrix(new_cuts, Matrix), svec),
                Domain.equalsTo(0.0),
            )
            active_cuts.extend(new_cuts)

    assert Sstar is not None and value is not None and lower is not None
    Z = np.array([[float(np.vdot(G(i, j), Sstar)) for j in range(q)] for i in range(p)])
    x = Z[0, 1:].copy()
    y = Z[1:, 0].copy()
    full_count = len(skew_pairs(instance.n)) * len(skew_pairs(instance.m))
    rank1 = rank1_residual(Z)
    return MethodResult(
        "Lazy SEP",
        "Optimal",
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
            "max_skew_violation": max_violation,
            "fro_skew_violation": fro_violation,
            "relative_max_skew_violation": relative_residual(max_violation, Sstar),
            "relative_fro_skew_violation": relative_residual(fro_violation, Sstar),
            "n_cuts": len(active_cuts),
            "n_iters": len(history),
            "n_rounds": len(history),
            "full_constraint_count": full_count,
            "lazy_cut_strategy": cut_strategy,
        },
    )
