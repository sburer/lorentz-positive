"""Dual LOP/TRS solver in the notation of ``paper/Lorentz.tex``."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .instances import BilinearInstance
from .metrics import relative_residual, relative_stop, value_scale
from .sep_utils import MethodResult, fusion_sparse


@dataclass
class TrsResult:
    alpha: float
    feasible: bool
    max_violation: float
    relative_max_violation: float | None
    x: np.ndarray | None
    margin: float | None
    status: str
    rank1_residual: float | None
    rank1_repaired: bool = False
    rank1_residual_initial: float | None = None


def oracle_margin(instance: BilinearInstance, alpha: float, x: np.ndarray) -> float:
    """Return M_0(alpha)'x - ||Mbar(alpha) x|| = c'x - alpha - ||d + R x||."""

    return float(instance.c @ x - alpha - np.linalg.norm(instance.d + instance.R @ x))


def y_from_x(instance: BilinearInstance, x: np.ndarray) -> np.ndarray:
    """Closed-form minimizer over y after fixing x."""

    vector = instance.d + instance.R @ x
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(instance.n)
    return -vector / norm


def _sign_condition_certificate(
    instance: BilinearInstance, alpha: float
) -> np.ndarray:
    """Return xbar violating LOP when alpha > -max(||c||,||d||).

    The paper's Section 3 notes two necessary conditions on alpha: the first row
    M_0(alpha) = (-alpha ; c) must lie in L_{m+1}, giving alpha <= -||c||, and
    applying the same argument to M(alpha)' gives alpha <= -||d||.  A violation
    of the first is exhibited by xbar = -c/||c||, and a violation of the second
    by xbar = 0.  Return whichever has the more negative margin.
    """

    norm_c = float(np.linalg.norm(instance.c))
    candidates = [np.zeros(instance.m)]
    if norm_c > 0.0:
        candidates.append(-instance.c / norm_c)
    return min(candidates, key=lambda xbar: oracle_margin(instance, alpha, xbar))


def _trs_matrices(
    instance: BilinearInstance, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Qhat, C00, Ctr) defining the TRS separation SDP for alpha."""

    H = instance.R.T @ instance.R - np.outer(instance.c, instance.c)
    h = instance.R.T @ instance.d + alpha * instance.c
    const = float(instance.d @ instance.d - alpha * alpha)
    Qhat = np.empty((instance.m + 1, instance.m + 1))
    Qhat[0, 0] = const
    Qhat[0, 1:] = h
    Qhat[1:, 0] = h
    Qhat[1:, 1:] = H

    C00 = np.zeros((instance.m + 1, instance.m + 1))
    C00[0, 0] = 1.0
    Ctr = np.zeros((instance.m + 1, instance.m + 1))
    Ctr[1:, 1:] = np.eye(instance.m)
    return Qhat, C00, Ctr


def _rank1_repair(
    instance: BilinearInstance,
    alpha: float,
    Qhat: np.ndarray,
    C00: np.ndarray,
    Ctr: np.ndarray,
    optimal_value: float,
    rng: np.random.Generator,
    *,
    face_tol: float,
    verbose: bool,
) -> tuple[np.ndarray, float] | None:
    """Re-solve the TRS SDP with a random objective to obtain a rank-one point.

    This is the recovery procedure described in Section 2.1 of the paper: fix
    the objective value of the TRS SDP at its optimal value z* and re-solve with
    a random cost matrix.  Because the TRS SDP is an exact representation of the
    TRS, every extreme point of its optimal face is the rank-one lift of a TRS
    optimal solution, and a random objective selects such an extreme point with
    probability one.

    The value constraint is imposed as <Qhat,X> >= z* - delta rather than as an
    exact equality, so that the second SDP stays strictly feasible under the
    first solve's numerical tolerance.  The interior-point value z* can overshoot
    the true optimum by roughly the solver tolerance, which makes a too-small
    delta infeasible, so delta is widened geometrically until the repair solve
    succeeds.  Returns (xbar, rank-one residual) or None if every attempt fails.
    """

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    m = instance.m
    scale = max(1.0, abs(optimal_value), float(np.linalg.norm(Qhat)))

    for attempt in range(4):
        G = rng.standard_normal((m + 1, m + 1))
        G = 0.5 * (G + G.T)
        threshold = optimal_value - face_tol * (100.0**attempt) * scale

        with Model("paper_dual_lop_trs_rank1") as model:
            if verbose:
                import sys

                model.setLogHandler(sys.stdout)
            X = model.variable("X", Domain.inPSDCone(m + 1))
            model.constraint(
                "homogeneous", Expr.dot(fusion_sparse(C00, Matrix), X), Domain.equalsTo(1.0)
            )
            model.constraint("ball", Expr.dot(fusion_sparse(Ctr, Matrix), X), Domain.lessThan(1.0))
            model.constraint(
                "optimal_face",
                Expr.dot(fusion_sparse(Qhat, Matrix), X),
                Domain.greaterThan(threshold),
            )
            model.objective(ObjectiveSense.Maximize, Expr.dot(fusion_sparse(G, Matrix), X))
            model.solve()

            if model.getPrimalSolutionStatus().name != "Optimal":
                continue
            Xstar = np.array(X.level()).reshape(m + 1, m + 1)

        xbar = Xstar[1:, 0].copy()
        rank1 = float(np.linalg.norm(Xstar - np.outer(Xstar[:, 0], Xstar[:, 0])))
        return xbar, rank1

    return None


def solve_trs_oracle(
    instance: BilinearInstance,
    alpha: float,
    *,
    tol: float = 1e-8,
    rank1_tol: float = 1e-6,
    face_tol: float = 1e-9,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> TrsResult:
    """Check LOP feasibility of C-alpha E00 by the exact TRS SDP relaxation."""

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    norm_c = float(np.linalg.norm(instance.c))
    norm_d = float(np.linalg.norm(instance.d))
    if alpha > -max(norm_c, norm_d) + tol * value_scale(norm_c, norm_d):
        xbar = _sign_condition_certificate(instance, alpha)
        return TrsResult(
            alpha,
            False,
            float("inf"),
            None,
            xbar,
            oracle_margin(instance, alpha, xbar),
            "sign",
            0.0,
        )

    Qhat, C00, Ctr = _trs_matrices(instance, alpha)

    with Model("paper_dual_lop_trs") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)
        X = model.variable("X", Domain.inPSDCone(instance.m + 1))
        model.constraint("homogeneous", Expr.dot(fusion_sparse(C00, Matrix), X), Domain.equalsTo(1.0))
        model.constraint("ball", Expr.dot(fusion_sparse(Ctr, Matrix), X), Domain.lessThan(1.0))
        model.objective(ObjectiveSense.Maximize, Expr.dot(fusion_sparse(Qhat, Matrix), X))
        model.solve()

        status = str(model.getPrimalSolutionStatus())
        if model.getPrimalSolutionStatus().name != "Optimal":
            return TrsResult(alpha, False, float("nan"), None, None, None, status, None)
        Xstar = np.array(X.level()).reshape(instance.m + 1, instance.m + 1)
        max_violation = float(model.primalObjValue())

    x = Xstar[1:, 0].copy()
    rank1 = float(np.linalg.norm(Xstar - np.outer(Xstar[:, 0], Xstar[:, 0])))
    relative_rank1 = relative_residual(rank1, Xstar, np.outer(Xstar[:, 0], Xstar[:, 0]))
    relative_violation = relative_residual(max_violation, Qhat)
    feasible = relative_violation is not None and relative_violation <= tol
    margin = oracle_margin(instance, alpha, x)
    rank1_initial = rank1
    repaired = False

    # A separating hyperplane is only needed when alpha is rejected.  If the TRS
    # SDP solution is not numerically rank one then its first column need not be
    # a TRS solution, so re-solve on the optimal face with a random objective.
    if not feasible and relative_rank1 is not None and relative_rank1 > rank1_tol:
        repair = _rank1_repair(
            instance,
            alpha,
            Qhat,
            C00,
            Ctr,
            max_violation,
            np.random.default_rng() if rng is None else rng,
            face_tol=face_tol,
            verbose=verbose,
        )
        if repair is not None:
            x_repaired, rank1_repaired = repair
            margin_repaired = oracle_margin(instance, alpha, x_repaired)
            if margin_repaired < margin:
                x = x_repaired
                margin = margin_repaired
                rank1 = rank1_repaired
                repaired = True

    return TrsResult(
        alpha,
        feasible,
        max_violation,
        relative_violation,
        x,
        margin,
        status,
        rank1,
        repaired,
        rank1_initial,
    )


def solve_dual_lop(
    instance: BilinearInstance,
    *,
    tol: float = 1e-6,
    oracle_tol: float = 1e-8,
    rank1_tol: float = 1e-6,
    max_iters: int = 80,
    verbose: bool = False,
) -> MethodResult:
    """Solve the scalar conic dual by bisection and TRS separation."""

    start = time.perf_counter()
    norm_c = float(np.linalg.norm(instance.c))
    norm_d = float(np.linalg.norm(instance.d))
    op_norm = float(np.linalg.norm(instance.R, ord=2))
    # Paper Section 3: alpha_min = -||c|| - ||d|| - ||R||_2 and
    # alpha_max = -max(||c||,||d||).
    lo = -norm_c - norm_d - op_norm
    hi = -max(norm_c, norm_d)
    # Deterministic per instance, but a fresh draw at each repair, so that the
    # "with probability one" argument applies to each oracle call separately.
    rng = np.random.default_rng(instance.seed)

    last = None
    best_x = None
    best_margin = None
    oracle_solves = 0
    rank1_repairs = 0
    max_rank1_residual = 0.0

    def call_oracle(alpha: float) -> TrsResult:
        nonlocal oracle_solves, rank1_repairs, max_rank1_residual
        result = solve_trs_oracle(
            instance,
            alpha,
            tol=oracle_tol,
            rank1_tol=rank1_tol,
            rng=rng,
            verbose=verbose,
        )
        oracle_solves += 1
        if result.rank1_repaired:
            rank1_repairs += 1
        if result.rank1_residual_initial is not None:
            max_rank1_residual = max(max_rank1_residual, result.rank1_residual_initial)
        return result

    def stats() -> dict[str, object]:
        return {
            "oracle_solves": oracle_solves,
            "rank1_repairs": rank1_repairs,
            "rank1_residual_initial_max": max_rank1_residual,
        }

    for _ in range(10):
        last = call_oracle(lo)
        if last.feasible:
            break
        lo -= max(1.0, abs(lo))
    else:
        return MethodResult(
            "Dual LOP/TRS",
            "FAILED_BRACKET",
            None,
            None,
            None,
            None,
            None,
            time.perf_counter() - start,
            stats(),
        )

    last = call_oracle(hi)
    if last.feasible:
        # alpha cannot exceed -max(||c||,||d||), so the upper endpoint is optimal.
        xbar = _sign_condition_certificate(instance, hi)
        y = y_from_x(instance, xbar)
        value = instance.objective(xbar, y)
        return MethodResult(
            "Dual LOP/TRS",
            "OPTIMAL",
            value,
            hi,
            value,
            xbar,
            y,
            time.perf_counter() - start,
            {
                "alpha_lo": hi,
                "alpha_hi": hi,
                "bracket_gap": 0.0,
                "relative_bracket_gap": 0.0,
                **stats(),
            },
        )
    if last.x is not None:
        best_x = last.x
        best_margin = last.margin

    for _ in range(max_iters):
        if relative_stop(hi - lo, lo, hi, tol):
            break
        mid = 0.5 * (lo + hi)
        last = call_oracle(mid)
        if last.feasible:
            lo = mid
        else:
            hi = mid
            if last.x is not None:
                best_x = last.x
                best_margin = last.margin

    if best_x is None:
        best_x = _sign_condition_certificate(instance, lo)
        best_margin = oracle_margin(instance, lo, best_x)
    y = y_from_x(instance, best_x)
    value = instance.objective(best_x, y)
    return MethodResult(
        "Dual LOP/TRS",
        "OPTIMAL",
        value,
        lo,
        value,
        best_x,
        y,
        time.perf_counter() - start,
        {
            "alpha_lo": lo,
            "alpha_hi": hi,
            "bracket_gap": hi - lo,
            "relative_bracket_gap": (hi - lo) / value_scale(lo, hi),
            "oracle_margin": best_margin,
            "last_trs_violation": None if last is None else last.max_violation,
            "relative_last_trs_violation": (
                None if last is None else last.relative_max_violation
            ),
            "rank1_residual": None if last is None else last.rank1_residual,
            "relative_rank1_residual": None
            if last is None
            else relative_residual(last.rank1_residual, 1.0),
            **stats(),
        },
    )
