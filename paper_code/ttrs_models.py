"""Conic relaxations for centered affine-diagonal TTRS instances."""

from __future__ import annotations

import time

import numpy as np

from .metrics import relative_residual
from .quad_models import (
    _kron_affine_matrix,
    _kron_coeff_matrices,
    _kron_cut_expr,
    _kron_expr_from_coeffs,
    _separate_kron_psd,
)
from .sep_utils import (
    MethodResult,
    W_basis,
    cut_batch_matrix,
    fusion_sparse,
    separate_skew_skew,
    skew_kron_rows,
    skew_pairs,
)
from .ttrs_instances import TtrsInstance


METHODS = ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP")


def _sep_atoms(n: int):
    """Return G(i,j)=W(e_i) tensor W(e_j) for the square SEP lift."""

    W = W_basis(n + 1)

    def G(i: int, j: int) -> np.ndarray:
        return np.kron(W[i], W[j])

    return G


def _objective_matrix(instance: TtrsInstance) -> np.ndarray:
    """Cost matrix C with <C,H> = x'Qx + c'x."""

    n = instance.n
    C = np.zeros((n + 1, n + 1))
    C[1:, 1:] = instance.Q
    C[1:, 0] = instance.c
    return C


def _rank_diagnostics(H: np.ndarray, x: np.ndarray) -> dict[str, float | None]:
    rank1 = float(np.linalg.norm(H[1:, 1:] - np.outer(x, x)))
    eigvals = np.linalg.eigvalsh(H)
    eig_largest = float(eigvals[-1])
    eig_second = float(eigvals[-2]) if eigvals.size >= 2 else 0.0
    denom = eig_second if eig_second > np.finfo(float).eps else abs(eig_second)
    ratio = float("inf") if denom <= np.finfo(float).eps else eig_largest / denom
    return {
        "rank1_residual": rank1,
        "relative_rank1_residual": relative_residual(
            rank1, H[1:, 1:], np.outer(x, x)
        ),
        "rank1_eig_largest": eig_largest,
        "rank1_eig_second": eig_second,
        "rank1_eig_ratio": ratio,
    }


def _kron_v_expr(H, i: int, j: int, a: np.ndarray, b: np.ndarray, Expr):
    """Expression for V_ij = y_i x_j in the KRON orientation."""

    return Expr.add(
        Expr.mul(float(a[i]), H.index(j + 1, i + 1)),
        Expr.mul(float(b[i]), H.index(j + 1, 0)),
    )


def solve_ttrs_relax(
    instance: TtrsInstance,
    *,
    method: str,
    lazy_tol: float = 1e-7,
    lazy_max_iters: int = 100,
    lazy_cuts_per_iter: int = 100,
    lazy_cut_strategy: str = "coordinate",
    verbose: bool = False,
    progress: bool = False,
    return_matrices: bool = False,
) -> MethodResult:
    """Solve Shor, KRON, Lazy KRON, Full SEP, or Lazy SEP for one TTRS instance."""

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if lazy_cuts_per_iter < 1:
        raise ValueError("lazy_cuts_per_iter must be positive")

    n = instance.n
    a = np.asarray(instance.a, dtype=float).reshape(-1)
    b = np.asarray(instance.b, dtype=float).reshape(-1)
    C = _objective_matrix(instance)
    Bx = np.zeros((n + 1, n + 1))
    By = np.zeros((n + 1, n + 1))
    for i in range(n):
        Bx[i + 1, i + 1] = 1.0
        By[i + 1, i + 1] = a[i] * a[i]
        By[i + 1, 0] = 2.0 * a[i] * b[i]

    start = time.perf_counter()
    uses_sep = method in ("Full SEP", "Lazy SEP")
    G = _sep_atoms(n) if uses_sep else None
    block_size = n * n
    active_cuts: list[tuple[np.ndarray, np.ndarray]] = []
    active_kron_cuts: list[np.ndarray] = []
    history: list[dict[str, float | int]] = []
    max_violation = fro_violation = None
    max_kron_violation = fro_kron_violation = None
    Kstar = None
    Sstar = None

    with Model(f"paper_ttrs_{method.lower().replace(' ', '_')}") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)

        H = model.variable("H", Domain.inPSDCone(n + 1))
        model.constraint("moment00", H.index(0, 0), Domain.equalsTo(1.0))
        model.constraint(
            "x_ball", Expr.dot(fusion_sparse(Bx, Matrix), H), Domain.lessThan(1.0)
        )
        y_ball_expr = Expr.add(
            Expr.dot(fusion_sparse(By, Matrix), H), float(b @ b)
        )
        model.constraint("y_ball", y_ball_expr, Domain.lessThan(1.0))
        model.objective(
            ObjectiveSense.Minimize, Expr.dot(fusion_sparse(C, Matrix), H)
        )

        uses_kron = method in ("KRON", "Lazy KRON")
        if uses_kron:
            identity, Ex, Ey, Exy = _kron_coeff_matrices(n, n)
            x_exprs = [H.index(j + 1, 0) for j in range(n)]
            y_exprs = [
                Expr.add(Expr.mul(float(a[i]), H.index(i + 1, 0)), float(b[i]))
                for i in range(n)
            ]
            v_exprs = [
                [_kron_v_expr(H, i, j, a, b, Expr) for j in range(n)]
                for i in range(n)
            ]
            if method == "KRON":
                K = _kron_expr_from_coeffs(
                    identity, Ex, Ey, Exy, x_exprs, y_exprs, v_exprs, Expr, Matrix
                )
                model.constraint("kron", K, Domain.inPSDCone())

        S = None
        if uses_sep:
            assert G is not None
            S = model.variable("S", Domain.inPSDCone(block_size))
            svec = Expr.flatten(S)
            G00 = G(0, 0)
            model.constraint(
                "sep_trace", Expr.dot(fusion_sparse(G00, Matrix), S), Domain.equalsTo(1.0)
            )
            for i in range(n):
                xi = Expr.dot(fusion_sparse(G(i + 1, 0), Matrix), S)
                model.constraint(
                    f"sep_x{i}",
                    Expr.sub(H.index(i + 1, 0), xi),
                    Domain.equalsTo(0.0),
                )
                for j in range(n):
                    model.constraint(
                        f"sep_V{i}_{j}",
                        Expr.sub(
                            Expr.dot(fusion_sparse(G(i + 1, j + 1), Matrix), S),
                            Expr.add(
                                Expr.mul(float(a[j]), H.index(i + 1, j + 1)),
                                Expr.mul(float(b[j]), H.index(i + 1, 0)),
                            ),
                        ),
                        Domain.equalsTo(0.0),
                    )
            for j in range(n):
                model.constraint(
                    f"sep_y{j}",
                    Expr.sub(
                        Expr.dot(fusion_sparse(G(0, j + 1), Matrix), S),
                        Expr.add(Expr.mul(float(a[j]), H.index(j + 1, 0)), float(b[j])),
                    ),
                    Domain.equalsTo(0.0),
                )

            if method == "Full SEP":
                skew = skew_kron_rows(n, n)
                if skew.nnz:
                    model.constraint(
                        "skew",
                        Expr.mul(fusion_sparse(skew, Matrix), svec),
                        Domain.equalsTo(0.0),
                    )

            if method == "Lazy SEP":
                for iteration in range(lazy_max_iters + 1):
                    solve_start = time.perf_counter()
                    model.solve()
                    solve_seconds = time.perf_counter() - solve_start
                    status = str(model.getPrimalSolutionStatus())
                    if model.getPrimalSolutionStatus().name != "Optimal":
                        return MethodResult(
                            method,
                            status,
                            None,
                            None,
                            None,
                            None,
                            None,
                            time.perf_counter() - start,
                            {
                                "n_cuts": len(active_cuts),
                                "n_iters": iteration,
                                "n_rounds": iteration,
                            },
                        )
                    Sstar = np.array(S.level()).reshape(block_size, block_size)
                    sep_start = time.perf_counter()
                    max_violation, fro_violation, cuts = separate_skew_skew(
                        Sstar,
                        n,
                        n,
                        ncuts=lazy_cuts_per_iter,
                        strategy=lazy_cut_strategy,
                    )
                    relative_max = relative_residual(max_violation, Sstar)
                    relative_fro = relative_residual(fro_violation, Sstar)
                    history.append(
                        {
                            "iteration": iteration,
                            "n_cuts": len(active_cuts),
                            "value": float(model.primalObjValue()),
                            "max_skew_violation": max_violation,
                            "relative_max_skew_violation": relative_max,
                            "fro_skew_violation": fro_violation,
                            "relative_fro_skew_violation": relative_fro,
                            "solve_seconds": solve_seconds,
                            "separation_seconds": time.perf_counter() - sep_start,
                        }
                    )
                    if verbose or progress:
                        print(
                            f"[Lazy SEP] iter={iteration} cuts={len(active_cuts)} "
                            f"value={float(model.primalObjValue()):.12g} "
                            f"rel_skew={relative_max:.3e} "
                            f"solve={solve_seconds:.2f}s sep={history[-1]['separation_seconds']:.2f}s",
                            flush=True,
                        )
                    if relative_max is not None and relative_max <= lazy_tol:
                        break
                    if iteration == lazy_max_iters:
                        break
                    scale = max(1.0, float(np.linalg.norm(Sstar)))
                    new_cuts = [(A, B) for sigma, A, B in cuts if sigma / scale > lazy_tol]
                    if not new_cuts:
                        break
                    model.constraint(
                        f"cuts_{iteration}",
                        Expr.mul(cut_batch_matrix(new_cuts, Matrix), svec),
                        Domain.equalsTo(0.0),
                    )
                    active_cuts.extend(new_cuts)
            else:
                model.solve()
        elif method == "Lazy KRON":
            for iteration in range(lazy_max_iters + 1):
                solve_start = time.perf_counter()
                model.solve()
                solve_seconds = time.perf_counter() - solve_start
                status = str(model.getPrimalSolutionStatus())
                if model.getPrimalSolutionStatus().name != "Optimal":
                    return MethodResult(
                        method,
                        status,
                        None,
                        None,
                        None,
                        None,
                        None,
                        time.perf_counter() - start,
                        {"n_cuts": len(active_kron_cuts), "n_iters": iteration},
                    )
                Hround = np.array(H.level()).reshape(n + 1, n + 1)
                xround = Hround[1:, 0].copy()
                yround = a * xround + b
                Vround = np.empty((n, n))
                for i in range(n):
                    for j in range(n):
                        Vround[i, j] = (
                            a[i] * Hround[j + 1, i + 1]
                            + b[i] * Hround[j + 1, 0]
                        )
                sep_start = time.perf_counter()
                Kstar = _kron_affine_matrix(
                    identity, Ex, Ey, Exy, xround, yround, Vround
                )
                max_kron_violation, fro_kron_violation, cuts = _separate_kron_psd(
                    Kstar, ncuts=lazy_cuts_per_iter
                )
                relative_max_kron = relative_residual(max_kron_violation, Kstar)
                relative_fro_kron = relative_residual(fro_kron_violation, Kstar)
                history.append(
                    {
                        "iteration": iteration,
                        "n_cuts": len(active_kron_cuts),
                        "value": float(model.primalObjValue()),
                        "max_kron_violation": max_kron_violation,
                        "fro_kron_violation": fro_kron_violation,
                        "relative_max_kron_violation": relative_max_kron,
                        "relative_fro_kron_violation": relative_fro_kron,
                        "solve_seconds": solve_seconds,
                        "separation_seconds": time.perf_counter() - sep_start,
                    }
                )
                if verbose or progress:
                    print(
                        f"[Lazy KRON] iter={iteration} cuts={len(active_kron_cuts)} "
                        f"value={float(model.primalObjValue()):.12g} "
                        f"rel_viol={relative_max_kron:.3e} "
                        f"solve={solve_seconds:.2f}s sep={history[-1]['separation_seconds']:.2f}s",
                        flush=True,
                    )
                if relative_max_kron <= lazy_tol or iteration == lazy_max_iters:
                    break
                scale = max(1.0, float(np.linalg.norm(Kstar)))
                new_cuts = [
                    cut
                    for cut in cuts
                    if float(-(cut @ Kstar @ cut)) / scale > lazy_tol
                ]
                if not new_cuts:
                    break
                for offset, cut in enumerate(new_cuts):
                    model.constraint(
                        f"kron_cut_{iteration}_{offset}",
                        _kron_cut_expr(
                            cut, identity, Ex, Ey, Exy, x_exprs, y_exprs, v_exprs, Expr
                        ),
                        Domain.greaterThan(0.0),
                    )
                active_kron_cuts.extend(new_cuts)
        else:
            model.solve()

        status = str(model.getPrimalSolutionStatus())
        if model.getPrimalSolutionStatus().name != "Optimal":
            return MethodResult(
                method, status, None, None, None, None, None, time.perf_counter() - start
            )
        Hstar = np.array(H.level()).reshape(n + 1, n + 1)
        value = float(model.primalObjValue())
        dual = float(model.dualObjValue())
        if uses_sep and S is not None:
            Sstar = np.array(S.level()).reshape(block_size, block_size)

    x = Hstar[1:, 0].copy()
    y = a * x + b
    recovered = instance.objective(x)
    x_ball = instance.x_ball_value(x)
    y_ball = instance.y_ball_value(x)
    primal_violation = max(0.0, x_ball - 1.0, y_ball - 1.0)
    extra: dict[str, float | int | list | None] = {
        "pd_gap": value - dual,
        **_rank_diagnostics(Hstar, x),
        "x_trace": float(np.trace(Hstar[1:, 1:])),
        "y_trace": float(
            np.sum((a * a) * np.diag(Hstar[1:, 1:])) + 2.0 * (a * b) @ x + b @ b
        ),
        "x_ball": x_ball,
        "y_ball": y_ball,
        "primal_feasibility_violation": primal_violation,
        "recovered_upper_bound": recovered,
        "certification_gap": recovered - dual,
    }
    if uses_sep:
        full_constraint_count = len(skew_pairs(n)) ** 2
        if method == "Full SEP":
            max_violation, fro_violation, _ = separate_skew_skew(
                Sstar, n, n, ncuts=1, strategy="svd"
            )
        extra.update(
            {
                "max_skew_violation": max_violation,
                "fro_skew_violation": fro_violation,
                "relative_max_skew_violation": relative_residual(max_violation, Sstar),
                "relative_fro_skew_violation": relative_residual(fro_violation, Sstar),
                "n_cuts": len(active_cuts),
                "n_iters": len(history),
                "n_rounds": len(history),
                "full_constraint_count": full_constraint_count,
                "lazy_cut_strategy": lazy_cut_strategy,
                "history": history,
            }
        )
    if return_matrices:
        extra["H"] = Hstar
        if uses_sep:
            extra["S"] = Sstar
    if method == "Lazy KRON":
        extra.update(
            {
                "max_kron_violation": max_kron_violation,
                "fro_kron_violation": fro_kron_violation,
                "relative_max_kron_violation": relative_residual(
                    max_kron_violation, Kstar
                ),
                "relative_fro_kron_violation": relative_residual(
                    fro_kron_violation, Kstar
                ),
                "n_cuts": len(active_kron_cuts),
                "n_iters": len(history),
                "n_rounds": len(history),
                "history": history,
            }
        )
    return MethodResult(
        method,
        "Optimal",
        value,
        value,
        recovered,
        x,
        y,
        time.perf_counter() - start,
        extra,
    )
