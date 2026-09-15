"""Conic relaxations of the paper's problem (12) over two unit balls.

All four relaxations share the Shor moment matrix of the manuscript's (14),

    min  <C, U>
    s.t. tr X <= 1, tr Y <= 1, U >= 0,          U = [[1, x', y'], [x, X, V'], [y, V, Y]],

and differ only in how the cross block V is constrained:

- ``Shor``     : no extra constraint (the baseline of (14));
- ``KRON``     : the Kronecker constraint K(Z) >= 0 built from Ihat_x and Ihat_y;
- ``Lazy KRON``: Shor plus separated scalar inequalities b' K(Z) b >= 0;
- ``Full SEP`` : Z = [[1, x'], [y, V]] in SEP(n+1, m+1) via Proposition 2,
                 Z = W*(T) with T >= 0 and all skew-skew equalities <T, X> = 0
                 for X in A(n) tensor A(m) -- this is the manuscript's (15);
- ``Lazy SEP`` : the same model with the skew-skew equalities added only as they
                 are found violated, exactly as in ``lazy_sep`` for Section 3.

Unlike the pure bilinear problem of Section 3, none of these is exact for (12),
so every method returns a *lower* bound on the optimal value of (12).  Feasible
upper bounds come from the (always feasible) first-column point of each
relaxation and, more usefully, from a time-limited Gurobi solve.

The SEP/skew machinery is reused verbatim from ``sep_utils``; only the outer
Shor scaffold and the coupling constraints are new here.
"""

from __future__ import annotations

import time

import numpy as np

from .metrics import relative_residual
from .quad_instances import QuadInstance
from .sep_utils import (
    MethodResult,
    W_basis,
    cut_batch_matrix,
    fusion_sparse,
    rank1_residual,
    separate_skew_skew,
    skew_kron_rows,
    skew_pairs,
)


METHODS = ("Shor", "KRON", "Lazy KRON", "Full SEP", "Lazy SEP")


def _rank_diagnostics(Ustar: np.ndarray, z: np.ndarray) -> dict[str, float | None]:
    """Return Frobenius and spectral rank-one diagnostics for a Shor matrix."""

    rank1 = float(np.linalg.norm(Ustar[1:, 1:] - np.outer(z, z)))
    eigvals = np.linalg.eigvalsh(Ustar)
    eig_largest = float(eigvals[-1])
    eig_second = float(eigvals[-2]) if eigvals.size >= 2 else 0.0
    eig_denom = eig_second if eig_second > np.finfo(float).eps else abs(eig_second)
    eig_ratio = float("inf") if eig_denom <= np.finfo(float).eps else eig_largest / eig_denom
    return {
        "rank1_residual": rank1,
        "relative_rank1_residual": relative_residual(
            rank1,
            Ustar[1:, 1:],
            np.outer(z, z),
        ),
        "rank1_eig_largest": eig_largest,
        "rank1_eig_second": eig_second,
        "rank1_eig_ratio": eig_ratio,
    }


def _sep_atoms(n: int, m: int):
    """Return G(i,j) = W_{n+1}(e_i) tensor W_{m+1}(e_j) for the SEP lift.

    ``Z[i, j] = <G(i, j), T>`` recovers the manuscript's Z = [[1, x'], [y, V]],
    where T is the (nm) x (nm) PSD matrix of Proposition 2.
    """

    Wp = W_basis(n + 1)
    Wq = W_basis(m + 1)

    def G(i: int, j: int) -> np.ndarray:
        return np.kron(Wp[i], Wq[j])

    return G


def _kron_coeff_matrices(n: int, m: int):
    """Constant matrices for the KRON constraint K(Z) >= 0 of Section 4.1.

    With Ihat_x = [[I_m, x], [x', 1]] in S^{m+1} and
    Ihat_y = [[I_n, y], [y', 1]] in S^{n+1}, the Kronecker product expands as

        Ihat_x tensor Ihat_y = I + sum_j x_j (E_j tensor I)
                                 + sum_i y_i (I tensor F_i)
                                 + sum_{i,j} x_j y_i (E_j tensor F_i),

    and replacing each product x_j y_i by V_ij gives the affine matrix K(Z).
    Ported from ``code/kron_mosek._kron_coeff_matrices`` with the paper's
    dimension convention (x in R^m, y in R^n).
    """

    p, q = m + 1, n + 1
    E = []  # E_j in S^{m+1}: ones at (j, m) and (m, j)
    for j in range(m):
        Ej = np.zeros((p, p))
        Ej[j, m] = 1.0
        Ej[m, j] = 1.0
        E.append(Ej)
    F = []  # F_i in S^{n+1}: ones at (i, n) and (n, i)
    for i in range(n):
        Fi = np.zeros((q, q))
        Fi[i, n] = 1.0
        Fi[n, i] = 1.0
        F.append(Fi)

    identity = np.eye(p * q)
    Ex = [np.kron(E[j], np.eye(q)) for j in range(m)]
    Ey = [np.kron(np.eye(p), F[i]) for i in range(n)]
    Exy = [[np.kron(E[j], F[i]) for j in range(m)] for i in range(n)]
    return identity, Ex, Ey, Exy


def _kron_affine_matrix(
    identity: np.ndarray,
    Ex: list[np.ndarray],
    Ey: list[np.ndarray],
    Exy: list[list[np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Evaluate the affine KRON matrix K(Z)."""

    K = identity.copy()
    for j, xj in enumerate(x):
        K += float(xj) * Ex[j]
    for i, yi in enumerate(y):
        K += float(yi) * Ey[i]
    for i in range(len(y)):
        for j in range(len(x)):
            K += float(V[i, j]) * Exy[i][j]
    return 0.5 * (K + K.T)


def _separate_kron_psd(
    K: np.ndarray, *, ncuts: int
) -> tuple[float, float, list[np.ndarray]]:
    """Separate K >= 0 by negative eigenvectors, returning b'K b cuts."""

    eigvals, eigvecs = np.linalg.eigh(K)
    negative = np.where(eigvals < 0.0)[0]
    if negative.size == 0:
        return 0.0, 0.0, []
    order = negative[np.argsort(eigvals[negative])]
    selected = order[:ncuts]
    cuts = [eigvecs[:, k].copy() for k in selected]
    return float(-eigvals[order[0]]), float(np.linalg.norm(eigvals[negative])), cuts


def _kron_expr_from_coeffs(
    identity, Ex, Ey, Exy, x_exprs, y_exprs, v_exprs, Expr, Matrix
):
    """Build the Fusion expression for the full affine KRON matrix."""

    terms = [Expr.constTerm(fusion_sparse(identity, Matrix))]
    for j, xj in enumerate(x_exprs):
        terms.append(Expr.mul(xj, fusion_sparse(Ex[j], Matrix)))
    for i, yi in enumerate(y_exprs):
        terms.append(Expr.mul(yi, fusion_sparse(Ey[i], Matrix)))
    for i, row in enumerate(v_exprs):
        for j, vij in enumerate(row):
            terms.append(Expr.mul(vij, fusion_sparse(Exy[i][j], Matrix)))
    K = Expr.add(terms)
    return Expr.mul(0.5, Expr.add(K, Expr.transpose(K)))


def _kron_cut_expr(b, identity, Ex, Ey, Exy, x_exprs, y_exprs, v_exprs, Expr):
    """Fusion expression for the scalar cut b'K(Z)b >= 0 from the Kroncut note."""

    terms = [Expr.constTerm(float(b @ identity @ b))]
    for j, xj in enumerate(x_exprs):
        coeff = float(b @ Ex[j] @ b)
        if coeff:
            terms.append(Expr.mul(coeff, xj))
    for i, yi in enumerate(y_exprs):
        coeff = float(b @ Ey[i] @ b)
        if coeff:
            terms.append(Expr.mul(coeff, yi))
    for i, row in enumerate(v_exprs):
        for j, vij in enumerate(row):
            coeff = float(b @ Exy[i][j] @ b)
            if coeff:
                terms.append(Expr.mul(coeff, vij))
    return Expr.add(terms)


def solve_shor_random_face(
    instance: QuadInstance,
    optimal_value: float,
    *,
    seed: int,
    face_tol: float = 1e-7,
    verbose: bool = False,
) -> MethodResult:
    """Re-solve the Shor optimal face with a deterministic random objective."""

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    start = time.perf_counter()
    n, m = instance.n, instance.m
    dim = 1 + m + n
    x_slice = slice(1, 1 + m)
    y_slice = slice(1 + m, dim)
    C = instance.shor_cost_matrix()

    x_trace = np.zeros((dim, dim))
    x_trace[x_slice, x_slice] = np.eye(m)
    y_trace = np.zeros((dim, dim))
    y_trace[y_slice, y_slice] = np.eye(n)

    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, dim))
    G = 0.5 * (G + G.T)
    face_width = face_tol * max(1.0, abs(optimal_value), float(np.linalg.norm(C)))

    with Model("paper_quad_shor_random_face") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)
        U = model.variable("U", Domain.inPSDCone(dim))
        model.constraint("homogeneous", U.index(0, 0), Domain.equalsTo(1.0))
        model.constraint(
            "x_ball", Expr.dot(fusion_sparse(x_trace, Matrix), U), Domain.lessThan(1.0)
        )
        model.constraint(
            "y_ball", Expr.dot(fusion_sparse(y_trace, Matrix), U), Domain.lessThan(1.0)
        )
        model.constraint(
            "optimal_face",
            Expr.dot(fusion_sparse(C, Matrix), U),
            Domain.lessThan(optimal_value + face_width),
        )
        model.objective(ObjectiveSense.Minimize, Expr.dot(fusion_sparse(G, Matrix), U))
        model.solve()

        status = str(model.getPrimalSolutionStatus())
        if model.getPrimalSolutionStatus().name != "Optimal":
            return MethodResult(
                "Shor face",
                status,
                None,
                optimal_value,
                None,
                None,
                None,
                time.perf_counter() - start,
                {"face_width": face_width},
            )
        Ustar = np.array(U.level()).reshape(dim, dim)

    x = Ustar[0, x_slice].copy()
    y = Ustar[0, y_slice].copy()
    z = Ustar[0, 1:]
    x_ball = float(x @ x)
    y_ball = float(y @ y)
    extra: dict = {
        **_rank_diagnostics(Ustar, z),
        "face_width": face_width,
        "mean_upper_bound": instance.objective(x, y),
        "x_trace": float(np.trace(Ustar[x_slice, x_slice])),
        "y_trace": float(np.trace(Ustar[y_slice, y_slice])),
    }
    return MethodResult(
        "Shor face",
        "Optimal",
        None,
        optimal_value,
        extra["mean_upper_bound"],
        x,
        y,
        time.perf_counter() - start,
        extra,
    )


def solve_quad(
    instance: QuadInstance,
    *,
    method: str = "Shor",
    solver_threads: int | None = None,
    lazy_tol: float = 1e-7,
    lazy_max_iters: int = 100,
    lazy_cuts_per_iter: int = 50,
    lazy_cut_strategy: str = "coordinate",
    verbose: bool = False,
) -> MethodResult:
    """Solve one of the Section 4.1 relaxations of (12) with Mosek Fusion."""

    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    n, m = instance.n, instance.m
    if method in ("Full SEP", "Lazy SEP") and min(n, m) < 2:
        raise NotImplementedError(
            f"{method} needs SEP(n+1,m+1) with min(n+1,m+1) >= 3, i.e. min(n,m) >= 2."
        )

    start = time.perf_counter()
    dim = 1 + m + n
    x_slice = slice(1, 1 + m)
    y_slice = slice(1 + m, dim)
    C = instance.shor_cost_matrix()

    x_trace = np.zeros((dim, dim))
    x_trace[x_slice, x_slice] = np.eye(m)
    y_trace = np.zeros((dim, dim))
    y_trace[y_slice, y_slice] = np.eye(n)

    uses_sep = method in ("Full SEP", "Lazy SEP")
    block_size = n * m
    G = _sep_atoms(n, m) if uses_sep else None

    history: list[dict[str, float | int]] = []
    active_cuts: list[tuple[np.ndarray, np.ndarray]] = []
    active_kron_cuts: list[np.ndarray] = []
    max_violation = fro_violation = None
    max_kron_violation = fro_kron_violation = None
    Kstar = None
    Tstar = None

    with Model(f"paper_quad_{method.lower().replace(' ', '_')}") as model:
        if solver_threads is not None:
            if solver_threads < 1:
                raise ValueError("solver_threads must be positive")
            model.setSolverParam("numThreads", solver_threads)
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)

        U = model.variable("U", Domain.inPSDCone(dim))
        model.constraint("homogeneous", U.index(0, 0), Domain.equalsTo(1.0))
        model.constraint(
            "x_ball", Expr.dot(fusion_sparse(x_trace, Matrix), U), Domain.lessThan(1.0)
        )
        model.constraint(
            "y_ball", Expr.dot(fusion_sparse(y_trace, Matrix), U), Domain.lessThan(1.0)
        )
        model.objective(
            ObjectiveSense.Minimize, Expr.dot(fusion_sparse(C, Matrix), U)
        )

        uses_kron = method in ("KRON", "Lazy KRON")
        if uses_kron:
            identity, Ex, Ey, Exy = _kron_coeff_matrices(n, m)
            x_exprs = [U.index(0, 1 + j) for j in range(m)]
            y_exprs = [U.index(0, 1 + m + i) for i in range(n)]
            v_exprs = [
                [U.index(1 + m + i, 1 + j) for j in range(m)] for i in range(n)
            ]
            if method == "KRON":
                K = _kron_expr_from_coeffs(
                    identity, Ex, Ey, Exy, x_exprs, y_exprs, v_exprs, Expr, Matrix
                )
                model.constraint("kron", K, Domain.inPSDCone())

        T = None
        if uses_sep:
            T = model.variable("T", Domain.inPSDCone(block_size))
            # Z_00 = tr(T) = 1 is implied by U_00 = 1; state it directly.
            model.constraint("sep_trace", Expr.sum(T.diag()), Domain.equalsTo(1.0))
            for j in range(m):
                model.constraint(
                    f"sep_x{j}",
                    Expr.sub(
                        Expr.dot(fusion_sparse(G(0, j + 1), Matrix), T),
                        U.index(0, 1 + j),
                    ),
                    Domain.equalsTo(0.0),
                )
            for i in range(n):
                model.constraint(
                    f"sep_y{i}",
                    Expr.sub(
                        Expr.dot(fusion_sparse(G(i + 1, 0), Matrix), T),
                        U.index(0, 1 + m + i),
                    ),
                    Domain.equalsTo(0.0),
                )
            for i in range(n):
                for j in range(m):
                    model.constraint(
                        f"sep_V{i}_{j}",
                        Expr.sub(
                            Expr.dot(fusion_sparse(G(i + 1, j + 1), Matrix), T),
                            U.index(1 + m + i, 1 + j),
                        ),
                        Domain.equalsTo(0.0),
                    )

        if method == "Full SEP":
            skew = skew_kron_rows(n, m)
            if skew.nnz:
                model.constraint(
                    "skew",
                    Expr.mul(fusion_sparse(skew, Matrix), Expr.flatten(T)),
                    Domain.equalsTo(0.0),
                )

        if method == "Lazy KRON":
            for iteration in range(lazy_max_iters + 1):
                solve_start = time.perf_counter()
                model.solve()
                solve_seconds = time.perf_counter() - solve_start
                if model.getPrimalSolutionStatus().name != "Optimal":
                    return MethodResult(
                        method,
                        str(model.getPrimalSolutionStatus()),
                        None,
                        None,
                        None,
                        None,
                        None,
                        time.perf_counter() - start,
                        {"n_cuts": len(active_kron_cuts), "n_iters": iteration},
                    )
                Uround = np.array(U.level()).reshape(dim, dim)
                xround = Uround[0, x_slice].copy()
                yround = Uround[0, y_slice].copy()
                Vround = Uround[y_slice, x_slice].copy()
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
                if relative_max_kron <= lazy_tol or iteration == lazy_max_iters:
                    break
                scale = max(1.0, float(np.linalg.norm(Kstar)))
                new_cuts = [
                    b
                    for b in cuts
                    if float(-(b @ Kstar @ b)) / scale > lazy_tol
                ]
                if not new_cuts:
                    break
                for offset, b in enumerate(new_cuts):
                    model.constraint(
                        f"kron_cut_{iteration}_{offset}",
                        _kron_cut_expr(
                            b, identity, Ex, Ey, Exy, x_exprs, y_exprs, v_exprs, Expr
                        ),
                        Domain.greaterThan(0.0),
                    )
                active_kron_cuts.extend(new_cuts)
        elif method == "Lazy SEP":
            tvec = Expr.flatten(T)
            for iteration in range(lazy_max_iters + 1):
                solve_start = time.perf_counter()
                model.solve()
                solve_seconds = time.perf_counter() - solve_start
                if model.getPrimalSolutionStatus().name != "Optimal":
                    return MethodResult(
                        method,
                        str(model.getPrimalSolutionStatus()),
                        None,
                        None,
                        None,
                        None,
                        None,
                        time.perf_counter() - start,
                        {"n_cuts": len(active_cuts), "n_iters": iteration},
                    )
                Tstar = np.array(T.level()).reshape(block_size, block_size)
                sep_start = time.perf_counter()
                max_violation, fro_violation, cuts = separate_skew_skew(
                    Tstar, n, m, ncuts=lazy_cuts_per_iter, strategy=lazy_cut_strategy
                )
                relative_max_violation = relative_residual(max_violation, Tstar)
                relative_fro_violation = relative_residual(fro_violation, Tstar)
                history.append(
                    {
                        "iteration": iteration,
                        "n_cuts": len(active_cuts),
                        "value": float(model.primalObjValue()),
                        "max_skew_violation": max_violation,
                        "fro_skew_violation": fro_violation,
                        "relative_max_skew_violation": relative_max_violation,
                        "relative_fro_skew_violation": relative_fro_violation,
                        "solve_seconds": solve_seconds,
                        "separation_seconds": time.perf_counter() - sep_start,
                    }
                )
                if relative_max_violation <= lazy_tol or iteration == lazy_max_iters:
                    break
                scale = max(1.0, float(np.linalg.norm(Tstar)))
                new_cuts = [(A, B) for sigma, A, B in cuts if sigma / scale > lazy_tol]
                if not new_cuts:
                    break
                model.constraint(
                    f"cuts_{iteration}",
                    Expr.mul(cut_batch_matrix(new_cuts, Matrix), tvec),
                    Domain.equalsTo(0.0),
                )
                active_cuts.extend(new_cuts)
        else:
            model.solve()

        status = str(model.getPrimalSolutionStatus())
        if model.getPrimalSolutionStatus().name != "Optimal":
            return MethodResult(
                method, status, None, None, None, None, None, time.perf_counter() - start
            )
        Ustar = np.array(U.level()).reshape(dim, dim)
        if uses_sep:
            Tstar = np.array(T.level()).reshape(block_size, block_size)
        value = float(model.primalObjValue())
        dual = float(model.dualObjValue())

    x = Ustar[0, x_slice].copy()
    y = Ustar[0, y_slice].copy()
    z = Ustar[0, 1:]
    x_ball = float(x @ x)
    y_ball = float(y @ y)
    extra: dict = {
        "pd_gap": value - dual,
        # U is rank one exactly when the relaxation is tight at (x, y).
        **_rank_diagnostics(Ustar, z),
        "x_trace": float(np.trace(Ustar[x_slice, x_slice])),
        "y_trace": float(np.trace(Ustar[y_slice, y_slice])),
        "x_ball": x_ball,
        "y_ball": y_ball,
        "primal_feasibility_violation": max(0.0, x_ball - 1.0, y_ball - 1.0),
        "mean_upper_bound": instance.objective(x, y),
    }
    if uses_sep:
        Z = np.empty((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                Z[i, j] = float(np.vdot(G(i, j), Tstar))
        extra["sep_rank1_residual"] = rank1_residual(Z)
        extra["relative_sep_rank1_residual"] = relative_residual(
            extra["sep_rank1_residual"], Z[1:, 1:], np.outer(Z[1:, 0], Z[0, 1:])
        )
    if method == "Lazy SEP":
        extra.update(
            {
                "max_skew_violation": max_violation,
                "fro_skew_violation": fro_violation,
                "relative_max_skew_violation": relative_residual(max_violation, Tstar),
                "relative_fro_skew_violation": relative_residual(fro_violation, Tstar),
                "n_cuts": len(active_cuts),
                "n_iters": len(history),
                "n_rounds": len(history),
                "full_constraint_count": len(skew_pairs(n)) * len(skew_pairs(m)),
                "lazy_cut_strategy": lazy_cut_strategy,
            }
        )
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
    if method == "Full SEP":
        extra["full_constraint_count"] = len(skew_pairs(n)) * len(skew_pairs(m))

    # Every method here is a relaxation of (12): the SDP value is a lower bound,
    # and the first-column point (x, y) is feasible, hence an upper bound.
    return MethodResult(
        method,
        "Optimal",
        value,
        value,
        extra["mean_upper_bound"],
        x,
        y,
        time.perf_counter() - start,
        extra,
    )
