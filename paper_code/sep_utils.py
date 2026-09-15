"""Shared Hildebrand-II Sep utilities for the paper-code solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp


def W_basis(r: int) -> list[np.ndarray]:
    """Arrowhead basis W_r(e_i) from Hildebrand's polynomial LOP LMI."""

    d = r - 1
    mats = []
    for i in range(r):
        mat = np.zeros((d, d))
        if i == 0:
            mat = np.eye(d)
        elif i == 1:
            mat[0, 0] = 1.0
            for j in range(1, d):
                mat[j, j] = -1.0
        else:
            j = i - 1
            mat[0, j] = 1.0
            mat[j, 0] = 1.0
        mats.append(mat)
    return mats


def skew_pairs(k: int) -> list[tuple[int, int]]:
    """Index pairs for the orthonormal skew basis (E_ij-E_ji)/sqrt(2)."""

    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def skew_from_coeffs(
    k: int, pairs: list[tuple[int, int]], coeffs: np.ndarray
) -> np.ndarray:
    """Build a skew matrix from coefficients in the orthonormal skew basis."""

    mat = np.zeros((k, k))
    scale = 1.0 / np.sqrt(2.0)
    for coeff, (i, j) in zip(coeffs, pairs):
        mat[i, j] += scale * coeff
        mat[j, i] -= scale * coeff
    return mat


def skew_kron_rows(n: int, m: int) -> sp.csc_matrix:
    """Sparse rows encoding all skew-skew equalities <A tensor B,S>=0."""

    d = n * m
    rows, cols, vals = [], [], []
    row = 0
    for a, b in combinations(range(n), 2):
        for c, e in combinations(range(m), 2):
            for i, j, value in (
                (a * m + c, b * m + e, 1.0),
                (a * m + e, b * m + c, -1.0),
                (b * m + c, a * m + e, -1.0),
                (b * m + e, a * m + c, 1.0),
            ):
                rows.append(row)
                cols.append(i * d + j)
                vals.append(value)
            row += 1
    return sp.csc_matrix((vals, (rows, cols)), shape=(max(row, 1), d * d))


def skew_residual_matrix(S: np.ndarray, n: int, m: int) -> np.ndarray:
    """Return R_ij = <A_i tensor B_j,S> over orthonormal skew bases."""

    pairs_n = skew_pairs(n)
    pairs_m = skew_pairs(m)
    residual = np.zeros((len(pairs_n), len(pairs_m)))
    for row, (a, b) in enumerate(pairs_n):
        for col, (c, e) in enumerate(pairs_m):
            residual[row, col] = 0.5 * (
                S[a * m + c, b * m + e]
                - S[a * m + e, b * m + c]
                - S[b * m + c, a * m + e]
                + S[b * m + e, a * m + c]
            )
    return residual


def separate_skew_skew(
    S: np.ndarray, n: int, m: int, *, ncuts: int, strategy: str = "svd"
) -> tuple[float, float, list[tuple[float, np.ndarray, np.ndarray]]]:
    """Separate omitted skew-skew equalities.

    The default ``svd`` strategy adds the most violated dense singular-vector
    cuts.  The ``coordinate`` strategy instead adds individual sparse
    skew-skew equations with largest residual entries; this can be numerically
    preferable for large TTRS instances where many cuts are needed.
    """

    residual = skew_residual_matrix(S, n, m)
    if residual.size == 0:
        return 0.0, 0.0, []
    if strategy not in ("svd", "coordinate"):
        raise ValueError("strategy must be 'svd' or 'coordinate'")
    if strategy == "coordinate":
        pairs_n = skew_pairs(n)
        pairs_m = skew_pairs(m)
        flat = np.abs(residual).ravel()
        count = min(ncuts, flat.size)
        if count == 0:
            return 0.0, float(np.linalg.norm(residual)), []
        if count == flat.size:
            indices = np.argsort(flat)[::-1]
        else:
            candidates = np.argpartition(flat, -count)[-count:]
            indices = candidates[np.argsort(flat[candidates])[::-1]]
        cuts = []
        for index in indices:
            row, col = np.unravel_index(int(index), residual.shape)
            coeff_n = np.zeros(len(pairs_n))
            coeff_m = np.zeros(len(pairs_m))
            coeff_n[row] = 1.0
            coeff_m[col] = 1.0
            cuts.append(
                (
                    float(abs(residual[row, col])),
                    skew_from_coeffs(n, pairs_n, coeff_n),
                    skew_from_coeffs(m, pairs_m, coeff_m),
                )
            )
        return float(np.max(flat)), float(np.linalg.norm(residual)), cuts
    U, sigmas, Vt = np.linalg.svd(residual, full_matrices=False)
    pairs_n = skew_pairs(n)
    pairs_m = skew_pairs(m)
    cuts = []
    for k, sigma in enumerate(sigmas[:ncuts]):
        A = skew_from_coeffs(n, pairs_n, U[:, k])
        B = skew_from_coeffs(m, pairs_m, Vt[k, :])
        cuts.append((float(sigma), A, B))
    return float(sigmas[0]), float(np.linalg.norm(residual)), cuts


def fusion_sparse(array: np.ndarray, Matrix):
    """Convert a NumPy/SciPy array to a Mosek Fusion sparse matrix."""

    coo = sp.coo_matrix(array)
    return Matrix.sparse(
        coo.shape[0],
        coo.shape[1],
        coo.row.tolist(),
        coo.col.tolist(),
        coo.data.tolist(),
    )


def cut_batch_matrix(cuts: list[tuple[np.ndarray, np.ndarray]], Matrix):
    """Build a Fusion sparse matrix T with (T vec(S))_k=<A_k tensor B_k,S>."""

    if not cuts:
        raise ValueError("cuts must be nonempty")
    n = cuts[0][0].shape[0]
    m = cuts[0][1].shape[0]
    d = n * m
    rows, cols, vals = [], [], []
    for row, (A, B) in enumerate(cuts):
        K = sp.kron(sp.coo_matrix(A), sp.coo_matrix(B), format="coo")
        rows.extend([row] * K.nnz)
        cols.extend((K.row * d + K.col).tolist())
        vals.extend(K.data.tolist())
    return Matrix.sparse(len(cuts), d * d, rows, cols, vals)


def rank1_residual(Z: np.ndarray) -> float:
    """Return ||V-yx'||_F for Z = [[1,x'],[y,V]]."""

    x = Z[0, 1:]
    y = Z[1:, 0]
    return float(np.linalg.norm(Z[1:, 1:] - np.outer(y, x)))


@dataclass
class MethodResult:
    """Common result object used by paper-code methods."""

    method: str
    status: str
    value: float | None
    lower_bound: float | None
    upper_bound: float | None
    x: np.ndarray | None
    y: np.ndarray | None
    runtime: float
    extra: dict = field(default_factory=dict)
