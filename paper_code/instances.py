"""Problem instances using the notation of ``paper/Lorentz.tex``.

The pure bilinear test problem is the paper's (7),

    min c' x + d' y + y' R x
    s.t. ||x|| <= 1, ||y|| <= 1,

with x in R^m, y in R^n, c in R^m, d in R^n, and R in R^{n x m}.
The normalized Sep lift is Z = [[1, x'], [y, V]] in SEP(n+1,m+1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BilinearInstance:
    """One paper-notation bilinear instance."""

    n: int
    m: int
    seed: int
    c: np.ndarray
    d: np.ndarray
    R: np.ndarray
    data_scale: float = 1.0

    def __post_init__(self) -> None:
        c = np.asarray(self.c, dtype=float).reshape(-1)
        d = np.asarray(self.d, dtype=float).reshape(-1)
        R = np.asarray(self.R, dtype=float)
        if c.shape != (self.m,):
            raise ValueError(f"c must have shape {(self.m,)}, got {c.shape}")
        if d.shape != (self.n,):
            raise ValueError(f"d must have shape {(self.n,)}, got {d.shape}")
        if R.shape != (self.n, self.m):
            raise ValueError(f"R must have shape {(self.n, self.m)}, got {R.shape}")
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "data_scale", float(self.data_scale))

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate c'x + d'y + y'Rx."""

        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        return float(self.c @ x + self.d @ y + y @ self.R @ x)

    def sep_cost_matrix(self) -> np.ndarray:
        """Return C = [[0, c'], [d, R]] with <C,Z> = c'x + d'y + y'Rx.

        This is the augmented cost matrix of the paper's Section 3, matching
        the Sep block V = y x'.
        """

        C = np.zeros((self.n + 1, self.m + 1))
        C[0, 1:] = self.c
        C[1:, 0] = self.d
        C[1:, 1:] = self.R
        return C


def generate_instance(n: int, m: int, seed: int, *, scale: float = 1.0) -> BilinearInstance:
    """Generate i.i.d. standard-normal data in paper notation."""

    rng = np.random.default_rng(seed)
    # Draw in the transposed shape so the singular-vector convention matches
    # the correlated generator below before converting to paper notation.
    return _normalize_instance(
        BilinearInstance(
            n=n,
            m=m,
            seed=seed,
            c=scale * rng.standard_normal(m),
            d=scale * rng.standard_normal(n),
            R=scale * rng.standard_normal((m, n)).T,
        )
    )


def _normalize_instance(instance: BilinearInstance) -> BilinearInstance:
    """Scale (c,d,R) so max(||c||,||d||)=1 when possible."""

    data_scale = max(float(np.linalg.norm(instance.c)), float(np.linalg.norm(instance.d)))
    if data_scale == 0.0:
        return instance
    return BilinearInstance(
        n=instance.n,
        m=instance.m,
        seed=instance.seed,
        c=instance.c / data_scale,
        d=instance.d / data_scale,
        R=instance.R / data_scale,
        data_scale=data_scale,
    )


def generate_correlated_top_instance(
    n: int,
    m: int,
    seed: int,
    *,
    alpha: float = 3.0,
    sigma: float = 0.1,
) -> BilinearInstance:
    """Generate a correlated Gaussian instance with Shor-non-tight bias.

    The bilinear block R in R^{n x m} is standard Gaussian.  Let u_1 in R^n and
    v_1 in R^m be the leading left and right singular vectors of R.  The linear
    terms are correlated with that dominant singular mode, plus independent
    Gaussian noise: c = alpha*v_1 + sigma*g pairs with x in R^m, and
    d = alpha*u_1 + sigma*h pairs with y in R^n.
    """

    rng = np.random.default_rng(seed)
    # Factor R' = U S Vt so the columns of U are the right singular vectors of
    # R and the columns of Vt' are its left singular vectors.
    Rt = rng.standard_normal((m, n))
    U, _, Vt = np.linalg.svd(Rt, full_matrices=True)
    v1 = U[:, 0]
    u1 = Vt.T[:, 0]
    c = alpha * v1 + sigma * rng.standard_normal(m)
    d = alpha * u1 + sigma * rng.standard_normal(n)
    return _normalize_instance(BilinearInstance(n=n, m=m, seed=seed, c=c, d=d, R=Rt.T))


def generate_named_instance(
    n: int,
    m: int,
    seed: int,
    *,
    generator: str,
    alpha: float = 3.0,
    sigma: float = 0.1,
) -> BilinearInstance:
    """Generate an instance from one of the paper-code generator families."""

    if generator == "iid":
        return generate_instance(n, m, seed)
    if generator == "correlated-top":
        return generate_correlated_top_instance(n, m, seed, alpha=alpha, sigma=sigma)
    raise ValueError(f"unknown generator {generator!r}")


def paper_panel() -> list[tuple[str, int, int, int]]:
    """Return (panel, n, m, seed) for the planned Section 3.1 run."""

    rows: list[tuple[str, int, int, int]] = []
    for k in (2, 4, 6, 8, 10, 15, 20):
        for seed in range(10):
            rows.append(("square", k, k, seed))
    for n, m in ((4, 8), (8, 12), (10, 20), (15, 20), (15, 25)):
        for seed in range(10):
            rows.append(("rectangular", n, m, seed))
    return rows
