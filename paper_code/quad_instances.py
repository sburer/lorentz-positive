"""Max-distance instances for Section 4.1, in the notation of ``paper/Lorentz.tex``.

Section 4.1 studies

    min  c'x + d'y + x'Qx + y'Py + y'Rx
    s.t. x in E_x subset R^m,  y in E_y subset R^n,                      (12)

which after an affine change of variables has E_x = {||x|| <= 1} and
E_y = {||y|| <= 1}.  (The draft writes the linear term on y as b'y in (12) but
as d in the augmented cost matrix (13); we follow (13) and call it d.)

The max-distance family is the largest distance between two ellipsoids,

    max ||(A x + a) - (B y + b)||^2   s.t. ||x|| <= 1, ||y|| <= 1,

where E_x = {A x + a : ||x|| <= 1} and E_y = {B y + b : ||y|| <= 1} are the two
ellipsoids in the ambient space.  Expanding with s = a - b,

    -||A x - B y + s||^2 = -x'A'Ax - y'B'By + 2 y'B'Ax - 2 s'Ax + 2 s'By - s's,

so the minimization form (12) is obtained with

    Q = -A'A,  P = -B'B,  R = 2 B'A,  c = -2 A's,  d = 2 B's,

together with the additive constant -s's, which is tracked separately in
``QuadInstance.constant`` and reinstated only when reporting the max-distance
value itself:  max ||.||^2 = -(min (12) + constant).

Dimension convention (as in the manuscript): ``x in R^m`` is the
source/right variable and ``y in R^n`` is the target/left variable, so R is
n x m and the Sep lift is Z = [[1, x'], [y, V]] in SEP(n+1, m+1) with V ~ y x'.
The generators here are ported from the exploratory ``code/max_distance_ellipsoids.py``
(which used the opposite ``n = dim(x)`` convention); the ambient dimension is
taken equal to n = m, which is the square case used in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


RANDOM_KINDS = ("balls", "axis", "ellipsoids")
ADVERSARIAL_FAMILIES = ("tied", "spiked", "graded")
GENERATORS = RANDOM_KINDS + ADVERSARIAL_FAMILIES


@dataclass(frozen=True)
class QuadInstance:
    """One instance of the paper's problem (12) over two unit balls."""

    n: int
    m: int
    seed: int
    c: np.ndarray
    d: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    R: np.ndarray
    constant: float = 0.0
    generator: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = np.asarray(self.c, dtype=float).reshape(-1)
        d = np.asarray(self.d, dtype=float).reshape(-1)
        Q = 0.5 * (np.asarray(self.Q, dtype=float) + np.asarray(self.Q, dtype=float).T)
        P = 0.5 * (np.asarray(self.P, dtype=float) + np.asarray(self.P, dtype=float).T)
        R = np.asarray(self.R, dtype=float)
        if c.shape != (self.m,):
            raise ValueError(f"c must have shape {(self.m,)}, got {c.shape}")
        if d.shape != (self.n,):
            raise ValueError(f"d must have shape {(self.n,)}, got {d.shape}")
        if Q.shape != (self.m, self.m):
            raise ValueError(f"Q must have shape {(self.m, self.m)}, got {Q.shape}")
        if P.shape != (self.n, self.n):
            raise ValueError(f"P must have shape {(self.n, self.n)}, got {P.shape}")
        if R.shape != (self.n, self.m):
            raise ValueError(f"R must have shape {(self.n, self.m)}, got {R.shape}")
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "R", R)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate c'x + d'y + x'Qx + y'Py + y'Rx (the constant is excluded)."""

        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        return float(
            self.c @ x + self.d @ y + x @ self.Q @ x + y @ self.P @ y + y @ self.R @ x
        )

    def max_distance_squared(self, objective_value: float) -> float:
        """Convert a value of (12) back to the max-distance value it encodes."""

        return -(objective_value + self.constant)

    def shor_cost_matrix(self) -> np.ndarray:
        """Return C of the manuscript's (13) in the ordering (1, x, y).

        With U = [[1, x', y'], [x, X, V'], [y, V, Y]] we have
        <C,U> = c'x + d'y + <Q,X> + <P,Y> + <R,V>.
        """

        m, n = self.m, self.n
        dim = 1 + m + n
        C = np.zeros((dim, dim))
        xs = slice(1, 1 + m)
        ys = slice(1 + m, dim)
        C[0, xs] = 0.5 * self.c
        C[xs, 0] = 0.5 * self.c
        C[0, ys] = 0.5 * self.d
        C[ys, 0] = 0.5 * self.d
        C[xs, xs] = self.Q
        C[ys, ys] = self.P
        C[ys, xs] = 0.5 * self.R
        C[xs, ys] = 0.5 * self.R.T
        return C


def max_distance_instance(
    A: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    b: np.ndarray,
    *,
    seed: int = -1,
    generator: str = "",
) -> QuadInstance:
    """Build the (12)-form instance of max ||(Ax+a)-(By+b)||^2 over two balls."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    s = a - b
    m = A.shape[1]
    n = B.shape[1]
    return QuadInstance(
        n=n,
        m=m,
        seed=seed,
        c=-2.0 * A.T @ s,
        d=2.0 * B.T @ s,
        Q=-(A.T @ A),
        P=-(B.T @ B),
        R=2.0 * B.T @ A,
        constant=-float(s @ s),
        generator=generator,
        extra={"A": A, "a": a, "B": B, "b": b},
    )


def random_max_distance(
    k: int,
    seed: int,
    *,
    kind: str = "ellipsoids",
    center_scale: float = 1.0,
    condition: float = 5.0,
) -> QuadInstance:
    """Random square max-distance instance with n = m = ambient dimension = k.

    Ported from ``code/max_distance_ellipsoids.random_instance``.  ``balls``
    uses scalar multiples of the identity, ``axis`` uses diagonal maps, and
    ``ellipsoids`` uses full random singular-vector rotations; ``condition``
    controls the singular-value spread for the non-ball kinds.
    """

    rng = np.random.default_rng(seed)
    a = center_scale * rng.standard_normal(k)
    b = center_scale * rng.standard_normal(k)
    if kind == "balls":
        A = (0.25 + 1.5 * rng.random()) * np.eye(k)
        B = (0.25 + 1.5 * rng.random()) * np.eye(k)
    elif kind == "axis":
        A = np.diag(np.exp(rng.uniform(0.0, np.log(condition), size=k)))
        B = np.diag(np.exp(rng.uniform(0.0, np.log(condition), size=k)))
    elif kind == "ellipsoids":
        A = _random_full_rank_map(rng, k, condition)
        B = _random_full_rank_map(rng, k, condition)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return max_distance_instance(A, a, B, b, seed=seed, generator=kind)


def _random_full_rank_map(
    rng: np.random.Generator, k: int, condition: float
) -> np.ndarray:
    U, _ = np.linalg.qr(rng.standard_normal((k, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    singular_values = np.exp(rng.uniform(0.0, np.log(condition), size=k))
    return U @ np.diag(singular_values) @ V.T


def adversarial_max_distance(
    k: int,
    seed: int,
    *,
    family: str = "tied",
    condition: float = 5.0,
    center_scale: float = 1.0,
) -> QuadInstance:
    """Max-distance instances with several nearly competing farthest pairs.

    Ported from ``code/max_distance_ellipsoids.adversarial_instance``.  These
    were built to defeat the Shor relaxation, which is often already tight on
    the plain random families.
    """

    rng = np.random.default_rng(seed)
    Ua, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Ub, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Va, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Vb, _ = np.linalg.qr(rng.standard_normal((k, k)))

    if family == "tied":
        # Several almost-equal long axes create competing extreme pairs.
        tail = np.geomspace(max(1.0, condition / 8.0), 1.0, max(1, k - 2))
        singular_a = np.r_[condition, condition * (1.0 - 1e-3), tail][:k]
        singular_b = np.r_[condition, condition * (1.0 - 2e-3), tail[::-1]][:k]
    elif family == "spiked":
        # A dominant two-dimensional output subspace with weak remaining axes.
        singular_a = np.ones(k)
        singular_b = np.ones(k)
        singular_a[: min(2, k)] = condition
        singular_b[: min(2, k)] = condition
    elif family == "graded":
        singular_a = np.geomspace(condition, 1.0, k)
        singular_b = np.geomspace(condition, 1.0, k)[::-1]
    else:
        raise ValueError(f"unknown adversarial family {family!r}")

    A = Ua @ np.diag(singular_a) @ Va.T
    B = Ub @ np.diag(singular_b) @ Vb.T

    # Put the relative center in the span of the strongest left singular
    # directions.  Random signs and a small orthogonal perturbation break exact
    # symmetry while retaining several attractive farthest-point pairs.
    dominant = Ua[:, : min(2, k)] @ rng.standard_normal(min(2, k))
    dominant += Ub[:, : min(2, k)] @ rng.standard_normal(min(2, k))
    dominant += 0.05 * rng.standard_normal(k)
    norm = float(np.linalg.norm(dominant))
    if norm == 0.0:
        dominant[0] = 1.0
        norm = 1.0
    diff = center_scale * condition * dominant / norm
    return max_distance_instance(
        A, 0.5 * diff, B, -0.5 * diff, seed=seed, generator=family
    )


def generate_named_instance(
    k: int,
    seed: int,
    *,
    generator: str,
    condition: float = 5.0,
    center_scale: float = 1.0,
) -> QuadInstance:
    """Dispatch to one of the max-distance generator families."""

    if generator in RANDOM_KINDS:
        return random_max_distance(
            k, seed, kind=generator, center_scale=center_scale, condition=condition
        )
    if generator in ADVERSARIAL_FAMILIES:
        return adversarial_max_distance(
            k, seed, family=generator, condition=condition, center_scale=center_scale
        )
    raise ValueError(f"unknown generator {generator!r}; choose from {GENERATORS}")


def maxdist_panel(sizes: tuple[int, ...] = (2, 4, 6, 8, 10), cases: int = 10):
    """Return (panel, n, m, seed) rows for the planned Section 4.1 max-distance run.

    Only the square case n = m is used in the paper.  The smallest
    size is n = m = 2 because the SEP block for problem (15) lives in
    SEP(n+1, m+1) and Propositions 1-2 need min(n+1, m+1) >= 3.
    """

    return [("square", k, k, seed) for k in sizes for seed in range(cases)]
