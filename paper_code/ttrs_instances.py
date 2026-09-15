"""Affine-diagonal TTRS instances used for the paper computations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io


from paper_code.external import ttrs_instance_dir

# Optional historical cross-check: corrected beta reference values keyed by
# `.mat` filename.  It is not part of the paper reproduction (no table uses
# it), so a missing file simply yields no reference values.
DEFAULT_REFERENCE_CSV: Path | None = None


@dataclass(frozen=True)
class TtrsInstance:
    """Scaled centered diagonal TTRS instance in paper-code standard form.

    The problem is

        min x'Qx + c'x
        s.t. ||x|| <= 1, ||diag(a)x + b|| <= 1.

    For the ball-constraints Section 5.3 files, `b=0` after scaling.
    """

    name: str
    n: int
    a: np.ndarray
    b: np.ndarray
    Q: np.ndarray
    c: np.ndarray
    reference_value: float | None = None

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(x @ self.Q @ x + self.c @ x)

    def x_ball_value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(x @ x)

    def y_ball_value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = self.a * x + self.b
        return float(y @ y)


def load_beta_references(path: Path | None = DEFAULT_REFERENCE_CSV) -> dict[str, float]:
    """Load corrected beta reference values keyed by `.mat` filename."""

    import csv

    if path is None or not path.exists():
        return {}
    refs: dict[str, float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("current_beta_pval"):
                refs[row["file"]] = float(row["current_beta_pval"])
    return refs


def load_ballconstraints_case(
    path: Path, references: dict[str, float] | None = None
) -> TtrsInstance:
    """Load one Section 5.3 ball-constraints TTRS `.mat` instance."""

    mat = scipy.io.loadmat(path)
    Q = np.asarray(mat["Q"], dtype=float)
    c = np.asarray(mat["c"], dtype=float).reshape(-1)
    H = np.asarray(mat["H"], dtype=float)
    h = np.asarray(mat["h"], dtype=float).reshape(-1)
    r1 = float(np.asarray(mat["r1"]).reshape(-1)[0])
    r2 = float(np.asarray(mat["r2"]).reshape(-1)[0])

    n = Q.shape[0]
    if Q.shape != (n, n) or c.shape != (n,):
        raise ValueError(f"bad Q/c shapes in {path}")
    if np.linalg.norm(h) > 1e-8:
        raise ValueError(f"nonzero second-ball center in {path}")
    if np.linalg.norm(H - np.diag(np.diag(H))) > 1e-8:
        raise ValueError(f"non-diagonal H in {path}")
    if np.any(np.diag(H) <= 0.0):
        raise ValueError(f"H is not positive diagonal in {path}")

    Q_scaled = 0.5 * (Q + Q.T) * (r1 * r1)
    c_scaled = c * r1
    a = (r1 / r2) * np.sqrt(np.diag(H))
    b = np.zeros(n)
    return TtrsInstance(
        name=path.name,
        n=n,
        a=a,
        b=b,
        Q=Q_scaled,
        c=c_scaled,
        reference_value=None if references is None else references.get(path.name),
    )


def ballconstraints_panel(
    *,
    data_dir: Path | None = None,
    reference_csv: Path | None = DEFAULT_REFERENCE_CSV,
    dimensions: set[int] | None = None,
    limit: int | None = None,
) -> list[TtrsInstance]:
    """Return sorted ball-constraints TTRS instances with beta references."""

    if data_dir is None:
        data_dir = ttrs_instance_dir()
    refs = load_beta_references(reference_csv)
    paths = sorted(data_dir.glob("instance_*.mat"))
    if not paths:
        raise FileNotFoundError(f"no instance_*.mat files under {data_dir}")
    if dimensions is not None:
        paths = [path for path in paths if int(path.stem.split("_")[1]) in dimensions]
    if limit is not None:
        paths = paths[:limit]
    return [load_ballconstraints_case(path, refs) for path in paths]


def random_centered_diag_ttrs(
    n: int,
    seed: int,
    *,
    loga_radius: float = 1.0,
) -> TtrsInstance:
    """Generate a centered diagonal TTRS instance with deterministic data.

    This matches the research generator used in the earlier generated-instance note: the
    second trust-region map is ``diag(a)x`` with log-uniform positive diagonal
    entries, while the objective has a symmetric standard-normal quadratic
    part and a standard-normal linear part.
    """

    if loga_radius < 0.0:
        raise ValueError("loga_radius must be nonnegative")
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(-loga_radius, loga_radius, size=n))
    B = rng.standard_normal((n, n))
    Q = 0.5 * (B + B.T)
    c = rng.standard_normal(n)
    return TtrsInstance(
        name=f"generated_centered_n_{n}_seed_{seed}",
        n=n,
        a=a,
        b=np.zeros(n),
        Q=Q,
        c=c,
        reference_value=None,
    )
