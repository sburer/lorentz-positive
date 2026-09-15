"""Shared scale-aware diagnostics for the paper-code experiments."""

from __future__ import annotations

import numpy as np


# One paper-wide rule: a method is called tight when its method-specific
# relative recovered-point gap is at most this value.
PAPER_TIGHT_REL_TOL = 1e-5


def value_scale(*values: float | None) -> float:
    """Return max(1, abs(values...)) while ignoring missing entries."""

    scale = 1.0
    for value in values:
        if value is not None:
            scale = max(scale, abs(float(value)))
    return scale


def relative_gap(
    upper: float | None, lower: float | None, *, floor: float = 1.0
) -> float | None:
    """Return max(0, upper-lower) divided by a natural objective scale."""

    if upper is None or lower is None:
        return None
    return max(0.0, float(upper) - float(lower)) / max(
        floor, abs(float(upper)), abs(float(lower))
    )


def relative_error(
    first: float | None, second: float | None, *, floor: float = 1.0
) -> float | None:
    """Return abs(first-second) divided by a natural objective scale."""

    if first is None or second is None:
        return None
    return abs(float(first) - float(second)) / max(
        floor, abs(float(first)), abs(float(second))
    )


def relative_residual(
    residual: float | None, *references: np.ndarray | float | None
) -> float | None:
    """Scale a residual by max(1, Frobenius norms of reference arrays)."""

    if residual is None:
        return None
    scale = 1.0
    for reference in references:
        if reference is None:
            continue
        if isinstance(reference, np.ndarray):
            scale = max(scale, float(np.linalg.norm(reference)))
        else:
            scale = max(scale, abs(float(reference)))
    return float(residual) / scale


def relative_stop(width: float, lower: float, upper: float, tolerance: float) -> bool:
    """True when an interval width is below a relative stopping tolerance."""

    return width <= tolerance * value_scale(lower, upper)
