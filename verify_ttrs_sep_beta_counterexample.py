#!/usr/bin/env python3
"""Verify the shifted TTRS counterexample where SEP and beta are inexact.

This is a fixed two-dimensional instance with rational data, separate from the
computational panels.  It follows the paper notation

    min x'Qx + c'x
    s.t. ||x|| <= 1, ||Ax + b|| <= 1,

with diagonal positive A.  The script solves the Full SEP relaxation using the
paper-code TTRS model, solves the natural shifted beta relaxation using the
ballconstraints SDP builder, and certifies the true optimum by enumerating the
one-dimensional boundary arcs of the feasible set.
"""

from __future__ import annotations

import argparse
import sys
import types

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from paper_code.external import ballconstraints_root
from paper_code.ttrs_gurobi import solve_ttrs_gurobi
from paper_code.ttrs_instances import TtrsInstance
from paper_code.ttrs_models import solve_ttrs_relax


@dataclass(frozen=True)
class CounterexampleValues:
    shor_bound: float
    sep_bound: float
    beta_bound: float
    optimum: float


KNOWN_VALUES = CounterexampleValues(
    shor_bound=-0.768293943469,
    sep_bound=-0.632844508289,
    beta_bound=-0.550899217259,
    optimum=-0.546797627007,
)


def format_vector(vector: np.ndarray) -> str:
    return np.array2string(
        np.asarray(vector, dtype=float),
        formatter={"float_kind": lambda value: f"{value:.12f}"},
    )


def counterexample_instance() -> TtrsInstance:
    """Return the shifted n=2 instance from the counterexample note.

    All data are rational with small denominators:

        a = (4/3, 13/12),   b = (11/20, -1/20),
        Q = -(1/16) [[24, 1], [1, 13]],   c = (-1, 0).

    Equivalently, scaling the objective by 16 (which scales every bound and the
    optimum by 16 and so leaves the counterexample intact) makes the objective
    data integral: Q = [[-24, -1], [-1, -13]], c = (-16, 0).
    """

    a = np.array([4 / 3, 13 / 12])
    b = np.array([11 / 20, -1 / 20])
    Q = np.array(
        [
            [-24 / 16, -1 / 16],
            [-1 / 16, -13 / 16],
        ]
    )
    c = np.array([-1.0, 0.0])
    return TtrsInstance(
        name="shifted-n2-sep-beta-counterexample",
        n=2,
        a=a,
        b=b,
        Q=Q,
        c=c,
        reference_value=KNOWN_VALUES.optimum,
    )


def _periodic_roots(function, grid_size: int = 20_000) -> list[float]:
    grid = np.linspace(0.0, 2.0 * np.pi, grid_size + 1)
    values = np.array([function(theta) for theta in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if abs(f_left) < 1e-12:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(float(brentq(function, left, right, xtol=1e-14)))
    roots.sort()
    unique: list[float] = []
    for root in roots:
        wrapped = root % (2.0 * np.pi)
        if not unique or abs(wrapped - unique[-1]) > 1e-8:
            unique.append(wrapped)
    if len(unique) > 1 and 2.0 * np.pi - unique[-1] + unique[0] < 1e-8:
        unique.pop()
    return unique


def boundary_enumeration(
    instance: TtrsInstance,
) -> tuple[float, np.ndarray, int]:
    """Certify the n=2 optimum by checking both ellipse boundary arcs."""

    a = instance.a
    b = instance.b

    def first_curve(theta: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([np.cos(theta), np.sin(theta)])
        dx = np.array([-np.sin(theta), np.cos(theta)])
        return x, dx

    def second_curve(theta: float) -> tuple[np.ndarray, np.ndarray]:
        u = np.array([np.cos(theta), np.sin(theta)])
        du = np.array([-np.sin(theta), np.cos(theta)])
        return (u - b) / a, du / a

    candidates: list[np.ndarray] = []
    for curve, other_residual in (
        (first_curve, lambda x: np.linalg.norm(a * x + b) ** 2 - 1.0),
        (second_curve, lambda x: np.linalg.norm(x) ** 2 - 1.0),
    ):
        derivative = lambda theta: float(
            (2.0 * instance.Q @ curve(theta)[0] + instance.c) @ curve(theta)[1]
        )
        intersection = lambda theta: float(other_residual(curve(theta)[0]))
        for theta in _periodic_roots(derivative) + _periodic_roots(intersection):
            x, _ = curve(theta)
            if np.linalg.norm(x) <= 1.0 + 1e-9 and np.linalg.norm(a * x + b) <= 1.0 + 1e-9:
                candidates.append(x)

    if not candidates:
        raise RuntimeError("boundary enumeration produced no feasible candidates")
    values = np.array([instance.objective(x) for x in candidates])
    best = int(np.argmin(values))
    return float(values[best]), candidates[best], len(candidates)


def _load_ballconstraints():
    """Load the companion ballconstraints SDP builder (see paper_code.external)."""

    root = ballconstraints_root() / "src"
    sys.path.insert(0, str(root))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    from define_functions import build_sdp_standard_form, solve_sdp_standard_form

    return build_sdp_standard_form, solve_sdp_standard_form


def build_shifted_beta_standard_form(instance: TtrsInstance) -> dict[str, object]:
    """Build the natural shifted beta lift for ||x|| and ||Ax+b||."""

    a = np.asarray(instance.a, dtype=float).reshape(-1)
    b = np.asarray(instance.b, dtype=float).reshape(-1)
    Q = np.asarray(instance.Q, dtype=float)
    c = np.asarray(instance.c, dtype=float).reshape(-1)
    n = instance.n
    dim = 1 + 2 * n

    Qlift = np.zeros((dim, dim))
    Qlift[0, 1 : n + 1] = 0.5 * c
    Qlift[1 : n + 1, 0] = 0.5 * c
    Qlift[1 : n + 1, 1 : n + 1] = 0.5 * (Q + Q.T)

    first_ball = np.zeros((1, dim))
    first_ball[0, 0] = 1.0
    first_ball[0, n + 1 :] = -1.0

    second_ball = np.zeros((1, dim))
    second_ball[0, 0] = 1.0 - float(b @ b)
    second_ball[0, 1 : n + 1] = -2.0 * a * b
    second_ball[0, n + 1 :] = -(a * a)

    rows = [first_ball, second_ball]
    cone_dims = [2]
    cone_types = ["nonneg"]
    for i in range(n):
        rsoc = np.zeros((3, dim))
        rsoc[0, 0] = 0.5
        rsoc[1, n + 1 + i] = 1.0
        rsoc[2, 1 + i] = 1.0
        rows.append(rsoc)
        cone_dims.append(3)
        cone_types.append("rsoc")

    return {
        "n": dim,
        "Q": Qlift,
        "A": np.vstack(rows),
        "K_dims": cone_dims,
        "K_types": cone_types,
        "x0": np.r_[1.0, np.zeros(2 * n)].reshape(-1, 1),
        "first_entry_1": True,
        "Shor_implies_feas": True,
        "Shor_bounded": True,
    }


def solve_shifted_beta(instance: TtrsInstance) -> dict[str, float | str]:
    build_sdp_standard_form, solve_sdp_standard_form = _load_ballconstraints()
    std = build_shifted_beta_standard_form(instance)
    options = {
        "Shor": True,
        "RLT": True,
        "SOCRLT": True,
        "Kron": True,
        "singleRLT0": True,
    }
    sdp = build_sdp_standard_form(std, options)
    result = solve_sdp_standard_form(std, sdp)
    return {
        "status": str(result["return_code"]),
        "primal": float(result["pval"]),
        "dual": float(result["dval"]),
        "rel_gap": float(result["rel_gap"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-gurobi", action="store_true")
    parser.add_argument("--gurobi-time-limit", type=float, default=120.0)
    args = parser.parse_args()

    instance = counterexample_instance()
    shor = solve_ttrs_relax(instance, method="Shor")
    sep = solve_ttrs_relax(instance, method="Full SEP", lazy_tol=1e-9)
    beta = solve_shifted_beta(instance)
    enum_value, enum_x, enum_count = boundary_enumeration(instance)

    gurobi = None
    if args.with_gurobi:
        gurobi = solve_ttrs_gurobi(
            instance,
            time_limit=args.gurobi_time_limit,
            mip_gap=1e-10,
            verbose=False,
        )

    print("Shifted n=2 TTRS counterexample")
    print("--------------------------------")
    print(f"a:                         {format_vector(instance.a)}")
    print(f"b:                         {format_vector(instance.b)}")
    print(f"min eig(Q):                {np.linalg.eigvalsh(instance.Q)[0]:.12f}")
    print()
    print(f"Shor bound:                {shor.lower_bound:.12f}")
    print(f"Full SEP bound:            {sep.lower_bound:.12f}")
    print(f"shifted beta dual bound:   {beta['dual']:.12f}")
    print(f"boundary optimum:          {enum_value:.12f}")
    print(f"boundary optimizer:        {format_vector(enum_x)}")
    print(f"boundary candidates:       {enum_count}")
    if gurobi is not None:
        print(f"Gurobi status:             {gurobi.status}")
        print(f"Gurobi incumbent:          {gurobi.upper_bound:.12f}")
        print(f"Gurobi lower bound:        {gurobi.lower_bound:.12f}")
    print()
    print(f"OPT - SEP gap:             {enum_value - float(sep.lower_bound):.12f}")
    print(f"OPT - beta gap:            {enum_value - float(beta['dual']):.12f}")
    print(f"SEP x-norm squared:        {sep.extra['x_ball']:.12f}")
    print(f"SEP y-norm squared:        {sep.extra['y_ball']:.12f}")
    print(f"SEP trace x moment:        {sep.extra['x_trace']:.12f}")
    print(f"SEP trace y moment:        {sep.extra['y_trace']:.12f}")
    print(f"SEP rank-one residual:     {sep.extra['rank1_residual']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
