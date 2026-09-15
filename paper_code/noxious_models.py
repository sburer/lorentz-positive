"""Relaxations for the planar noxious-facility problem (Section 4.3).

This is the unit-sphere formulation.  With ``sigma >= 0`` and
``||x||^2 + sigma^2 = 1``, ``r_i = sqrt(1 + ||p_i||^2)`` and
``pbar_i = p_i / r_i``, define the Lorentz vectors

    z_0 = (1, x_1, x_2, sigma),
    z_i = (r_i - pbar_i'x, pbar_i'x, theta, sigma),   i = 1..m,

so that ``z_i in L_4`` is equivalent at rank one to ``theta^2 <= ||x-p_i||^2``.
For the moment matrix ``H = (1,u)(1,u)'`` with ``u = (x_1, x_2, theta, sigma)``
every cone pair gives the affine image ``M_ij = G_i H G_j' in SEP(4,4)``.

Methods: ``Shor``, ``RLT``, ``KRON``, ``Full SEP``, ``Lazy SEP``.

The generic conic standard-form builder comes from the ``ballconstraints``
submodule (see :mod:`paper_code.external`).
"""

from __future__ import annotations

import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .external import ballconstraints_root
from .noxious_instances import FacilityInstance
from .sep_utils import (
    W_basis,
    cut_batch_matrix,
    fusion_sparse,
    separate_skew_skew,
    skew_kron_rows,
)


@dataclass(frozen=True)
class FacilityResult:
    """Result from one noxious-facility relaxation."""

    method: str
    status: str
    instance: str
    num_sites: int
    theta_bound: float | None
    distance_bound: float | None
    runtime: float
    extra: dict[str, float | int | None]


@dataclass(frozen=True)
class FacilityGurobiResult:
    """Global nonconvex QCP reference for the original facility problem."""

    status: str
    theta: float | None
    x: np.ndarray | None
    mip_gap: float | None
    runtime: float




def _load_ballconstraints_api(root: Path | None = None):
    """Load the generic conic SDP builder from the ballconstraints checkout."""

    if root is None:
        root = ballconstraints_root()
    sys.modules.setdefault("pandas", types.SimpleNamespace())
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from define_functions import build_sdp_standard_form, solve_sdp_standard_form  # type: ignore

    return build_sdp_standard_form, solve_sdp_standard_form


def _sqrt_psd(matrix: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return a row-minimal factor L with L.T @ L equal to a PSD matrix."""

    eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if eigvals[0] < -tolerance:
        raise ValueError(f"matrix is not PSD; lambda_min={eigvals[0]}")
    keep = eigvals > tolerance
    if not np.any(keep):
        return np.zeros((0, matrix.shape[0]))
    return np.diag(np.sqrt(eigvals[keep])) @ eigvecs[:, keep].T


def _soc_transform(cone_type: str, dim: int) -> np.ndarray:
    """Map SOC/RSOC coordinates to ordinary Lorentz coordinates."""

    if cone_type == "soc":
        return np.eye(dim)
    if cone_type != "rsoc":
        raise ValueError(f"unsupported cone type {cone_type!r}")
    transform = np.zeros((dim, dim))
    transform[0, 0] = 1.0
    transform[0, 1] = 1.0
    transform[1, 0] = 1.0
    transform[1, 1] = -1.0
    for i in range(2, dim):
        transform[i, i] = math.sqrt(2.0)
    return transform


def _soc_pair_count(standard: dict[str, object]) -> int:
    """Count SOC/RSOC cone pairs available for KRON or SEP."""

    types = [str(v) for v in standard["K_types"]]
    return sum(
        1
        for k in range(len(types))
        for ell in range(k)
        if types[k] in {"soc", "rsoc"} and types[ell] in {"soc", "rsoc"}
    )


def add_full_sep_pair_products(model, X, standard: dict[str, object]) -> int:
    """Replace SOC/RSOC KRON blocks by Full SEP blocks for each cone pair."""

    from mosek.fusion import Domain, Expr, Matrix

    dim = int(standard["n"])
    A = np.asarray(standard["A"], dtype=float)
    cone_dims = [int(v) for v in standard["K_dims"]]
    cone_types = [str(v) for v in standard["K_types"]]

    blocks: list[np.ndarray] = []
    offset = 0
    for cone_dim in cone_dims:
        blocks.append(A[offset : offset + cone_dim, :dim])
        offset += cone_dim

    count = 0
    for k in range(len(cone_dims)):
        for ell in range(k):
            if cone_types[k] not in {"soc", "rsoc"}:
                continue
            if cone_types[ell] not in {"soc", "rsoc"}:
                continue

            Lk = _soc_transform(cone_types[k], cone_dims[k]) @ blocks[k]
            Ll = _soc_transform(cone_types[ell], cone_dims[ell]) @ blocks[ell]
            p, q = cone_dims[k], cone_dims[ell]
            Wp = W_basis(p)
            Wq = W_basis(q)
            S = model.variable(
                f"Full_SEP_cone_{k}_{ell}_S",
                Domain.inPSDCone((p - 1) * (q - 1)),
            )
            for a in range(p):
                for b in range(q):
                    entry = Expr.dot(Lk[a, :], Expr.mul(X, Ll[b, :].T))
                    G = np.kron(Wp[a], Wq[b])
                    model.constraint(
                        f"Full_SEP_cone_{k}_{ell}_{a}_{b}",
                        Expr.sub(Expr.dot(fusion_sparse(G, Matrix), S), entry),
                        Domain.equalsTo(0.0),
                    )

            skew_rows = skew_kron_rows(p - 1, q - 1)
            if skew_rows.nnz:
                model.constraint(
                    f"Full_SEP_cone_{k}_{ell}_skew",
                    Expr.mul(fusion_sparse(skew_rows, Matrix), Expr.flatten(S)),
                    Domain.equalsTo(0.0),
                )
            count += 1
    return count




def solve_facility_gurobi(
    instance: FacilityInstance,
    *,
    time_limit: float | None = None,
    output: bool = False,
) -> FacilityGurobiResult:
    """Globally solve the original two-dimensional reverse-convex QCP."""

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model()
    model.Params.OutputFlag = int(output)
    model.Params.NonConvex = 2
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    x1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x1")
    x2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x2")
    theta = model.addVar(lb=0.0, ub=instance.theta_upper, name="theta")
    for k, (a, b) in enumerate(
        zip(instance.halfspace_A, instance.halfspace_b, strict=True)
    ):
        model.addConstr(a[0] * x1 + a[1] * x2 <= b, name=f"hull_{k}")
    for k, p in enumerate(instance.points):
        model.addQConstr(
            theta <= (x1 - p[0]) * (x1 - p[0]) + (x2 - p[1]) * (x2 - p[1]),
            name=f"dist_{k}",
        )
    model.setObjective(theta, GRB.MAXIMIZE)

    start = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - start
    status = _gurobi_status_name(model.Status, GRB)
    if model.SolCount > 0:
        theta_value = float(theta.X)
        x_value = np.array([x1.X, x2.X], dtype=float)
    else:
        theta_value = None
        x_value = None
    mip_gap = float(model.MIPGap) if model.SolCount > 0 else None
    model.dispose()
    return FacilityGurobiResult(
        status=status,
        theta=theta_value,
        x=x_value,
        mip_gap=mip_gap,
        runtime=runtime,
    )


def _gurobi_status_name(status: int, grb) -> str:
    """Return a stable name for a Gurobi status code."""

    names = {
        grb.OPTIMAL: "OPTIMAL",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.NUMERIC: "NUMERIC",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
    }
    if status in names:
        return names[status]
    return str(status)




METHODS = (
    "Shor",
    "RLT",
    "KRON",
    "Full SEP",
    "Lazy SEP",
)


@dataclass(frozen=True)
class FacilityLift:
    """Standard-form data for the affine unit-sphere construction."""

    standard: dict[str, object]
    sphere_matrix: np.ndarray
    cone_maps: tuple[np.ndarray, ...]
    x_indices: tuple[int, int]
    theta_index: int
    sigma_index: int
    num_sites: int
    num_hull_facets: int


def build_facility_lift(instance: FacilityInstance) -> FacilityLift:
    """Build the affine Lorentz-cone maps for the unit-sphere model.

    The leading coordinate is fixed to one solely to represent affine products
    in the moment matrix.  It is not the artificial homogenizing sphere used by
    the legacy formulation.
    """

    max_site_norm = float(np.max(np.linalg.norm(instance.points, axis=1)))
    if max_site_norm > 1.0 + 1e-10:
        raise ValueError(
            "the unit-sphere formulation requires all sites in the unit disk; "
            f"maximum norm is {max_site_norm:.12g}"
        )

    constant, x1, x2, theta, sigma = range(5)
    dim = 5

    # The nonnegative block contains exactly the convex-hull slacks.  This lets
    # the generic RLT layer reproduce the RLT system for the unit-sphere model.
    hull_rows = np.zeros((len(instance.halfspace_b), dim))
    hull_rows[:, constant] = instance.halfspace_b
    hull_rows[:, x1] = -instance.halfspace_A[:, 0]
    hull_rows[:, x2] = -instance.halfspace_A[:, 1]

    z0 = np.zeros((4, dim))
    z0[0, constant] = 1.0
    z0[1, x1] = 1.0
    z0[2, x2] = 1.0
    z0[3, sigma] = 1.0

    cone_maps = [z0]
    for point in instance.points:
        radius = math.sqrt(1.0 + float(point @ point))
        scaled = point / radius
        zi = np.zeros((4, dim))
        zi[0, constant] = radius
        zi[0, x1] = -scaled[0]
        zi[0, x2] = -scaled[1]
        zi[1, x1] = scaled[0]
        zi[1, x2] = scaled[1]
        zi[2, theta] = 1.0
        zi[3, sigma] = 1.0
        cone_maps.append(zi)

    A_blocks = [hull_rows, *cone_maps]
    cone_dims = [len(hull_rows), *([4] * len(cone_maps))]
    cone_types = ["nonneg", *(["soc"] * len(cone_maps))]

    # Minimizing -theta is represented as <Q,H> with symmetric Q.
    Q = np.zeros((dim, dim))
    Q[constant, theta] = -0.5
    Q[theta, constant] = -0.5

    J4 = np.diag([1.0, -1.0, -1.0, -1.0])
    sphere_matrix = z0.T @ J4 @ z0
    standard = {
        "n": dim,
        "Q": Q,
        "A": np.vstack(A_blocks),
        "K_dims": cone_dims,
        "K_types": cone_types,
        "x0": np.array([[1.0], [0.0], [0.0], [0.0], [0.0]]),
        "first_entry_1": True,
        # First-column constraints are added explicitly by method tier below.
        "Shor_implies_feas": True,
        "Shor_bounded": True,
    }
    return FacilityLift(
        standard=standard,
        sphere_matrix=sphere_matrix,
        cone_maps=tuple(cone_maps),
        x_indices=(x1, x2),
        theta_index=theta,
        sigma_index=sigma,
        num_sites=len(instance.points),
        num_hull_facets=len(instance.halfspace_b),
    )


def _add_unit_sphere_constraints(model, H, lift: FacilityLift, *, with_rlt: bool) -> None:
    """Add the sphere, sigma sign, and stated hull/SOC-RLT constraints."""

    from mosek.fusion import Domain, Expr, Matrix

    model.constraint(
        "unit_sphere_equality",
        Expr.dot(fusion_sparse(lift.sphere_matrix, Matrix), H),
        Domain.equalsTo(0.0),
    )
    model.constraint(
        "sigma_nonnegative",
        H.index([lift.sigma_index, 0]),
        Domain.greaterThan(0.0),
    )
    A = np.asarray(lift.standard["A"], dtype=float)
    hull = A[: lift.num_hull_facets, :]
    model.constraint(
        "hull_first_column",
        Expr.flatten(Expr.mul(fusion_sparse(hull, Matrix), H.slice([0, 0], [5, 1]))),
        Domain.greaterThan(0.0),
    )
    if not with_rlt:
        return

    z0 = lift.cone_maps[0]
    first_column = H.slice([0, 0], [5, 1])
    for index, cone_map in enumerate(lift.cone_maps):
        model.constraint(
            f"affine_soc_first_column_{index}",
            Expr.flatten(Expr.mul(fusion_sparse(cone_map, Matrix), first_column)),
            Domain.inQCone(4),
        )
    for index, slack in enumerate(hull):
        product = Expr.mul(z0, Expr.mul(H, slack.T))
        model.constraint(
            f"hull_unit_ball_socrlt_{index}",
            Expr.flatten(product),
            Domain.inQCone(4),
        )


def _build_base_model(
    lift: FacilityLift,
    *,
    method: str,
    ballconstraints: Path | None,
    mosek_time_limit: float | None,
):
    """Build the common Shor/RLT/KRON model without SEP auxiliaries."""

    build_sdp_standard_form, solve_sdp_standard_form = _load_ballconstraints_api(
        ballconstraints
    )
    with_rlt = method != "Shor"
    options = {
        "Shor": True,
        "RLT": with_rlt,
        "SOCRLT": False,
        "Kron": method == "KRON",
        "singleRLT0": False,
    }
    sdp = build_sdp_standard_form(lift.standard, options)
    model = sdp["M"]
    H = sdp["X"]
    if mosek_time_limit is not None:
        model.setSolverParam("optimizerMaxTime", float(mosek_time_limit))
    _add_unit_sphere_constraints(model, H, lift, with_rlt=with_rlt)
    return sdp, solve_sdp_standard_form


def _result_from_standard_solve(
    instance: FacilityInstance,
    lift: FacilityLift,
    method: str,
    result: dict[str, object],
    runtime: float,
    *,
    sep_pair_blocks: int = 0,
) -> FacilityResult:
    """Translate the companion solver result to the facility result schema."""

    theta_bound = None
    if result["return_code"] == "PrimalAndDualFeasible":
        theta_bound = -float(result["dval"])
    return FacilityResult(
        method=method,
        status=str(result["return_code"]),
        instance=instance.name,
        num_sites=lift.num_sites,
        theta_bound=theta_bound,
        distance_bound=theta_bound,
        runtime=runtime,
        extra={
            "formulation": "unit-sphere",
            "theta_semantics": "distance",
            "squared_distance_bound": (
                None if theta_bound is None else max(0.0, theta_bound) ** 2
            ),
            "rel_gap": float(result["rel_gap"]),
            "eval_ratio": float(result["eval_ratio"]),
            "soc_pair_blocks": _soc_pair_count(lift.standard),
            "sep_pair_blocks": sep_pair_blocks,
            "num_distance_blocks": lift.num_sites,
            "num_hull_facets": lift.num_hull_facets,
            "num_cone_blocks": len(lift.standard["K_dims"]),
            "num_nonnegative_rows": lift.num_hull_facets,
            "radius_squared": 1.0,
            "lazy_cut_strategy": "",
            "n_cuts": 0,
            "n_rounds": 1,
            "max_skew_violation": None,
        },
    )


def _add_relaxed_sep_pairs(model, H, lift: FacilityLift):
    """Add SEP linking equations but initially omit skew-skew equations."""

    from mosek.fusion import Domain, Expr, Matrix

    W = W_basis(4)
    auxiliaries = {}
    for right in range(1, len(lift.cone_maps)):
        for left in range(right):
            Br = lift.cone_maps[right]
            Bl = lift.cone_maps[left]
            S = model.variable(
                f"Lazy_SEP_cone_{right}_{left}_S", Domain.inPSDCone(9)
            )
            for row in range(4):
                for col in range(4):
                    moment = Expr.dot(Br[row, :], Expr.mul(H, Bl[col, :].T))
                    image = np.kron(W[row], W[col])
                    model.constraint(
                        f"Lazy_SEP_link_{right}_{left}_{row}_{col}",
                        Expr.sub(Expr.dot(fusion_sparse(image, Matrix), S), moment),
                        Domain.equalsTo(0.0),
                    )
            auxiliaries[(right, left)] = S
    return auxiliaries


def _solve_lazy_sep(
    instance: FacilityInstance,
    lift: FacilityLift,
    *,
    ballconstraints: Path | None,
    mosek_time_limit: float | None,
    lazy_tol: float,
    lazy_max_iters: int,
    lazy_cuts_per_iter: int,
    lazy_cut_strategy: str,
    verbose: bool,
) -> FacilityResult:
    """Solve Full SEP by adding violated skew-skew equations lazily."""

    from mosek.fusion import Domain, Expr, Matrix, SolutionType

    start = time.perf_counter()
    sdp, _ = _build_base_model(
        lift,
        method="RLT",
        ballconstraints=ballconstraints,
        mosek_time_limit=mosek_time_limit,
    )
    model = sdp["M"]
    H = sdp["X"]
    if verbose:
        import sys

        model.setLogHandler(sys.stdout)
    auxiliaries = _add_relaxed_sep_pairs(model, H, lift)
    n_cuts = 0
    max_relative_violation = math.inf
    rounds = 0
    status = "Unknown"
    primal = dual = None

    for iteration in range(lazy_max_iters + 1):
        rounds = iteration + 1
        model.solve()
        status = model.getProblemStatus(SolutionType.Default).name
        if status != "PrimalAndDualFeasible":
            break
        primal = float(model.primalObjValue())
        dual = float(model.dualObjValue())

        candidates = []
        max_relative_violation = 0.0
        for key, S in auxiliaries.items():
            Sstar = np.asarray(S.level(), dtype=float).reshape(9, 9)
            scale = max(1.0, float(np.linalg.norm(Sstar)))
            maximum, _, cuts = separate_skew_skew(
                Sstar,
                3,
                3,
                ncuts=9,
                strategy=lazy_cut_strategy,
            )
            max_relative_violation = max(max_relative_violation, maximum / scale)
            for violation, A, B in cuts:
                if violation / scale > lazy_tol:
                    candidates.append((violation / scale, key, A, B))

        if max_relative_violation <= lazy_tol or iteration == lazy_max_iters:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:lazy_cuts_per_iter]
        if not selected:
            break
        grouped = {}
        for _, key, A, B in selected:
            grouped.setdefault(key, []).append((A, B))
        for key, cuts in grouped.items():
            model.constraint(
                f"lazy_skew_{iteration}_{key[0]}_{key[1]}",
                Expr.mul(cut_batch_matrix(cuts, Matrix), Expr.flatten(auxiliaries[key])),
                Domain.equalsTo(0.0),
            )
        n_cuts += len(selected)

    theta_bound = None if dual is None else -dual
    rel_gap = math.inf
    eval_ratio = -math.inf
    if primal is not None and dual is not None:
        rel_gap = abs(primal - dual) / max(1.0, abs(primal), abs(dual))
        Hstar = np.asarray(H.level(), dtype=float).reshape(5, 5)
        eigvals = np.linalg.eigvalsh(0.5 * (Hstar + Hstar.T))[::-1]
        eval_ratio = float(eigvals[0] / max(abs(eigvals[1]), 1e-16))
    model.dispose()
    return FacilityResult(
        method="Lazy SEP",
        status=status,
        instance=instance.name,
        num_sites=lift.num_sites,
        theta_bound=theta_bound,
        distance_bound=theta_bound,
        runtime=time.perf_counter() - start,
        extra={
            "formulation": "unit-sphere",
            "theta_semantics": "distance",
            "squared_distance_bound": (
                None if theta_bound is None else max(0.0, theta_bound) ** 2
            ),
            "rel_gap": rel_gap,
            "eval_ratio": eval_ratio,
            "soc_pair_blocks": len(auxiliaries),
            "sep_pair_blocks": len(auxiliaries),
            "num_distance_blocks": lift.num_sites,
            "num_hull_facets": lift.num_hull_facets,
            "num_cone_blocks": len(lift.standard["K_dims"]),
            "num_nonnegative_rows": lift.num_hull_facets,
            "radius_squared": 1.0,
            "lazy_cut_strategy": lazy_cut_strategy,
            "n_cuts": n_cuts,
            "n_rounds": rounds,
            "max_skew_violation": max_relative_violation,
        },
    )


def solve_facility_relaxation(
    instance: FacilityInstance,
    *,
    method: str,
    ballconstraints: Path | None = None,
    mosek_time_limit: float | None = None,
    lazy_tol: float = 1e-7,
    lazy_max_iters: int = 100,
    lazy_cuts_per_iter: int = 50,
    lazy_cut_strategy: str = "coordinate",
    verbose: bool = False,
) -> FacilityResult:
    """Solve one of the relaxation tiers in the unit-sphere model."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    lift = build_facility_lift(instance)
    if method == "Lazy SEP":
        return _solve_lazy_sep(
            instance,
            lift,
            ballconstraints=ballconstraints,
            mosek_time_limit=mosek_time_limit,
            lazy_tol=lazy_tol,
            lazy_max_iters=lazy_max_iters,
            lazy_cuts_per_iter=lazy_cuts_per_iter,
            lazy_cut_strategy=lazy_cut_strategy,
            verbose=verbose,
        )

    start = time.perf_counter()
    sdp, solve_sdp_standard_form = _build_base_model(
        lift,
        method=method,
        ballconstraints=ballconstraints,
        mosek_time_limit=mosek_time_limit,
    )
    sep_pair_blocks = 0
    if method == "Full SEP":
        sep_pair_blocks = add_full_sep_pair_products(sdp["M"], sdp["X"], lift.standard)
    result = solve_sdp_standard_form(lift.standard, sdp)
    return _result_from_standard_solve(
        instance,
        lift,
        method,
        result,
        time.perf_counter() - start,
        sep_pair_blocks=sep_pair_blocks,
    )


def rank_one_diagnostics(
    instance: FacilityInstance, x: np.ndarray, theta: float
) -> dict[str, float]:
    """Check the affine cone identities on one feasible rank-one point."""

    x = np.asarray(x, dtype=float).reshape(2)
    norm_sq = float(x @ x)
    if norm_sq > 1.0 + 1e-10:
        raise ValueError("x lies outside the unit disk")
    sigma = math.sqrt(max(0.0, 1.0 - norm_sq))
    w = np.array([1.0, x[0], x[1], theta, sigma])
    H = np.outer(w, w)
    lift = build_facility_lift(instance)
    cone_margins = []
    identity_errors = []
    J4 = np.diag([1.0, -1.0, -1.0, -1.0])
    for index, cone_map in enumerate(lift.cone_maps):
        z = cone_map @ w
        cone_margins.append(float(z[0] - np.linalg.norm(z[1:])))
        if index:
            direct = float(np.sum((x - instance.points[index - 1]) ** 2) - theta**2)
            identity_errors.append(abs(float(z @ J4 @ z) - direct))
    sphere = abs(float(np.vdot(lift.sphere_matrix, H)))
    hull_violation = float(
        np.max(instance.halfspace_A @ x - instance.halfspace_b)
    )
    return {
        "sphere_residual": sphere,
        "max_distance_identity_error": max(identity_errors, default=0.0),
        "minimum_cone_margin": min(cone_margins),
        "maximum_hull_violation": max(0.0, hull_violation),
    }

