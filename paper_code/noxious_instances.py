"""Planar noxious-facility instances (Section 4.3).

Given sites ``p_i in R^2`` the problem maximizes the distance from a location
in their convex hull to the nearest site,

    max theta  s.t.  ||x - p_i|| >= theta (i = 1..m),  x in conv{p_1..p_m}.

This module holds the instance container, the hull geometry, the three
generators used in the paper (regular ``m``-gons, random points in the unit
disk normalized by their minimum enclosing disk, and the fixed four-point
instance of eq. (targeted)), and the exact planar reference value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class FacilityInstance:
    """Planar sites plus the halfspace description of their convex hull."""

    name: str
    points: np.ndarray
    hull_vertices: np.ndarray
    halfspace_A: np.ndarray
    halfspace_b: np.ndarray
    theta_upper: float



def hull_halfspaces(points: np.ndarray, *, tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized inequalities A x <= b for the planar convex hull."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (m, 2)")
    if pts.shape[0] < 3:
        raise ValueError("at least three planar points are required")

    hull = ConvexHull(pts)
    vertices = pts[hull.vertices]
    centroid = np.mean(vertices, axis=0)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i, p in enumerate(vertices):
        q = vertices[(i + 1) % len(vertices)]
        edge = q - p
        normal = np.array([edge[1], -edge[0]], dtype=float)
        bound = float(normal @ p)
        if normal @ centroid > bound + tolerance:
            normal = -normal
            bound = -bound
        scale = float(np.linalg.norm(normal))
        if scale <= tolerance:
            raise ValueError("degenerate hull edge")
        rows.append(normal / scale)
        rhs.append(bound / scale)
    return np.vstack(rows), np.asarray(rhs), vertices


def regular_polygon(num_sites: int, *, radius: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate sites equally spaced on a circle."""

    angles = phase + 2.0 * math.pi * np.arange(num_sites) / num_sites
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def random_disk_points(num_sites: int, *, seed: int, radius: float = 1.0) -> np.ndarray:
    """Generate random sites uniformly in a disk."""

    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=num_sites)
    radii = radius * np.sqrt(rng.uniform(0.0, 1.0, size=num_sites))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def targeted_seed_410_points() -> np.ndarray:
    """Return the archived m=4 instance on which legacy Full SEP helped most."""

    return np.array(
        [
            [-0.6188988868092025, 0.1752460828966060],
            [0.6115320348460802, -0.5806915227354644],
            [0.5533450603521616, -0.7177653857874852],
            [-0.6230854840531512, 0.1902842181941680],
        ]
    )


def minimum_enclosing_disk(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the exact planar minimum enclosing disk by finite enumeration."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError("points must have shape (m, 2) with m positive")

    best_center = pts[0].copy()
    best_radius = math.inf

    def consider(center: np.ndarray) -> None:
        nonlocal best_center, best_radius
        distances = np.linalg.norm(pts - center, axis=1)
        radius = float(np.max(distances))
        tolerance = 1e-10 * max(1.0, radius)
        if radius < best_radius and np.all(distances <= radius + tolerance):
            best_center = center.copy()
            best_radius = radius

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            center = 0.5 * (pts[i] + pts[j])
            radius = 0.5 * float(np.linalg.norm(pts[i] - pts[j]))
            distances = np.linalg.norm(pts - center, axis=1)
            if np.all(distances <= radius + 1e-10 * max(1.0, radius)):
                consider(center)

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                matrix = 2.0 * np.vstack((pts[j] - pts[i], pts[k] - pts[i]))
                if abs(float(np.linalg.det(matrix))) <= 1e-12:
                    continue
                rhs = np.array(
                    [
                        pts[j] @ pts[j] - pts[i] @ pts[i],
                        pts[k] @ pts[k] - pts[i] @ pts[i],
                    ]
                )
                center = np.linalg.solve(matrix, rhs)
                radius = float(np.linalg.norm(center - pts[i]))
                distances = np.linalg.norm(pts - center, axis=1)
                if np.all(distances <= radius + 1e-10 * max(1.0, radius)):
                    consider(center)

    if not math.isfinite(best_radius):
        # This occurs only when all points coincide.
        consider(pts[0])
    return best_center, best_radius


def normalize_by_minimum_enclosing_disk(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Translate and scale points so their minimum enclosing disk is unit."""

    center, radius = minimum_enclosing_disk(points)
    if radius <= 1e-12:
        raise ValueError("cannot normalize coincident sites")
    return (np.asarray(points, dtype=float) - center) / radius, center, radius


def make_instance(points: np.ndarray, *, name: str = "custom") -> FacilityInstance:
    """Build an instance and simple global bounds from a point cloud."""

    pts = np.asarray(points, dtype=float)
    A, b, vertices = hull_halfspaces(pts)
    diffs = pts[:, None, :] - pts[None, :, :]
    pairwise_sq = np.sum(diffs * diffs, axis=2)
    theta_upper = float(np.max(pairwise_sq))
    if theta_upper <= 0.0:
        raise ValueError("sites must not all coincide")
    return FacilityInstance(
        name=name,
        points=pts,
        hull_vertices=vertices,
        halfspace_A=A,
        halfspace_b=b,
        theta_upper=theta_upper,
    )


def true_value_arrangement(instance: FacilityInstance) -> tuple[float, np.ndarray]:
    """Compute the planar optimum from hull/Voronoi arrangement candidates.

    On each nearest-site Voronoi cell the squared-distance objective is one
    convex quadratic.  Its maximum over the cell intersected with the polygon
    occurs at a vertex, all of which are included in the candidates below.
    """

    points = instance.points
    vertices = instance.hull_vertices
    candidates = [point.copy() for point in vertices]

    bisectors = []
    for i, j in combinations(range(len(points)), 2):
        normal = 2.0 * (points[j] - points[i])
        rhs = float(points[j] @ points[j] - points[i] @ points[i])
        if np.linalg.norm(normal) > 1e-12:
            bisectors.append((i, j, normal, rhs))

    # Intersections of hull edges with pairwise bisectors.
    for start_index, start in enumerate(vertices):
        end = vertices[(start_index + 1) % len(vertices)]
        direction = end - start
        for _, _, normal, rhs in bisectors:
            denominator = float(normal @ direction)
            if abs(denominator) <= 1e-12:
                continue
            fraction = (rhs - float(normal @ start)) / denominator
            if -1e-10 <= fraction <= 1.0 + 1e-10:
                candidates.append(start + np.clip(fraction, 0.0, 1.0) * direction)

    # Voronoi vertices are circumcenters of triples of sites.
    for i, j, k in combinations(range(len(points)), 3):
        matrix = 2.0 * np.vstack((points[j] - points[i], points[k] - points[i]))
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            continue
        rhs = np.array(
            [
                points[j] @ points[j] - points[i] @ points[i],
                points[k] @ points[k] - points[i] @ points[i],
            ]
        )
        candidate = np.linalg.solve(matrix, rhs)
        if np.all(
            instance.halfspace_A @ candidate
            <= instance.halfspace_b + 1e-9
        ):
            candidates.append(candidate)

    array = np.vstack(candidates)
    feasible = np.all(
        array @ instance.halfspace_A.T <= instance.halfspace_b + 1e-9,
        axis=1,
    )
    array = array[feasible]
    differences = array[:, None, :] - points[None, :, :]
    squared_values = np.min(np.sum(differences * differences, axis=2), axis=1)
    best = int(np.argmax(squared_values))
    return math.sqrt(max(0.0, float(squared_values[best]))), array[best].copy()


def true_value_grid(instance: FacilityInstance, *, grid: int = 401) -> tuple[float, np.ndarray]:
    """Compute a crude feasible lower bound by gridding the hull bounding box."""

    pts = instance.points
    lo = np.min(instance.hull_vertices, axis=0)
    hi = np.max(instance.hull_vertices, axis=0)
    best_theta = -math.inf
    best_x = np.zeros(2)
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    for x_coord in xs:
        candidates = np.column_stack([np.full(grid, x_coord), ys])
        feasible = np.all(candidates @ instance.halfspace_A.T <= instance.halfspace_b + 1e-12, axis=1)
        if not np.any(feasible):
            continue
        diff = candidates[feasible, None, :] - pts[None, :, :]
        theta_vals = np.min(np.sum(diff * diff, axis=2), axis=1)
        idx = int(np.argmax(theta_vals))
        if theta_vals[idx] > best_theta:
            best_theta = float(theta_vals[idx])
            best_x = candidates[feasible][idx]
    return best_theta, best_x


