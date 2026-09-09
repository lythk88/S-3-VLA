"""Fit and select conservative geometric primitives for fused obstacle clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class PrimitiveFit:
    kind: str
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    score: float
    surface_rmse: float
    candidate_scores: dict[str, float]


def primitive_bounding_box_half_extents(kind: str, size: np.ndarray) -> np.ndarray:
    """Return local OBB half-extents that conservatively contain a primitive."""
    size = np.asarray(size, dtype=np.float64)
    if kind in {"aabb", "obb", "ellipsoid"}:
        if size.shape != (3,):
            raise ValueError(f"{kind} size must contain three values")
        return size.copy()
    if kind == "sphere":
        if size.shape != (1,):
            raise ValueError("sphere size must contain one radius")
        return np.repeat(size[0], 3)
    if kind in {"cylinder", "capsule"}:
        if size.shape != (2,):
            raise ValueError(f"{kind} size must contain radius and half-length")
        radius, half_length = size
        axial_extent = half_length + radius if kind == "capsule" else half_length
        return np.asarray([radius, radius, axial_extent], dtype=np.float64)
    raise ValueError(f"unsupported primitive kind: {kind!r}")


def _to_world(local_points: np.ndarray, fit: PrimitiveFit) -> np.ndarray:
    return local_points @ fit.rotation.T + fit.center


def plot_primitive_fit(
    points: np.ndarray,
    fit: PrimitiveFit,
    save_path: str | Path,
    *,
    maximum_display_points: int = 12000,
) -> None:
    """Save a 3D view of an obstacle point cloud and its selected primitive."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("plot points must have shape (N, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError("plot requires at least one finite point")
    if len(points) > maximum_display_points:
        indices = np.linspace(0, len(points) - 1, maximum_display_points, dtype=np.int64)
        display_points = points[indices]
    else:
        display_points = points

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        s=2.5,
        c="#4b5563",
        alpha=0.45,
        depthshade=False,
    )

    shape_color = "#1687d9"
    rendered_points = []
    if fit.kind == "ellipsoid":
        radii = np.asarray(fit.size, dtype=np.float64)
        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        phi = np.linspace(0.0, np.pi, 40)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        ellipsoid_local = np.column_stack(
            (
                (radii[0] * np.sin(phi_grid) * np.cos(theta_grid)).ravel(),
                (radii[1] * np.sin(phi_grid) * np.sin(theta_grid)).ravel(),
                (radii[2] * np.cos(phi_grid)).ravel(),
            )
        )
        ellipsoid_world = _to_world(ellipsoid_local, fit).reshape(*theta_grid.shape, 3)
        axis.plot_surface(
            ellipsoid_world[..., 0],
            ellipsoid_world[..., 1],
            ellipsoid_world[..., 2],
            color=shape_color,
            alpha=0.25,
            linewidth=0,
            antialiased=True,
        )
        rendered_points.append(ellipsoid_world.reshape(-1, 3))
    elif fit.kind in {"obb", "aabb"}:
        half_extents = np.asarray(fit.size, dtype=np.float64)
        signs = np.asarray(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        vertices = _to_world(signs * half_extents, fit)
        face_indices = (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
        faces = [[vertices[index] for index in face] for face in face_indices]
        axis.add_collection3d(
            Poly3DCollection(
                faces,
                facecolor=shape_color,
                edgecolor="#005a9c",
                linewidth=1.2,
                alpha=0.25,
            )
        )
        rendered_points.append(vertices)
    elif fit.kind in {"cylinder", "capsule"}:
        radius, half_length = np.asarray(fit.size, dtype=np.float64)
        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        axial = np.linspace(-half_length, half_length, 25)
        theta_grid, axial_grid = np.meshgrid(theta, axial)
        cylinder_local = np.column_stack(
            (
                (radius * np.cos(theta_grid)).ravel(),
                (radius * np.sin(theta_grid)).ravel(),
                axial_grid.ravel(),
            )
        )
        cylinder_world = _to_world(cylinder_local, fit).reshape(*theta_grid.shape, 3)
        axis.plot_surface(
            cylinder_world[..., 0],
            cylinder_world[..., 1],
            cylinder_world[..., 2],
            color=shape_color,
            alpha=0.25,
            linewidth=0,
            antialiased=True,
        )
        rendered_points.append(cylinder_world.reshape(-1, 3))

        if fit.kind == "cylinder":
            radial = np.linspace(0.0, radius, 20)
            theta_cap, radial_grid = np.meshgrid(theta, radial)
            for end in (-half_length, half_length):
                cap_local = np.column_stack(
                    (
                        (radial_grid * np.cos(theta_cap)).ravel(),
                        (radial_grid * np.sin(theta_cap)).ravel(),
                        np.full(theta_cap.size, end),
                    )
                )
                cap_world = _to_world(cap_local, fit).reshape(*theta_cap.shape, 3)
                axis.plot_surface(
                    cap_world[..., 0],
                    cap_world[..., 1],
                    cap_world[..., 2],
                    color=shape_color,
                    alpha=0.25,
                    linewidth=0,
                )
                rendered_points.append(cap_world.reshape(-1, 3))
        else:
            phi = np.linspace(0.0, 0.5 * np.pi, 28)
            theta_dome, phi_grid = np.meshgrid(theta, phi)
            for direction in (-1.0, 1.0):
                dome_local = np.column_stack(
                    (
                        (radius * np.sin(phi_grid) * np.cos(theta_dome)).ravel(),
                        (radius * np.sin(phi_grid) * np.sin(theta_dome)).ravel(),
                        (
                            direction
                            * (half_length + radius * np.cos(phi_grid))
                        ).ravel(),
                    )
                )
                dome_world = _to_world(dome_local, fit).reshape(*theta_dome.shape, 3)
                axis.plot_surface(
                    dome_world[..., 0],
                    dome_world[..., 1],
                    dome_world[..., 2],
                    color=shape_color,
                    alpha=0.25,
                    linewidth=0,
                    antialiased=True,
                )
                rendered_points.append(dome_world.reshape(-1, 3))
    elif fit.kind == "sphere":
        radius = float(np.asarray(fit.size, dtype=np.float64)[0])
        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        phi = np.linspace(0.0, np.pi, 40)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        sphere_local = np.column_stack(
            (
                (radius * np.sin(phi_grid) * np.cos(theta_grid)).ravel(),
                (radius * np.sin(phi_grid) * np.sin(theta_grid)).ravel(),
                (radius * np.cos(phi_grid)).ravel(),
            )
        )
        sphere_world = _to_world(sphere_local, fit).reshape(*theta_grid.shape, 3)
        axis.plot_surface(
            sphere_world[..., 0],
            sphere_world[..., 1],
            sphere_world[..., 2],
            color=shape_color,
            alpha=0.25,
            linewidth=0,
            antialiased=True,
        )
        rendered_points.append(sphere_world.reshape(-1, 3))
    else:
        raise ValueError(f"unsupported primitive kind for plotting: {fit.kind}")

    bounds_points = np.vstack([display_points, *rendered_points])
    lower = np.min(bounds_points, axis=0)
    upper = np.max(bounds_points, axis=0)
    center = 0.5 * (lower + upper)
    half_span = max(0.5 * float(np.max(upper - lower)), 1e-3) * 1.08
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")
    axis.set_title(
        f"Obstacle point cloud + selected {fit.kind.upper()}\n"
        f"Surface RMSE: {fit.surface_rmse:.4f} m"
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4b5563",
                markersize=6,
                label=f"Filtered point cloud ({len(points):,} points)",
            ),
            Patch(facecolor=shape_color, edgecolor="#005a9c", alpha=0.25, label=fit.kind.upper()),
        ],
        loc="upper right",
    )
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    rotation = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    return center, rotation


def _obb_distance(local: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    offset = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
    inside = np.minimum(np.max(offset, axis=1), 0.0)
    return np.abs(outside + inside)


def _ellipsoid_distance(local: np.ndarray, radii: np.ndarray) -> np.ndarray:
    radial_norm = np.linalg.norm(local / radii, axis=1)
    point_norm = np.linalg.norm(local, axis=1)
    safe_norm = np.maximum(radial_norm, 1e-12)
    return np.abs(1.0 - 1.0 / safe_norm) * point_norm


def _cylinder_distance(local: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    offset = np.column_stack(
        (np.linalg.norm(local[:, :2], axis=1) - radius, np.abs(local[:, 2]) - half_length)
    )
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
    inside = np.minimum(np.max(offset, axis=1), 0.0)
    return np.abs(outside + inside)


def _capsule_distance(local: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    axial_excess = np.maximum(np.abs(local[:, 2]) - half_length, 0.0)
    return np.abs(np.sqrt(np.sum(np.square(local[:, :2]), axis=1) + axial_excess**2) - radius)


def _sphere_distance(local: np.ndarray, radius: float) -> np.ndarray:
    return np.abs(np.linalg.norm(local, axis=1) - radius)


def _score(distances: np.ndarray, scale: float, parameters: int) -> tuple[float, float]:
    cutoff = np.quantile(distances, 0.95)
    trimmed = distances[distances <= cutoff]
    rmse = float(np.sqrt(np.mean(np.square(trimmed))))
    normalized_mse = (rmse / max(scale, 1e-6)) ** 2
    bic_per_point = parameters * np.log(max(len(trimmed), 2)) / max(len(trimmed), 1)
    return float(np.log(normalized_mse + 1e-12) + bic_per_point), rmse


def _axis_rotation(pca_rotation: np.ndarray, axis_index: int) -> np.ndarray:
    remaining = [index for index in range(3) if index != axis_index]
    rotation = np.column_stack(
        (pca_rotation[:, remaining[0]], pca_rotation[:, remaining[1]], pca_rotation[:, axis_index])
    )
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 1] *= -1.0
    return rotation


def _fit_obb(points: np.ndarray, rotation: np.ndarray, padding: float):
    local = (points - np.mean(points, axis=0)) @ rotation
    lower, upper = np.min(local, axis=0), np.max(local, axis=0)
    local_center = 0.5 * (lower + upper)
    center = np.mean(points, axis=0) + rotation @ local_center
    half_extents = np.maximum(0.5 * (upper - lower) + padding, padding)
    centered = (points - center) @ rotation
    return center, rotation, half_extents, _obb_distance(centered, half_extents)


def _fit_ellipsoid(points: np.ndarray, padding: float):
    """Fit the same minimum-volume enclosing ellipsoid used by VLSA."""
    # Keep one canonical MVEE implementation: this is the exact fitter used by
    # the original VLSA path, rather than the older PCA/std approximation.
    from utils import fit_ellipse

    center, rotation, radii = fit_ellipse(points, plot=False)
    radii = np.maximum(np.asarray(radii, dtype=np.float64) + padding, padding)
    local = (points - center) @ rotation
    return center, rotation, radii, _ellipsoid_distance(local, radii)


def _fit_cylinder(points: np.ndarray, rotation: np.ndarray, padding: float):
    origin = np.mean(points, axis=0)
    local = (points - origin) @ rotation
    axial_center = 0.5 * (np.min(local[:, 2]) + np.max(local[:, 2]))
    local[:, 2] -= axial_center
    center = origin + rotation[:, 2] * axial_center
    radius = float(np.max(np.linalg.norm(local[:, :2], axis=1)) + padding)
    half_length = float(np.max(np.abs(local[:, 2])) + padding)
    size = np.asarray([radius, half_length], dtype=np.float64)
    return center, rotation, size, _cylinder_distance(local, radius, half_length)


def _fit_capsule(points: np.ndarray, rotation: np.ndarray, padding: float):
    origin = np.mean(points, axis=0)
    local = (points - origin) @ rotation
    axial_center = 0.5 * (np.min(local[:, 2]) + np.max(local[:, 2]))
    local[:, 2] -= axial_center
    center = origin + rotation[:, 2] * axial_center
    radial_squared = np.sum(np.square(local[:, :2]), axis=1)
    axial = np.abs(local[:, 2])
    maximum_axial = float(np.max(axial))

    def radius_for(half_length: float) -> float:
        return float(
            np.max(np.sqrt(radial_squared + np.square(np.maximum(axial - half_length, 0.0))))
            + padding
        )

    def volume(half_length: float) -> float:
        radius = radius_for(half_length)
        return float(2.0 * np.pi * radius**2 * half_length + 4.0 * np.pi * radius**3 / 3.0)

    result = minimize_scalar(volume, bounds=(0.0, maximum_axial), method="bounded")
    half_length = float(result.x)
    radius = radius_for(half_length)
    size = np.asarray([radius, half_length], dtype=np.float64)
    return center, rotation, size, _capsule_distance(local, radius, half_length)


def _fit_sphere(points: np.ndarray, padding: float):
    """Fit a deterministic conservative bounding sphere using Ritter expansion."""
    seed = points[0]
    first = points[np.argmax(np.linalg.norm(points - seed, axis=1))]
    second = points[np.argmax(np.linalg.norm(points - first, axis=1))]
    center = 0.5 * (first + second)
    radius = 0.5 * float(np.linalg.norm(second - first))
    for point in points:
        delta = point - center
        distance = float(np.linalg.norm(delta))
        if distance > radius:
            expanded = 0.5 * (radius + distance)
            center = center + ((expanded - radius) / max(distance, 1e-12)) * delta
            radius = expanded
    # Guard against floating-point drift and make the primitive enclosing.
    radius = float(np.max(np.linalg.norm(points - center, axis=1)) + padding)
    size = np.asarray([radius], dtype=np.float64)
    return center, np.eye(3, dtype=np.float64), size, _sphere_distance(points - center, radius)


def fit_primitive_candidates(
    points: np.ndarray,
    *,
    padding: float = 0.005,
    maximum_score_points: int = 10000,
    allowed_kinds: tuple[str, ...] = ("obb", "cylinder", "capsule"),
) -> dict[str, PrimitiveFit]:
    """Fit one best conservative candidate for each requested primitive kind."""
    allowed = frozenset(allowed_kinds)
    supported = {"obb", "aabb", "ellipsoid", "cylinder", "capsule", "sphere"}
    if not allowed or not allowed <= supported:
        raise ValueError(f"allowed_kinds must be a non-empty subset of {sorted(supported)}")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16:
        raise ValueError("primitive fitting requires at least 16 finite 3D points")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 16:
        raise ValueError("primitive fitting requires at least 16 finite 3D points")
    center, pca_rotation = _pca(points)
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if scale < 1e-5:
        raise ValueError("point cloud is degenerate")
    if len(points) > maximum_score_points:
        score_indices = np.linspace(0, len(points) - 1, maximum_score_points, dtype=np.int64)
    else:
        score_indices = np.arange(len(points))

    candidates = []
    if "obb" in allowed:
        obb = _fit_obb(points, pca_rotation, padding)
        candidates.append(("obb", *obb[:3], obb[3][score_indices], 9))
    if "aabb" in allowed:
        aabb = _fit_obb(points, np.eye(3, dtype=np.float64), padding)
        candidates.append(("aabb", *aabb[:3], aabb[3][score_indices], 6))
    if "ellipsoid" in allowed:
        ellipsoid = _fit_ellipsoid(points, padding)
        candidates.append(("ellipsoid", *ellipsoid[:3], ellipsoid[3][score_indices], 9))
    if "sphere" in allowed:
        sphere = _fit_sphere(points, padding)
        candidates.append(("sphere", *sphere[:3], sphere[3][score_indices], 4))
    for axis_index in range(3):
        rotation = _axis_rotation(pca_rotation, axis_index)
        if "cylinder" in allowed:
            cylinder = _fit_cylinder(points, rotation, padding)
            candidates.append(("cylinder", *cylinder[:3], cylinder[3][score_indices], 7))
        if "capsule" in allowed:
            capsule = _fit_capsule(points, rotation, padding)
            candidates.append(("capsule", *capsule[:3], capsule[3][score_indices], 7))

    scored = []
    for kind, candidate_center, rotation, size, distances, parameters in candidates:
        score, rmse = _score(distances, scale, parameters)
        scored.append((score, rmse, kind, candidate_center, rotation, size))
    candidate_scores = {}
    for score, _, kind, *_ in scored:
        candidate_scores[kind] = min(score, candidate_scores.get(kind, np.inf))
    shared_scores = {
        key: float(value) for key, value in sorted(candidate_scores.items())
    }
    best_by_kind = {}
    for score, rmse, kind, candidate_center, rotation, size in sorted(scored):
        if kind in best_by_kind:
            continue
        best_by_kind[kind] = PrimitiveFit(
            kind=kind,
            center=np.asarray(candidate_center, dtype=np.float64),
            rotation=np.asarray(rotation, dtype=np.float64),
            size=np.asarray(size, dtype=np.float64),
            score=float(score),
            surface_rmse=float(rmse),
            candidate_scores=shared_scores,
        )
    return {kind: best_by_kind[kind] for kind in allowed_kinds}


def fit_best_primitive(
    points: np.ndarray,
    *,
    padding: float = 0.005,
    maximum_score_points: int = 10000,
    allowed_kinds: tuple[str, ...] = ("obb", "cylinder", "capsule"),
) -> PrimitiveFit:
    """Fit the allowed conservative primitives and select the best surface model."""
    candidates = fit_primitive_candidates(
        points,
        padding=padding,
        maximum_score_points=maximum_score_points,
        allowed_kinds=allowed_kinds,
    )
    return min(candidates.values(), key=lambda fit: fit.score)
