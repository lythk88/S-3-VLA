import numpy as np
import pytest

from primitive_fitting import (
    PrimitiveFit,
    fit_best_primitive,
    fit_primitive_candidates,
    plot_primitive_fit,
    primitive_bounding_box_half_extents,
)


def _box_surface() -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 13)
    points = []
    for axis in range(3):
        for side in (-1.0, 1.0):
            first, second = np.meshgrid(grid, grid, indexing="ij")
            face = np.zeros((grid.size, grid.size, 3))
            face[..., axis] = side
            remaining = [index for index in range(3) if index != axis]
            face[..., remaining[0]] = first
            face[..., remaining[1]] = second
            points.append(face.reshape(-1, 3))
    return np.vstack(points) * np.array([0.12, 0.07, 0.04])


def _cylinder_surface() -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    height = np.linspace(-0.13, 0.13, 25)
    angle_grid, height_grid = np.meshgrid(angle, height, indexing="ij")
    side = np.column_stack(
        (
            0.05 * np.cos(angle_grid).ravel(),
            0.05 * np.sin(angle_grid).ravel(),
            height_grid.ravel(),
        )
    )
    return side


def test_selector_uses_only_the_requested_three_shapes():
    fit = fit_best_primitive(_box_surface())
    assert fit.kind in {"obb", "cylinder", "capsule"}
    assert set(fit.candidate_scores) == {"obb", "cylinder", "capsule"}
    assert "ellipsoid" not in fit.candidate_scores


def test_selector_recognizes_box_and_cylinder_surfaces():
    assert fit_best_primitive(_box_surface(), padding=1e-5).kind == "obb"
    assert fit_best_primitive(_cylinder_surface(), padding=1e-5).kind == "cylinder"


def test_selector_supports_ellipsoid_kind_and_encloses_points():
    points = _box_surface()
    fit = fit_best_primitive(points, padding=0.0, allowed_kinds=("ellipsoid",))
    assert fit.kind == "ellipsoid"
    local = (points - fit.center) @ fit.rotation
    assert np.max(np.linalg.norm(local / fit.size, axis=1)) <= 1.0 + 1e-12


def test_selector_supports_requested_four_shape_set():
    fit = fit_best_primitive(
        _box_surface(),
        padding=0.0,
        allowed_kinds=("aabb", "ellipsoid", "cylinder", "sphere"),
    )
    assert fit.kind in {"aabb", "ellipsoid", "cylinder", "sphere"}
    assert set(fit.candidate_scores) == {"aabb", "ellipsoid", "cylinder", "sphere"}


def test_candidate_fitter_returns_one_fit_for_every_requested_kind():
    requested = ("aabb", "ellipsoid", "cylinder", "sphere")
    candidates = fit_primitive_candidates(
        _box_surface(), padding=0.0, allowed_kinds=requested
    )
    assert tuple(candidates) == requested
    assert all(fit.kind == kind for kind, fit in candidates.items())
    assert all(set(fit.candidate_scores) == set(requested) for fit in candidates.values())


def test_cylinder_axis_is_world_vertical():
    fit = fit_primitive_candidates(
        _box_surface(), padding=0.0, allowed_kinds=("cylinder",)
    )["cylinder"]
    assert np.allclose(fit.rotation, np.eye(3))
    assert np.allclose(fit.rotation[:, 2], [0.0, 0.0, 1.0])


def test_selector_supports_requested_aabb_cylinder_sphere_set():
    fit = fit_best_primitive(
        _box_surface(),
        padding=0.0,
        allowed_kinds=("aabb", "cylinder", "sphere"),
    )
    assert fit.kind in {"aabb", "cylinder", "sphere"}
    assert set(fit.candidate_scores) == {"aabb", "cylinder", "sphere"}


def test_sphere_fit_conservatively_encloses_points():
    points = _box_surface()
    fit = fit_best_primitive(points, padding=0.0, allowed_kinds=("sphere",))
    assert fit.kind == "sphere"
    assert np.max(np.linalg.norm(points - fit.center, axis=1)) <= fit.size[0] + 1e-12


@pytest.mark.parametrize(
    ("kind", "size", "expected"),
    [
        ("aabb", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
        ("ellipsoid", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
        ("sphere", [0.1], [0.1, 0.1, 0.1]),
        ("cylinder", [0.1, 0.3], [0.1, 0.1, 0.3]),
        ("capsule", [0.1, 0.3], [0.1, 0.1, 0.4]),
    ],
)
def test_primitive_bounding_box_half_extents(kind, size, expected):
    assert np.allclose(
        primitive_bounding_box_half_extents(kind, np.asarray(size)),
        expected,
    )


@pytest.mark.parametrize(
    ("kind", "size"),
    [
        ("obb", [0.12, 0.07, 0.04]),
        ("cylinder", [0.05, 0.13]),
        ("capsule", [0.05, 0.10]),
        ("sphere", [0.12]),
        ("aabb", [0.12, 0.07, 0.04]),
        ("ellipsoid", [0.12, 0.07, 0.04]),
    ],
)
def test_plot_primitive_fit_outputs_all_three_shapes(tmp_path, kind, size):
    fit = PrimitiveFit(
        kind=kind,
        center=np.zeros(3),
        rotation=np.eye(3),
        size=np.asarray(size),
        score=-1.0,
        surface_rmse=0.002,
        candidate_scores={kind: -1.0},
    )
    save_path = tmp_path / f"{kind}.png"
    plot_primitive_fit(_box_surface(), fit, save_path)
    assert save_path.is_file()
    assert save_path.stat().st_size > 10_000
