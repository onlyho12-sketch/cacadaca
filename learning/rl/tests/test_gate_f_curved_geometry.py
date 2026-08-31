"""CPU-only tests for the isolated Gate F macro-geometry helpers."""
from __future__ import annotations

import numpy as np
import pytest

from learning.rl.gate_f_curved_geometry import (
    SurfaceGeometrySpec,
    biased_normal_command,
    build_surface_mesh,
    collected_link6_target_quaternion_xyzw,
    normal_alignment_quaternion_wxyz,
    project_normal_force_n,
    quaternion_step_angle_deg,
    surface_height_normal,
    validate_surface_mesh,
    vertical_pad_target_z_m,
)


def test_flat_height_and_normal_are_exact() -> None:
    height, normal = surface_height_normal("flat", 1.0, (0.32, 0.32), 0.03, 0.29)
    assert height == 0.0
    np.testing.assert_array_equal(normal, np.array([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_center_is_zero_with_up_normal(kind: str) -> None:
    height, normal = surface_height_normal(kind, 0.60, (0.32, 0.32), 0.16, 0.16)
    assert height == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-14)


def test_cylinder_sagitta_matches_closed_form() -> None:
    height, normal = surface_height_normal("cylinder", 0.60, (0.32, 0.32), 0.0, 0.16)
    expected = np.sqrt(0.60**2 - 0.16**2) - 0.60
    assert height == pytest.approx(expected, abs=1e-14)
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-14)
    assert normal[0] < 0.0


def test_sphere_changes_in_both_tangent_directions() -> None:
    height_u, normal_u = surface_height_normal("sphere", 0.60, (0.32, 0.32), 0.0, 0.16)
    height_v, normal_v = surface_height_normal("sphere", 0.60, (0.32, 0.32), 0.16, 0.0)
    assert height_u == pytest.approx(height_v, abs=1e-14)
    assert abs(normal_u[0]) > 0.0 and normal_u[1] == pytest.approx(0.0)
    assert abs(normal_v[1]) > 0.0 and normal_v[0] == pytest.approx(0.0)


def test_large_radius_converges_to_flat() -> None:
    height, normal = surface_height_normal("sphere", 1.0e8, (0.32, 0.32), 0.0, 0.0)
    assert abs(height) < 1.0e-8
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=2.0e-9)


def test_freeform_normal_matches_numeric_gradient() -> None:
    point = np.array([0.13, 0.19])
    eps = 1.0e-6
    h0, normal = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), point[0], point[1], freeform_seed=5
    )
    hu, _ = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), point[0] + eps, point[1], freeform_seed=5
    )
    hv, _ = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), point[0], point[1] + eps, freeform_seed=5
    )
    numeric = np.array([-(hu - h0) / eps, -(hv - h0) / eps, 1.0])
    numeric /= np.linalg.norm(numeric)
    np.testing.assert_allclose(normal, numeric, atol=2.0e-5)


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_analytic_curved_normal_matches_numeric_height_gradient(kind: str) -> None:
    point = np.array([0.09, 0.21])
    eps = 1.0e-6
    h0, normal = surface_height_normal(kind, 0.60, (0.32, 0.32), point[0], point[1])
    hu, _ = surface_height_normal(kind, 0.60, (0.32, 0.32), point[0] + eps, point[1])
    hv, _ = surface_height_normal(kind, 0.60, (0.32, 0.32), point[0], point[1] + eps)
    numeric = np.array([-(hu - h0) / eps, -(hv - h0) / eps, 1.0])
    numeric /= np.linalg.norm(numeric)
    np.testing.assert_allclose(normal, numeric, atol=2.0e-6)


def test_freeform_seed_is_reproducible_and_distinct() -> None:
    u = np.linspace(0.0, 0.32, 9)
    v = np.linspace(0.0, 0.32, 9)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    first, normal_first = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), uu, vv, freeform_seed=7
    )
    repeat, normal_repeat = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), uu, vv, freeform_seed=7
    )
    different, _ = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), uu, vv, freeform_seed=8
    )
    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(normal_first, normal_repeat)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("kind", ["flat", "cylinder", "sphere", "freeform"])
def test_mesh_is_finite_unit_normal_and_upward_wound(kind: str) -> None:
    mesh = build_surface_mesh(
        SurfaceGeometrySpec(kind, (0.32, 0.32), 0.60, 11),
        grid_size=17,
        margin_m=0.01,
    )
    summary = validate_surface_mesh(mesh)
    assert summary["vertex_count"] == 17**2
    assert summary["triangle_count"] == 2 * 16**2
    assert summary["normal_length_max_error"] < 1.0e-10
    assert summary["minimum_face_normal_z"] > 0.0


def test_normal_force_projection_uses_local_normal() -> None:
    force = np.array([[3.0, 4.0, 12.0], [1.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 1.0]])
    projected = project_normal_force_n(force, normal)
    np.testing.assert_allclose(projected, [12.0, 1.0 / np.sqrt(2.0)], atol=1e-14)


@pytest.mark.parametrize(
    "spec",
    [
        SurfaceGeometrySpec("invalid", (0.32, 0.32), 0.60, 0),
        SurfaceGeometrySpec("sphere", (0.32, 0.32), 0.20, 0),
        SurfaceGeometrySpec("cylinder", (-0.32, 0.32), 0.60, 0),
    ],
)
def test_invalid_geometry_is_rejected(spec: SurfaceGeometrySpec) -> None:
    with pytest.raises(ValueError):
        spec.validate()


def test_nonfinite_force_is_rejected() -> None:
    with pytest.raises(ValueError):
        project_normal_force_n(np.array([np.nan, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def test_vertical_tracking_compensation_is_positive_z_feed_forward_only() -> None:
    baseline = vertical_pad_target_z_m(0.40, -0.003, 0.001, 0.0)
    compensated = vertical_pad_target_z_m(0.40, -0.003, 0.001, 0.0005)
    assert baseline == pytest.approx(0.398)
    assert compensated - baseline == pytest.approx(0.0005)


def test_negative_vertical_tracking_compensation_is_rejected() -> None:
    with pytest.raises(ValueError):
        vertical_pad_target_z_m(0.40, 0.0, 0.0, -1.0e-4)


def _quat_apply_numpy(quaternion_wxyz: np.ndarray, vector_xyz: np.ndarray) -> np.ndarray:
    w = quaternion_wxyz[..., :1]
    qv = quaternion_wxyz[..., 1:]
    vector = np.broadcast_to(vector_xyz, qv.shape)
    return vector + 2.0 * (
        w * np.cross(qv, vector) + np.cross(qv, np.cross(qv, vector))
    )


def test_flat_normal_alignment_is_exact_identity() -> None:
    quat = normal_alignment_quaternion_wxyz(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(quat, np.array([1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("kind", ["cylinder", "sphere", "freeform"])
def test_normal_alignment_rotates_positive_z_onto_surface_normal(kind: str) -> None:
    _, normal = surface_height_normal(
        kind, 0.60, (0.32, 0.32), 0.07, 0.24, freeform_seed=0
    )
    quat = normal_alignment_quaternion_wxyz(normal)
    rotated = _quat_apply_numpy(quat, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(rotated, normal, atol=2.0e-14)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1.0e-14)
    assert quat[0] > 0.0


def test_normal_alignment_quaternion_is_continuous_along_freeform_path() -> None:
    u = np.linspace(0.04, 0.28, 501)
    v = np.full_like(u, 0.18)
    _, normal = surface_height_normal(
        "freeform", 0.60, (0.32, 0.32), u, v, freeform_seed=0
    )
    quat = normal_alignment_quaternion_wxyz(normal)
    steps = quaternion_step_angle_deg(quat)
    assert np.isfinite(quat).all()
    assert np.all(quat[:, 0] > 0.0)
    assert steps.max() < 0.1


def test_collected_link6_command_preserves_flat_and_aligns_local_minus_z() -> None:
    normals = np.array([[0.0, 0.0, 1.0], [0.2, -0.1, np.sqrt(0.95)]])
    command = collected_link6_target_quaternion_xyzw(normals)
    np.testing.assert_array_equal(command[0], np.array([1.0, 0.0, 0.0, 0.0]))

    xyz = command[:, :3]
    w = command[:, 3:4]
    local_minus_z = np.broadcast_to(np.array([0.0, 0.0, -1.0]), (2, 3))
    twice_cross = 2.0 * np.cross(xyz, local_minus_z)
    rotated = local_minus_z + w * twice_cross + np.cross(xyz, twice_cross)
    np.testing.assert_allclose(rotated, normals, atol=1.0e-14)


def test_biased_normal_command_adds_frozen_world_tangent_and_normalizes() -> None:
    normal = np.array([[0.0, 0.0, 1.0], [0.1, -0.05, np.sqrt(0.9875)]])
    biased = biased_normal_command(normal, (-0.35, -0.03))
    assert np.isfinite(biased).all()
    np.testing.assert_allclose(np.linalg.norm(biased, axis=-1), 1.0, atol=1.0e-14)
    assert biased[0, 0] < 0.0
    assert biased[0, 1] < 0.0
