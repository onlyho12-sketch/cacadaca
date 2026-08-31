"""CPU-only Gate F2 parity checks against the established flat definitions."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from learning.polytwin.factory_surface_profiles import (
    FACTORY_PREPOLISH,
    make_factory_centered_patch,
)
from learning.rl.gate_f_curved_geometry import (
    SurfaceGeometrySpec,
    build_surface_mesh,
    project_normal_force_n,
    surface_height_normal,
)


PATCH_SIZE_M = (0.32, 0.32)


def _array_hashes(state) -> dict[str, str]:
    names = (
        "micro_height_um",
        "initial_micro_height_um",
        "initial_scratch_depth_um",
        "residual_scratch_depth_um",
        "cumulative_removal_um",
        "clearcoat_remaining_um",
        "initial_clearcoat_um",
        "temperature_c",
    )
    return {
        name: hashlib.sha256(np.ascontiguousarray(getattr(state, name)).tobytes()).hexdigest()
        for name in names
    }


def test_flat_vectorized_height_and_normal_are_exact_everywhere() -> None:
    u = np.linspace(-0.02, PATCH_SIZE_M[0] + 0.02, 41)
    v = np.linspace(-0.02, PATCH_SIZE_M[1] + 0.02, 41)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    height, normal = surface_height_normal("flat", 0.60, PATCH_SIZE_M, uu, vv)
    np.testing.assert_array_equal(height, np.zeros_like(uu))
    expected = np.zeros(uu.shape + (3,), dtype=np.float64)
    expected[..., 2] = 1.0
    np.testing.assert_array_equal(normal, expected)


def test_flat_mesh_matches_established_workpiece_coordinates() -> None:
    grid_size = 41
    margin = 0.02
    mesh = build_surface_mesh(
        SurfaceGeometrySpec("flat", PATCH_SIZE_M, 0.60, 0),
        grid_size=grid_size,
        margin_m=margin,
    )
    u = np.linspace(-margin, PATCH_SIZE_M[0] + margin, grid_size)
    v = np.linspace(-margin, PATCH_SIZE_M[1] + margin, grid_size)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    expected = np.stack(
        (
            uu - PATCH_SIZE_M[0] / 2,
            vv - PATCH_SIZE_M[1] / 2,
            np.zeros_like(uu),
        ),
        axis=-1,
    ).reshape(-1, 3)
    np.testing.assert_array_equal(mesh.vertices_m, expected)
    assert mesh.vertices_m[:, 0].min() == pytest.approx(-0.18)
    assert mesh.vertices_m[:, 0].max() == pytest.approx(0.18)
    assert mesh.vertices_m[:, 1].min() == pytest.approx(-0.18)
    assert mesh.vertices_m[:, 1].max() == pytest.approx(0.18)


def test_flat_local_normal_force_is_absolute_world_z() -> None:
    rng = np.random.default_rng(20260831)
    forces = rng.normal(0.0, 20.0, size=(2048, 3))
    normals = np.zeros_like(forces)
    normals[:, 2] = 1.0
    projected = project_normal_force_n(forces, normals)
    np.testing.assert_array_equal(projected, np.abs(forces[:, 2]))


def test_flat_target_z_and_gap_reduce_to_established_formulas() -> None:
    rng = np.random.default_rng(20260832)
    u = rng.uniform(0.0, PATCH_SIZE_M[0], 512)
    v = rng.uniform(0.0, PATCH_SIZE_M[1], 512)
    height, _ = surface_height_normal("flat", 0.60, PATCH_SIZE_M, u, v)
    work_top_m = 0.40
    clearance = rng.uniform(-0.003, 0.08, 512)
    pad_face_z = rng.uniform(0.35, 0.50, 512)
    gate_f_target_z = work_top_m + height + clearance
    gate_f_gap = pad_face_z - (work_top_m + height)
    np.testing.assert_array_equal(gate_f_target_z, work_top_m + clearance)
    np.testing.assert_array_equal(gate_f_gap, pad_face_z - work_top_m)


def test_radius_to_infinity_converges_over_full_map() -> None:
    u = np.linspace(0.0, PATCH_SIZE_M[0], 33)
    v = np.linspace(0.0, PATCH_SIZE_M[1], 33)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    for kind in ("cylinder", "sphere"):
        height, normal = surface_height_normal(kind, 1.0e8, PATCH_SIZE_M, uu, vv)
        assert float(np.max(np.abs(height))) < 1.0e-8
        assert float(np.max(np.abs(normal[..., :2]))) < 2.0e-9
        np.testing.assert_allclose(normal[..., 2], 1.0, atol=1.0e-14)


def test_factory_state_flat_macro_geometry_and_quality_arrays_are_unchanged() -> None:
    state, metadata = make_factory_centered_patch(
        FACTORY_PREPOLISH,
        map_size_m=PATCH_SIZE_M,
        roi_size_m=(0.20, 0.20),
        resolution_m=0.002,
        seed=26000,
    )
    before_hashes = _array_hashes(state)
    u = state.nominal_surface_xyz_m[..., 0]
    v = state.nominal_surface_xyz_m[..., 1]
    height, normal = surface_height_normal("flat", 0.60, PATCH_SIZE_M, u, v)
    np.testing.assert_array_equal(state.nominal_surface_xyz_m[..., 2], height)
    np.testing.assert_array_equal(state.normal_xyz, normal)
    assert _array_hashes(state) == before_hashes
    assert metadata["profile_id"] == FACTORY_PREPOLISH

