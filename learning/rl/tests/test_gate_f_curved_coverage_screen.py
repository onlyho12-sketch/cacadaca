"""CPU regression tests for the Gate F3 curved coverage screen."""
from __future__ import annotations

import numpy as np

from learning.rl.gate_c_path_coverage_screen import integrate_exposure
from learning.rl.gate_f_curved_coverage_screen import (
    DIRECTION_MODES,
    GeometryCandidate,
    geometry_candidates,
    screen_candidate,
    select_f4_candidates,
)


def test_candidate_matrix_has_declared_families_and_count() -> None:
    candidates = geometry_candidates()
    assert len(candidates) == 12
    assert {candidate.kind for candidate in candidates} == {
        "flat", "cylinder", "sphere", "freeform"
    }
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert len(candidates) * len(DIRECTION_MODES) == 24


def test_flat_coverage_matches_gate_c_definition() -> None:
    flat = GeometryCandidate("flat", "flat", 1.0e8, 0)
    gate_f = screen_candidate(flat, "cross_xy", coverage_sample_ds_m=0.004)
    gate_c = integrate_exposure(
        0.40, "balanced_extend5", "cross_xy", sample_ds_m=0.004
    )
    np.testing.assert_allclose(
        gate_f["normalized_exposure"], gate_c["normalized_exposure"], atol=2.0e-14
    )
    assert gate_f["scalars"]["path_length_3d_m"] == gate_c["scalars"]["path_length_m"]
    assert gate_f["scalars"]["minimum_quality_map_projected_margin_m"] >= -1.0e-12


def test_curved_screen_is_finite_and_physically_supported() -> None:
    candidate = GeometryCandidate("sphere_r0p60", "sphere", 0.60, 0)
    result = screen_candidate(
        candidate,
        "same_xx",
        resolution_m=0.010,
        coverage_sample_ds_m=0.020,
        geometry_sample_ds_m=0.020,
    )
    scalars = result["scalars"]
    assert scalars["all_finite"]
    assert scalars["physical_mesh_projected_footprint_inside"]
    assert scalars["coverage_any_fraction"] == 1.0
    assert scalars["coverage_effective_fraction"] >= 0.99
    assert scalars["surface_tilt_max_deg"] > 0.0
    assert scalars["pad_footprint_normal_change_max_deg"] > 0.0
    assert scalars["local_tangent_plane_deviation_max_m"] > 0.0
    assert result["normalized_exposure"].shape == (20, 20)
    assert result["tile_exposure_norm"].shape == (5, 5)


def test_selection_uses_representative_median_geometry_per_family() -> None:
    rows = []
    for kind, tilts in {
        "flat": [0.0],
        "cylinder": [5.0, 10.0, 15.0, 20.0],
        "sphere": [6.0, 11.0, 16.0, 21.0],
        "freeform": [3.0, 8.0, 13.0],
    }.items():
        for index, tilt in enumerate(tilts):
            for direction, cv in (("same_xx", 0.20), ("cross_xy", 0.10)):
                rows.append({
                    "candidate_id": f"{kind}_{index}",
                    "kind": kind,
                    "all_finite": True,
                    "physical_mesh_projected_footprint_inside": True,
                    "coverage_effective_fraction": 1.0,
                    "surface_tilt_max_deg": tilt,
                    "exposure_cv": cv,
                    "tile_exposure_range_norm": cv,
                    "path_length_3d_m": 2.1,
                    "direction_mode": direction,
                })
    selected = select_f4_candidates(rows)
    assert len(selected) == 4
    assert all(row["direction_mode"] == "cross_xy" for row in selected)
    selected_by_kind = {row["kind"]: row for row in selected}
    assert selected_by_kind["cylinder"]["surface_tilt_max_deg"] == 10.0
    assert selected_by_kind["sphere"]["surface_tilt_max_deg"] == 11.0
    assert selected_by_kind["freeform"]["surface_tilt_max_deg"] == 8.0

