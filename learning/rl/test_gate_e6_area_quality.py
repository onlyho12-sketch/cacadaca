"""CPU-only checks for Gate E6 sub-tile area accounting."""
from __future__ import annotations

import numpy as np

from learning.polytwin.surface_state import make_flat_patch
from learning.rl.gate_e6_area_quality import AreaQualityTargets, summarize_area_quality


def _blank_state():
    state = make_flat_patch(
        patch_size_m=(0.02, 0.02), resolution_m=0.002,
        seed=31, target_ra_um=0.0, with_scratches=False)
    state.micro_height_um[:] = 0.0
    state.initial_micro_height_um[:] = 0.0
    state.initial_scratch_depth_um[:] = 0.0
    state.residual_scratch_depth_um[:] = 0.0
    state.defect_mask[:] = False
    state.healthy_mask[:] = True
    state.cumulative_removal_um[:] = 0.0
    state.thermal_damage_proxy[:] = 0.0
    state.peak_temperature_c[:] = 25.0
    state.clearcoat_remaining_um[:] = 40.0
    return state


def test_blank_surface_passes_every_local_quality_cell():
    result = summarize_area_quality(_blank_state(), AreaQualityTargets(tiles_x=5, tiles_y=5))
    assert len(result["tile_rows"]) == 25
    assert result["summary"]["roi_all4_pass_area_pct"] == 100.0
    for row in result["tile_rows"]:
        assert row["all4_pass_area_pct"] == 100.0


def test_one_unimproved_scratch_cell_is_not_hidden_by_tile_average():
    state = _blank_state()
    state.initial_scratch_depth_um[4, 4] = 1.0
    state.defect_mask[4, 4] = True
    state.micro_height_um[4, 4] = -1.0
    result = summarize_area_quality(state)
    scratch_area = result["summary"]["roi_scratch_pass_area_pct"]
    assert 0.0 < scratch_area < 100.0
    assert any(row["scratch_pass_area_pct"] < 100.0 for row in result["tile_rows"])


def test_rough_region_reduces_local_ra_rz_and_all4_areas():
    state = _blank_state()
    checker = np.indices(state.shape).sum(axis=0) % 2
    state.micro_height_um[2:8, 2:8] = checker[2:8, 2:8] * 2.0 - 1.0
    result = summarize_area_quality(state)
    summary = result["summary"]
    assert summary["roi_ra_pass_area_pct"] < 100.0
    assert summary["roi_rz_pass_area_pct"] < 100.0
    assert summary["roi_all4_pass_area_pct"] < 100.0

