"""CPU tests for the F10-E bounded region planner."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f10_rework_planner as rp  # noqa: E402
from learning.rl.gate_f10_quality_state import (  # noqa: E402
    PASS_LOCKED, REWORK, UNSAFE_GEOMETRY,
)


def test_nearby_failed_cells_merge_into_one_region():
    status = np.full((80, 80), PASS_LOCKED, dtype="U24")
    status[20:25, 20:25] = REWORK
    status[20:25, 30:35] = REWORK
    labels, regions = rp.group_rework_regions(status, rp.PlannerConfig())
    assert len(regions) == 1
    assert set(np.unique(labels)) == {0, 1}


def test_distant_failures_remain_separate():
    status = np.full((100, 100), PASS_LOCKED, dtype="U24")
    status[10:15, 10:15] = REWORK
    status[70:75, 70:75] = REWORK
    _, regions = rp.group_rework_regions(status, rp.PlannerConfig())
    assert len(regions) == 2


def test_pad_footprint_records_touched_locked_cells():
    status = np.full((100, 100), PASS_LOCKED, dtype="U24")
    status[40:50, 40:50] = REWORK
    regions, _paths, changed, summary = rp.build_plan(status, rp.PlannerConfig())
    assert len(regions) == 1
    assert summary["rework_affected_locked_cells"] > 0
    assert changed.sum() == summary["revalidation_cells"]


def test_small_isolated_target_exposes_collateral_ratio():
    status = np.full((100, 100), PASS_LOCKED, dtype="U24")
    status[48:52, 48:52] = REWORK
    _, _, _, summary = rp.build_plan(status, rp.PlannerConfig())
    assert summary["changed_to_target_ratio"] > rp.PlannerConfig().max_changed_to_target_ratio


def test_barrier_intersection_rejects_region():
    status = np.full((100, 100), PASS_LOCKED, dtype="U24")
    status[40:50, 40:50] = REWORK
    status[50:55, 40:50] = UNSAFE_GEOMETRY
    regions, _, _, summary = rp.build_plan(status, rp.PlannerConfig())
    assert regions[0]["plan_state"] == "REJECT_BARRIER"
    assert summary["ready_regions"] == 0


def test_serpentine_path_is_one_continuous_region_path():
    mask = np.zeros((80, 80), bool)
    mask[20:60, 20:60] = True
    path = rp.raster_path_for_region(mask, rp.PlannerConfig())
    assert len(path) >= 4
    assert np.isfinite(path).all()
    assert rp.path_length_m(path) > 0.0


def test_time_guard_rejects_twelve_hour_structure(tmp_path):
    baseline = {"summary": {"coarse_parallel_h": 0.8, "rework_scenarios": [
        {"failed_path_fraction": 0.5, "fine_feed_mm_s": 8.0,
         "regions_per_active_rail": 3, "coarse_plus_fine_parallel_h": 1.42}]}}
    path = tmp_path / "time.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    guard = rp.time_guard(str(path), rp.PlannerConfig())
    assert guard["time_guard_pass"]
    assert not guard["twelve_hour_estimate_supported_by_local_path_model"]
