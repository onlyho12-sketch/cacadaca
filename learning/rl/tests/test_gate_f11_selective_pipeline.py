"""CPU integration tests for the standalone F11 selective pipeline."""
from __future__ import annotations

import numpy as np

from learning.polytwin.surface_state import make_flat_patch
from learning.rl import gate_f11_selective_pipeline as f11
from learning.rl.gate_f10_quality_state import (
    NO_IMPROVEMENT, PASS_LOCKED, REWORK_AFFECTED, REWORK_PASS, UNSAFE_GEOMETRY)
from learning.rl.gate_f10_rework_planner import PlannerConfig


def blank(size=0.04):
    state = make_flat_patch((size, size), 0.002, seed=11,
                            target_ra_um=0.0, with_scratches=False)
    for values in (state.micro_height_um, state.initial_micro_height_um,
                   state.initial_scratch_depth_um, state.residual_scratch_depth_um,
                   state.cumulative_removal_um, state.thermal_damage_proxy):
        values[:] = 0.0
    state.defect_mask[:] = False
    state.healthy_mask[:] = True
    state.clearcoat_remaining_um[:] = 40.0
    state.initial_clearcoat_um[:] = 40.0
    state.temperature_c[:] = 25.0
    state.peak_temperature_c[:] = 25.0
    return state


def geometry(state, surface_class="near_flat"):
    result = f11.CellGeometry.from_surface_state(state)
    result.surface_class[:] = surface_class
    return result


def planner_cfg(state):
    return PlannerConfig(
        resolution_m=state.resolution_m, pad_radius_m=0.004,
        fine_stepover_m=0.004, merge_gap_m=0.004,
        max_changed_to_target_ratio=20.0)


def test_blank_pipeline_locks_every_cell_and_has_no_rework_path():
    state = blank()
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              planner_cfg=planner_cfg(state))
    assert np.all(pipe.ledger.status == PASS_LOCKED)
    assert not pipe.planned_footprint_mask.any()
    assert pipe.regions == []
    assert pipe.planner_summary["automatic_rework_allowed"]


def test_concave_cells_are_explicitly_excluded_from_rework():
    state = blank()
    state.initial_scratch_depth_um[8:12, 8:12] = 1.0
    state.defect_mask[8:12, 8:12] = True
    state.micro_height_um[8:12, 8:12] = -1.0
    geom = geometry(state)
    geom.surface_class[9, 9] = "concave"
    pipe = f11.begin_pipeline(state, geometry=geom, planner_cfg=planner_cfg(state))
    assert pipe.ledger.status[9, 9] == UNSAFE_GEOMETRY
    bits = int(f11.failure_reason_mask(pipe)[9, 9])
    assert bits & int(f11.FailureReason.CONCAVE_EXCLUDED)
    assert bits & int(f11.FailureReason.UNSAFE_GEOMETRY)


def test_only_actual_surface_changes_are_revalidated_not_planned_footprint():
    state = blank()
    state.initial_scratch_depth_um[9:11, 9:11] = 1.0
    state.defect_mask[9:11, 9:11] = True
    state.micro_height_um[9:11, 9:11] = -1.0
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              planner_cfg=planner_cfg(state))
    assert pipe.planned_footprint_mask.sum() > 1
    f11.snapshot_before_rework(pipe, state)
    after = state.copy()
    changed_cell = (9, 9)
    after.micro_height_um[changed_cell] += 0.25
    summary = f11.finish_pipeline(pipe, after)
    assert summary["actual_changed_cells"] == 1
    assert summary["revalidated_cells"] == 1
    assert summary["planned_but_unchanged_cells"] > 0
    assert summary["revalidation_exactly_actual_changed"]
    assert summary["unchanged_pass_locked_revalidated"] == 0


def test_actual_pad_touch_unlocks_locked_cell_and_records_rework_affected():
    state = blank()
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              planner_cfg=planner_cfg(state))
    f11.snapshot_before_rework(pipe, state)
    after = state.copy()
    after.cumulative_removal_um[5, 5] += 0.01
    after.clearcoat_remaining_um[5, 5] -= 0.01
    before_finish = pipe.ledger.status.copy()
    assert before_finish[5, 5] == PASS_LOCKED
    summary = f11.finish_pipeline(pipe, after)
    assert pipe.ledger.rework_affected_ever[5, 5]
    assert pipe.ledger.status[5, 5] == REWORK_PASS
    assert summary["rework_affected_cells"] == 1
    assert pipe.ledger.validation_count[5, 5] == 2


def test_failed_cell_with_only_counter_change_becomes_no_improvement():
    state = blank()
    state.initial_scratch_depth_um[10, 10] = 1.0
    state.defect_mask[10, 10] = True
    state.micro_height_um[10, 10] = -1.0
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              planner_cfg=planner_cfg(state))
    f11.snapshot_before_rework(pipe, state)
    after = state.copy()
    after.pass_count[10, 10] += 1
    summary = f11.finish_pipeline(pipe, after)
    assert pipe.ledger.status[10, 10] == NO_IMPROVEMENT
    assert summary["no_improvement_cells"] == 1


def test_ui_rows_expose_all_four_flags_reason_and_geometry():
    state = blank()
    state.initial_scratch_depth_um[10, 10] = 1.0
    state.defect_mask[10, 10] = True
    state.micro_height_um[10, 10] = -1.0
    geom = geometry(state)
    geom.k1_1_m[10, 10] = -2.0
    geom.k2_1_m[10, 10] = -4.0
    geom.surface_class[10, 10] = "convex"
    pipe = f11.begin_pipeline(state, geometry=geom, planner_cfg=planner_cfg(state))
    row = next(item for item in f11.ui_cell_rows(pipe)
               if item["cell_i"] == 10 and item["cell_j"] == 10)
    assert {"local_gu", "local_ra_um", "local_rz_um", "scratch_residual_um",
            "gu_pass", "ra_pass", "rz_pass", "scratch_pass", "all4_pass",
            "failure_reason_bitmask", "failure_reasons", "status"} <= set(row)
    assert row["surface_class"] == "convex"
    assert row["k1_1_m"] == -2.0 and row["k2_1_m"] == -4.0
    assert row["failure_reason_bitmask"] != 0


def test_reached_and_unsafe_masks_keep_state_precedence():
    state = blank()
    reached = np.ones(state.shape, bool)
    unsafe = np.zeros(state.shape, bool)
    reached[0, 0] = False
    unsafe[0, 1] = True
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              reached_mask=reached, unsafe_geometry_mask=unsafe,
                              planner_cfg=planner_cfg(state))
    assert pipe.ledger.status[0, 0] != REWORK_AFFECTED
    assert pipe.ledger.status[0, 0] != REWORK_PASS
    bits0 = int(f11.failure_reason_mask(pipe)[0, 0])
    bits1 = int(f11.failure_reason_mask(pipe)[0, 1])
    assert bits0 & int(f11.FailureReason.NOT_REACHED)
    assert bits1 & int(f11.FailureReason.UNSAFE_GEOMETRY)


def test_changed_blocked_cell_remains_blocked_after_revalidation():
    state = blank()
    unsafe = np.zeros(state.shape, bool)
    unsafe[4, 4] = True
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              unsafe_geometry_mask=unsafe,
                              planner_cfg=planner_cfg(state))
    f11.snapshot_before_rework(pipe, state)
    after = state.copy()
    after.cumulative_removal_um[4, 4] += 0.01
    summary = f11.finish_pipeline(pipe, after)
    assert summary["blocked_cells_changed"] == 1
    assert pipe.ledger.status[4, 4] == UNSAFE_GEOMETRY


def test_collateral_limit_blocks_automatic_rework_without_deleting_plan():
    state = blank()
    state.initial_scratch_depth_um[10, 10] = 1.0
    state.defect_mask[10, 10] = True
    state.micro_height_um[10, 10] = -1.0
    cfg = PlannerConfig(
        resolution_m=state.resolution_m, pad_radius_m=0.01,
        fine_stepover_m=0.004, merge_gap_m=0.0,
        max_changed_to_target_ratio=1.0)
    pipe = f11.begin_pipeline(state, geometry=geometry(state), planner_cfg=cfg)
    assert pipe.regions
    assert not pipe.planner_summary["automatic_rework_allowed"]
    assert "COLLATERAL_FOOTPRINT_LIMIT" in pipe.planner_summary[
        "automatic_rework_block_reasons"]


def test_ui_rows_include_stable_surface_and_cell_identity():
    state = blank()
    pipe = f11.begin_pipeline(state, geometry=geometry(state),
                              planner_cfg=planner_cfg(state), surface_id="hood_C")
    row = f11.ui_cell_rows(pipe)[0]
    assert row["surface_id"] == "hood_C"
    assert row["cell_i"] == 0 and row["cell_j"] == 0
