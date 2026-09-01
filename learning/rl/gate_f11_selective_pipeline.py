"""F11 standalone selective-polishing contract over the validated F10 modules.

This module does not hook ``polishing_v5``.  It connects the existing local
all-4 quality calculation, PASS_LOCKED ledger, bounded region planner and an
actual surface-diff revalidation contract.  Concave cells are explicitly held
for review in the currently approved scope.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntFlag

import numpy as np

from learning.rl.gate_e6_area_quality import AreaQualityTargets, local_quality_maps
from learning.rl.gate_f10_quality_state import (
    CLEARCOAT_GUARD,
    NO_IMPROVEMENT,
    NOT_REACHED,
    PASS_LOCKED,
    REWORK,
    REWORK_AFFECTED,
    REWORK_FAIL,
    REWORK_PASS,
    UNSAFE_GEOMETRY,
    QualityLedger,
    apply_selective_revalidation,
    cell_rows,
    inspect_after_coarse,
    mark_surface_changes,
    state_counts,
)
from learning.rl.gate_f10_rework_planner import PlannerConfig, build_plan


class FailureReason(IntFlag):
    NONE = 0
    GU_BELOW_70 = 1 << 0
    RA_ABOVE_020 = 1 << 1
    RZ_ABOVE_200 = 1 << 2
    SCRATCH_NOT_IMPROVED = 1 << 3
    CLEARCOAT_GUARD = 1 << 4
    TEMPERATURE_GUARD = 1 << 5
    UNSAFE_GEOMETRY = 1 << 6
    NOT_REACHED = 1 << 7
    CONCAVE_EXCLUDED = 1 << 8
    NO_IMPROVEMENT = 1 << 9


SURFACE_DIFF_FIELDS = (
    "micro_height_um",
    "cumulative_removal_um",
    "clearcoat_remaining_um",
    "dwell_time_s",
    "pass_count",
    "peak_temperature_c",
    "thermal_damage_proxy",
)


@dataclass(frozen=True)
class CellGeometry:
    normal_xyz: np.ndarray
    k1_1_m: np.ndarray
    k2_1_m: np.ndarray
    curvature_radius_m: np.ndarray
    boundary_risk: np.ndarray
    geometry_risk: np.ndarray
    fit_confidence: np.ndarray
    surface_class: np.ndarray

    @classmethod
    def from_surface_state(cls, state):
        shape = state.shape
        zeros = np.zeros(shape, dtype=np.float64)
        return cls(
            normal_xyz=np.asarray(state.normal_xyz, dtype=np.float64).copy(),
            k1_1_m=zeros.copy(), k2_1_m=zeros.copy(),
            curvature_radius_m=np.full(shape, np.inf),
            boundary_risk=zeros.copy(), geometry_risk=zeros.copy(),
            fit_confidence=np.ones(shape, dtype=np.float64),
            surface_class=np.full(shape, "near_flat", dtype="U32"),
        )

    def validate(self, shape):
        arrays = {
            "normal_xyz": (self.normal_xyz, shape + (3,)),
            "k1_1_m": (self.k1_1_m, shape), "k2_1_m": (self.k2_1_m, shape),
            "curvature_radius_m": (self.curvature_radius_m, shape),
            "boundary_risk": (self.boundary_risk, shape),
            "geometry_risk": (self.geometry_risk, shape),
            "fit_confidence": (self.fit_confidence, shape),
            "surface_class": (self.surface_class, shape),
        }
        for name, (value, expected) in arrays.items():
            if np.asarray(value).shape != expected:
                raise ValueError(f"{name} shape {np.asarray(value).shape} != {expected}")


@dataclass
class SelectivePipeline:
    surface_id: str
    ledger: QualityLedger
    geometry: CellGeometry
    concave_excluded_mask: np.ndarray
    regions: list[dict]
    rework_path: list[dict]
    planned_footprint_mask: np.ndarray
    planner_summary: dict
    pre_rework_state: object | None = None
    actual_changed_mask: np.ndarray | None = None
    revalidated_mask: np.ndarray | None = None


def _bool_mask(value, shape, name, default=False):
    if value is None:
        return np.full(shape, default, dtype=bool)
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} != {shape}")
    return result.copy()


def begin_pipeline(state, *, geometry: CellGeometry | None = None,
                   reached_mask=None, unsafe_geometry_mask=None,
                   exclude_concave=True, targets: AreaQualityTargets | None = None,
                   planner_cfg: PlannerConfig | None = None,
                   surface_id: str = "surface_0") -> SelectivePipeline:
    """Inspect every cell once and build a bounded second-pass plan."""
    geometry = geometry or CellGeometry.from_surface_state(state)
    geometry.validate(state.shape)
    unsafe = _bool_mask(unsafe_geometry_mask, state.shape, "unsafe_geometry_mask")
    concave = np.asarray(geometry.surface_class) == "concave"
    if exclude_concave:
        unsafe |= concave
    ledger = inspect_after_coarse(
        state, reached_mask=reached_mask, unsafe_geometry_mask=unsafe,
        recipe_id="coarse_C_SL_SR_deterministic_g3", targets=targets)
    cfg = planner_cfg or PlannerConfig(resolution_m=float(state.resolution_m))
    if abs(cfg.resolution_m - float(state.resolution_m)) > 1e-12:
        raise ValueError("planner and surface resolution must match")
    regions, path, planned, summary = build_plan(ledger.status, cfg)
    summary["automatic_rework_allowed"] = bool(
        summary["rejected_barrier_regions"] == 0
        and summary["regions"] <= cfg.max_regions_total
        and summary["target_fraction"] <= cfg.max_rework_fraction
        and summary["changed_to_target_ratio"] <= cfg.max_changed_to_target_ratio
        and summary["nominal_rework_time_s"] <= cfg.max_parallel_vehicle_h * 3600.0)
    summary["automatic_rework_block_reasons"] = [
        name for condition, name in (
            (summary["rejected_barrier_regions"] > 0, "BARRIER_INTERSECTION"),
            (summary["regions"] > cfg.max_regions_total, "TOO_MANY_REGIONS"),
            (summary["target_fraction"] > cfg.max_rework_fraction,
             "REWORK_FRACTION_LIMIT"),
            (summary["changed_to_target_ratio"] > cfg.max_changed_to_target_ratio,
             "COLLATERAL_FOOTPRINT_LIMIT"),
            (summary["nominal_rework_time_s"] > cfg.max_parallel_vehicle_h * 3600.0,
             "PHYSICAL_TIME_LIMIT"),
        ) if condition]
    return SelectivePipeline(
        surface_id=str(surface_id), ledger=ledger, geometry=geometry,
        concave_excluded_mask=concave if exclude_concave else np.zeros(state.shape, bool),
        regions=regions, rework_path=path, planned_footprint_mask=planned,
        planner_summary=summary,
    )


def snapshot_before_rework(pipeline: SelectivePipeline, state) -> None:
    """Freeze the exact surface arrays used to detect real second-pass changes."""
    if tuple(state.shape) != pipeline.ledger.shape:
        raise ValueError("state and ledger shapes differ")
    pipeline.pre_rework_state = state.copy()


def actual_surface_change_mask(before, after, *, atol=1e-12) -> np.ndarray:
    """Detect cells changed by physics/model state, independent of planned footprint."""
    if tuple(before.shape) != tuple(after.shape):
        raise ValueError("before and after surface shapes differ")
    changed = np.zeros(before.shape, dtype=bool)
    for field in SURFACE_DIFF_FIELDS:
        old = np.asarray(getattr(before, field))
        new = np.asarray(getattr(after, field))
        if old.shape != before.shape or new.shape != before.shape:
            raise ValueError(f"surface field {field} does not match cell shape")
        if np.issubdtype(old.dtype, np.integer) or np.issubdtype(old.dtype, np.bool_):
            changed |= old != new
        else:
            changed |= ~np.isclose(old, new, rtol=0.0, atol=atol, equal_nan=False)
    return changed


def _improved(before_quality, after_quality, selected):
    return selected & (
        (after_quality["local_gu"] > before_quality["local_gu"] + 1e-12)
        | (after_quality["local_ra_um"] < before_quality["local_ra_um"] - 1e-12)
        | (after_quality["local_rz_um"] < before_quality["local_rz_um"] - 1e-12)
        | (after_quality["scratch_residual_um"]
           < before_quality["scratch_residual_um"] - 1e-12)
    )


def finish_pipeline(pipeline: SelectivePipeline, state_after,
                    *, targets: AreaQualityTargets | None = None) -> dict:
    """Revalidate only cells whose surface arrays actually changed."""
    if pipeline.pre_rework_state is None:
        raise RuntimeError("snapshot_before_rework must be called before finish_pipeline")
    before_status = pipeline.ledger.status.copy()
    before_quality = {name: value.copy() for name, value in pipeline.ledger.quality.items()}
    actual = actual_surface_change_mask(pipeline.pre_rework_state, state_after)
    mark_surface_changes(pipeline.ledger, actual)
    after_quality = local_quality_maps(state_after, targets)
    selected = apply_selective_revalidation(pipeline.ledger, after_quality, actual)
    blocked = np.isin(before_status, (UNSAFE_GEOMETRY, CLEARCOAT_GUARD, NOT_REACHED))
    pipeline.ledger.status[selected & blocked] = before_status[selected & blocked]
    originally_rework = before_status == REWORK
    no_improvement = selected & originally_rework & (pipeline.ledger.status == REWORK_FAIL)
    no_improvement &= ~_improved(before_quality, after_quality, selected)
    pipeline.ledger.status[no_improvement] = NO_IMPROVEMENT
    pipeline.ledger.pass_count = np.asarray(state_after.pass_count, dtype=np.int32).copy()
    pipeline.ledger.cumulative_contact_time_s = np.asarray(
        state_after.dwell_time_s, dtype=np.float64).copy()
    pipeline.actual_changed_mask = actual
    pipeline.revalidated_mask = selected
    unchanged_lock = pipeline.ledger.initially_pass_locked & ~actual
    planned = pipeline.planned_footprint_mask
    return {
        "actual_changed_cells": int(actual.sum()),
        "planned_footprint_cells": int(planned.sum()),
        "changed_inside_planned_cells": int((actual & planned).sum()),
        "changed_outside_planned_cells": int((actual & ~planned).sum()),
        "planned_but_unchanged_cells": int((planned & ~actual).sum()),
        "revalidated_cells": int(selected.sum()),
        "revalidation_exactly_actual_changed": bool(np.array_equal(selected, actual)),
        "unchanged_pass_locked_revalidated": int(
            pipeline.ledger.validation_count[unchanged_lock].sum() - unchanged_lock.sum()),
        "rework_affected_cells": int(pipeline.ledger.rework_affected_ever.sum()),
        "blocked_cells_changed": int((selected & blocked).sum()),
        "no_improvement_cells": int(no_improvement.sum()),
        "state_counts": state_counts(pipeline.ledger),
    }


def failure_reason_mask(pipeline: SelectivePipeline) -> np.ndarray:
    ledger = pipeline.ledger
    reason = np.zeros(ledger.shape, dtype=np.uint16)
    reason[~ledger.quality["gu_pass"]] |= int(FailureReason.GU_BELOW_70)
    reason[~ledger.quality["ra_pass"]] |= int(FailureReason.RA_ABOVE_020)
    reason[~ledger.quality["rz_pass"]] |= int(FailureReason.RZ_ABOVE_200)
    reason[~ledger.quality["scratch_pass"]] |= int(FailureReason.SCRATCH_NOT_IMPROVED)
    reason[~ledger.quality["clearcoat_pass"]] |= int(FailureReason.CLEARCOAT_GUARD)
    reason[~ledger.quality["temperature_pass"]] |= int(FailureReason.TEMPERATURE_GUARD)
    reason[ledger.unsafe_geometry_mask] |= int(FailureReason.UNSAFE_GEOMETRY)
    reason[~ledger.reached_mask] |= int(FailureReason.NOT_REACHED)
    reason[pipeline.concave_excluded_mask] |= int(FailureReason.CONCAVE_EXCLUDED)
    reason[ledger.status == NO_IMPROVEMENT] |= int(FailureReason.NO_IMPROVEMENT)
    return reason


def _reason_names(value):
    return "|".join(item.name for item in FailureReason
                    if item is not FailureReason.NONE and value & int(item)) or "NONE"


def ui_cell_rows(pipeline: SelectivePipeline) -> list[dict]:
    rows = cell_rows(pipeline.ledger)
    reason = failure_reason_mask(pipeline)
    geometry = pipeline.geometry
    for row in rows:
        i, j = int(row["cell_i"]), int(row["cell_j"])
        bits = int(reason[i, j])
        row.update({
            "surface_id": pipeline.surface_id,
            "failure_reason_bitmask": bits,
            "failure_reasons": _reason_names(bits),
            "normal_x": float(geometry.normal_xyz[i, j, 0]),
            "normal_y": float(geometry.normal_xyz[i, j, 1]),
            "normal_z": float(geometry.normal_xyz[i, j, 2]),
            "k1_1_m": float(geometry.k1_1_m[i, j]),
            "k2_1_m": float(geometry.k2_1_m[i, j]),
            "curvature_radius_m": float(geometry.curvature_radius_m[i, j]),
            "boundary_risk": float(geometry.boundary_risk[i, j]),
            "geometry_risk": float(geometry.geometry_risk[i, j]),
            "fit_confidence": float(geometry.fit_confidence[i, j]),
            "surface_class": str(geometry.surface_class[i, j]),
            "concave_excluded": bool(pipeline.concave_excluded_mask[i, j]),
            "planned_rework_footprint": bool(pipeline.planned_footprint_mask[i, j]),
            "actual_surface_changed": bool(
                pipeline.actual_changed_mask[i, j]
                if pipeline.actual_changed_mask is not None else False),
            "revalidated_after_rework": bool(
                pipeline.revalidated_mask[i, j]
                if pipeline.revalidated_mask is not None else False),
        })
    return rows


def write_evidence(out_dir, pipeline: SelectivePipeline, finish_summary: dict) -> None:
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir)
    outputs = {
        "cell_quality_ui.csv": ui_cell_rows(pipeline),
        "rework_regions.csv": pipeline.regions,
        "rework_path.csv": pipeline.rework_path,
    }
    for name, rows in outputs.items():
        if not rows:
            continue
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F11_SELECTIVE_PIPELINE_STANDALONE",
        "status": "CPU_CONTRACT_COMPLETE_PHYSX_NOT_YET_RUN",
        "actual_gu_sensor_connected": False,
        "quality_source": "learning.rl.gate_e6_area_quality.local_quality_maps",
        "concave_policy": "EXCLUDED_HOLD_REVIEW",
        "planner_summary": pipeline.planner_summary,
        "finish_summary": finish_summary,
        "polishing_v5_modified": False,
        "feature_flag_integration_performed": False,
    }
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
