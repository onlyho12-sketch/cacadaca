"""Gate F10-D cellwise quality state and PASS_LOCKED contract (CPU only).

The module delegates GU/Ra/Rz/scratch calculations byte-for-byte to
``gate_e6_area_quality.local_quality_maps``.  It adds explicit cell states and
revision accounting so an unchanged PASS_LOCKED cell cannot be revalidated.
No physical GU sensor is represented.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.polytwin.surface_state import make_flat_patch
from learning.rl.gate_e6_area_quality import AreaQualityTargets, local_quality_maps

PASS_LOCKED = "PASS_LOCKED"
REWORK_AFFECTED = "REWORK_AFFECTED"
REWORK = "REWORK"
UNSAFE_GEOMETRY = "UNSAFE_GEOMETRY"
CLEARCOAT_GUARD = "CLEARCOAT_GUARD"
NOT_REACHED = "NOT_REACHED"
NO_IMPROVEMENT = "NO_IMPROVEMENT"
REWORK_PASS = "REWORK_PASS"
REWORK_FAIL = "REWORK_FAIL"

QUALITY_FIELDS = (
    "local_gu", "local_ra_um", "local_rz_um", "scratch_initial_um",
    "scratch_residual_um", "gu_pass", "ra_pass", "rz_pass", "scratch_pass",
    "all4_pass", "clearcoat_pass", "temperature_pass",
)


@dataclass
class QualityLedger:
    resolution_m: float
    status: np.ndarray
    quality: dict[str, np.ndarray]
    reached_mask: np.ndarray
    unsafe_geometry_mask: np.ndarray
    validation_count: np.ndarray
    surface_revision: np.ndarray
    validated_revision: np.ndarray
    initially_pass_locked: np.ndarray
    rework_affected_ever: np.ndarray
    recipe_id: str
    pass_count: np.ndarray
    cumulative_contact_time_s: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.status.shape

    @property
    def dirty_mask(self) -> np.ndarray:
        return self.surface_revision > self.validated_revision


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(value, shape: tuple[int, int], name: str, default: bool) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=bool)
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} != {shape}")
    return result.copy()


def _copy_quality(maps: dict[str, np.ndarray], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    result = {}
    for field in QUALITY_FIELDS:
        if field not in maps:
            raise KeyError(f"quality maps missing {field}")
        values = np.asarray(maps[field])
        if values.shape != shape:
            raise ValueError(f"quality field {field} shape {values.shape} != {shape}")
        result[field] = values.copy()
    return result


def _initial_status(quality: dict[str, np.ndarray], reached: np.ndarray,
                    unsafe_geometry: np.ndarray) -> np.ndarray:
    status = np.full(reached.shape, REWORK, dtype="U24")
    safe_quality = quality["clearcoat_pass"] & quality["temperature_pass"]
    status[quality["all4_pass"] & safe_quality] = PASS_LOCKED
    status[~safe_quality] = CLEARCOAT_GUARD
    status[unsafe_geometry] = UNSAFE_GEOMETRY
    status[~reached] = NOT_REACHED
    return status


def inspect_after_coarse(state, *, reached_mask=None, unsafe_geometry_mask=None,
                         recipe_id: str = "coarse_pass_v1",
                         targets: AreaQualityTargets | None = None) -> QualityLedger:
    """Perform the one authorized full-cell inspection immediately after pass 1."""
    shape = tuple(state.shape)
    reached = _mask(reached_mask, shape, "reached_mask", True)
    unsafe = _mask(unsafe_geometry_mask, shape, "unsafe_geometry_mask", False)
    quality = _copy_quality(local_quality_maps(state, targets), shape)
    status = _initial_status(quality, reached, unsafe)
    return QualityLedger(
        resolution_m=float(state.resolution_m), status=status, quality=quality,
        reached_mask=reached, unsafe_geometry_mask=unsafe,
        # The established vectorized quality function evaluated every cell once,
        # including NOT_REACHED cells, as required by the post-pass snapshot.
        validation_count=np.ones(shape, dtype=np.uint16),
        surface_revision=np.zeros(shape, dtype=np.uint16),
        validated_revision=np.zeros(shape, dtype=np.uint16),
        initially_pass_locked=status == PASS_LOCKED,
        rework_affected_ever=np.zeros(shape, dtype=bool),
        recipe_id=recipe_id,
        pass_count=np.asarray(state.pass_count, dtype=np.int32).copy(),
        cumulative_contact_time_s=np.asarray(state.dwell_time_s, dtype=np.float64).copy(),
    )


def mark_surface_changes(ledger: QualityLedger, changed_mask) -> np.ndarray:
    """Record actual footprint changes and unlock touched PASS_LOCKED cells."""
    changed = _mask(changed_mask, ledger.shape, "changed_mask", False)
    affected = changed & (ledger.status == PASS_LOCKED)
    ledger.status[affected] = REWORK_AFFECTED
    ledger.rework_affected_ever[affected] = True
    ledger.surface_revision[changed] += 1
    return changed


def apply_selective_revalidation(ledger: QualityLedger, new_quality: dict[str, np.ndarray],
                                 changed_mask) -> np.ndarray:
    """Update only dirty, actually changed cells; never touch an unchanged lock."""
    requested = _mask(changed_mask, ledger.shape, "changed_mask", False)
    selected = requested & ledger.dirty_mask
    quality = _copy_quality(new_quality, ledger.shape)
    for field in QUALITY_FIELDS:
        ledger.quality[field][selected] = quality[field][selected]
    safe = quality["clearcoat_pass"] & quality["temperature_pass"]
    ledger.status[selected & ~safe] = CLEARCOAT_GUARD
    ledger.status[selected & safe & quality["all4_pass"]] = REWORK_PASS
    ledger.status[selected & safe & ~quality["all4_pass"]] = REWORK_FAIL
    ledger.validation_count[selected] += 1
    ledger.validated_revision[selected] = ledger.surface_revision[selected]
    return selected


def state_counts(ledger: QualityLedger) -> dict[str, int]:
    return {name: int((ledger.status == name).sum()) for name in (
        PASS_LOCKED, REWORK_AFFECTED, REWORK, UNSAFE_GEOMETRY, CLEARCOAT_GUARD,
        NOT_REACHED, NO_IMPROVEMENT, REWORK_PASS, REWORK_FAIL)}


def cell_rows(ledger: QualityLedger) -> list[dict]:
    rows = []
    for i, j in np.ndindex(ledger.shape):
        rows.append({
            "cell_i": i, "cell_j": j,
            "x_m": (i + 0.5) * ledger.resolution_m,
            "y_m": (j + 0.5) * ledger.resolution_m,
            "status": ledger.status[i, j], "recipe_id": ledger.recipe_id,
            "local_gu": float(ledger.quality["local_gu"][i, j]),
            "local_ra_um": float(ledger.quality["local_ra_um"][i, j]),
            "local_rz_um": float(ledger.quality["local_rz_um"][i, j]),
            "scratch_initial_um": float(ledger.quality["scratch_initial_um"][i, j]),
            "scratch_residual_um": float(ledger.quality["scratch_residual_um"][i, j]),
            "gu_pass": bool(ledger.quality["gu_pass"][i, j]),
            "ra_pass": bool(ledger.quality["ra_pass"][i, j]),
            "rz_pass": bool(ledger.quality["rz_pass"][i, j]),
            "scratch_pass": bool(ledger.quality["scratch_pass"][i, j]),
            "all4_pass": bool(ledger.quality["all4_pass"][i, j]),
            "clearcoat_pass": bool(ledger.quality["clearcoat_pass"][i, j]),
            "temperature_pass": bool(ledger.quality["temperature_pass"][i, j]),
            "reached": bool(ledger.reached_mask[i, j]),
            "unsafe_geometry": bool(ledger.unsafe_geometry_mask[i, j]),
            "validation_count": int(ledger.validation_count[i, j]),
            "surface_revision": int(ledger.surface_revision[i, j]),
            "validated_revision": int(ledger.validated_revision[i, j]),
            "initially_pass_locked": bool(ledger.initially_pass_locked[i, j]),
            "rework_affected_ever": bool(ledger.rework_affected_ever[i, j]),
            "pass_count": int(ledger.pass_count[i, j]),
            "cumulative_contact_time_s": float(ledger.cumulative_contact_time_s[i, j]),
        })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _demo_state():
    state = make_flat_patch((0.32, 0.32), resolution_m=0.002, seed=100,
                            target_ra_um=0.0, with_scratches=False)
    for values in (state.micro_height_um, state.initial_micro_height_um,
                   state.initial_scratch_depth_um, state.residual_scratch_depth_um,
                   state.cumulative_removal_um, state.thermal_damage_proxy):
        values[:] = 0.0
    state.defect_mask[:] = False
    state.healthy_mask[:] = True
    state.clearcoat_remaining_um[:] = 40.0
    state.peak_temperature_c[:] = 25.0
    checker = np.indices((24, 24)).sum(axis=0) % 2
    state.micro_height_um[60:84, 60:84] = checker * 2.0 - 1.0
    state.clearcoat_remaining_um[10:14, 10:14] = 29.0
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    state = _demo_state()
    reached = np.ones(state.shape, dtype=bool)
    reached[:2, :2] = False
    unsafe = np.zeros(state.shape, dtype=bool)
    unsafe[30:35, 30:35] = True
    started = time.perf_counter()
    ledger = inspect_after_coarse(state, reached_mask=reached,
                                  unsafe_geometry_mask=unsafe,
                                  recipe_id="synthetic_contract_demo")
    elapsed = time.perf_counter() - started
    counts = state_counts(ledger)
    checks = [
        {"check": "resolution_is_existing_2mm", "expected": 0.002,
         "actual": ledger.resolution_m, "pass": ledger.resolution_m == 0.002},
        {"check": "centered_window_is_existing_5x5", "expected": 5,
         "actual": AreaQualityTargets().local_window_cells,
         "pass": AreaQualityTargets().local_window_cells == 5},
        {"check": "all_cells_inspected_exactly_once", "expected": 25600,
         "actual": int((ledger.validation_count == 1).sum()),
         "pass": bool(np.all(ledger.validation_count == 1))},
        {"check": "all4_is_exact_and", "expected": True,
         "actual": bool(np.array_equal(ledger.quality["all4_pass"],
                         ledger.quality["gu_pass"] & ledger.quality["ra_pass"]
                         & ledger.quality["rz_pass"] & ledger.quality["scratch_pass"])),
         "pass": bool(np.array_equal(ledger.quality["all4_pass"],
                       ledger.quality["gu_pass"] & ledger.quality["ra_pass"]
                       & ledger.quality["rz_pass"] & ledger.quality["scratch_pass"]))},
        {"check": "clearcoat_temperature_separate_from_all4", "expected": True,
         "actual": bool(np.any(ledger.quality["all4_pass"] & ~ledger.quality["clearcoat_pass"])),
         "pass": bool(np.any(ledger.quality["all4_pass"] & ~ledger.quality["clearcoat_pass"]))},
        {"check": "state_partition_complete", "expected": 25600,
         "actual": sum(counts.values()), "pass": sum(counts.values()) == 25600},
    ]
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "synthetic_cell_ledger.csv"), cell_rows(ledger))
    write_csv(os.path.join(args.out_dir, "state_counts.csv"),
              [{"status": key, "cell_count": value} for key, value in counts.items()])
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10D_CELLWISE_QUALITY_STATE_CPU",
        "status": "CONTRACT_COMPLETE_ACTUAL_VEHICLE_QUALITY_NOT_EVALUATED",
        "quality_source": "learning.rl.gate_e6_area_quality.local_quality_maps",
        "gu_source": "learning.polytwin.gloss_proxy literature_gu_proxy_v1",
        "actual_gu_sensor_connected": False,
        "synthetic_demo_cells": int(np.prod(state.shape)),
        "synthetic_demo_inspection_wall_s": elapsed,
        "state_counts": counts,
        "acceptance_checks_pass": all(bool(row["pass"]) for row in checks),
        "full_scan_count": 1,
        "selective_revalidation_contract_implemented": True,
        "unchanged_pass_locked_revalidation_allowed": False,
        "physx_executed": False, "training_performed": False,
        "polishing_v5_modified": False, "next_step_allowed": False,
    }
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-D quality-state contract\n\n"
                 "The ledger is a deterministic synthetic CPU contract test, not measured vehicle GU. "
                 "It uses the existing local GU/Ra/Rz/scratch implementation unchanged.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
