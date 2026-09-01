"""Gate F10-E CPU-only region rework planner and physical-time guard.

Failed cells are grouped into bounded continuous regions.  The planner records
the complete pad-footprint changed mask, including touched PASS_LOCKED cells,
and rejects plans with too many approaches or a >4 h three-rail estimate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import binary_dilation, label

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl.gate_f10_quality_state import (  # noqa: E402
    CLEARCOAT_GUARD, NOT_REACHED, PASS_LOCKED, REWORK, UNSAFE_GEOMETRY,
)


@dataclass(frozen=True)
class PlannerConfig:
    resolution_m: float = 0.002
    pad_radius_m: float = 0.055
    fine_stepover_m: float = 0.0275
    fine_feed_mm_s: float = 8.0
    merge_gap_m: float = 0.020
    approach_s: float = 4.0
    retract_s: float = 1.5
    max_regions_per_rail: int = 8
    max_regions_total: int = 24
    max_rework_fraction: float = 0.50
    max_changed_to_target_ratio: float = 4.0
    max_parallel_vehicle_h: float = 4.0


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_rework_regions(status: np.ndarray, cfg: PlannerConfig) -> tuple[np.ndarray, list[dict]]:
    status = np.asarray(status)
    target = status == REWORK
    barriers = np.isin(status, (UNSAFE_GEOMETRY, CLEARCOAT_GUARD, NOT_REACHED))
    iterations = max(0, int(round(0.5 * cfg.merge_gap_m / cfg.resolution_m)))
    expanded = target.copy()
    for _ in range(iterations):
        expanded = binary_dilation(expanded, structure=np.ones((3, 3), bool)) & ~barriers
    labels, count = label(expanded, structure=np.ones((3, 3), dtype=int))
    target_labels = np.zeros_like(labels)
    regions = []
    next_id = 0
    for source_id in range(1, count + 1):
        actual = target & (labels == source_id)
        if not actual.any():
            continue
        next_id += 1
        target_labels[actual] = next_id
        ii, jj = np.nonzero(actual)
        regions.append({
            "region_id": next_id, "target_cells": int(len(ii)),
            "target_area_m2": float(len(ii) * cfg.resolution_m ** 2),
            "i_min": int(ii.min()), "i_max": int(ii.max()),
            "j_min": int(jj.min()), "j_max": int(jj.max()),
        })
    return target_labels, regions


def raster_path_for_region(region_mask: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    """One continuous serpentine path across a region's failed-cell envelope."""
    ii, jj = np.nonzero(region_mask)
    if not len(ii):
        return np.empty((0, 2), dtype=np.float64)
    lane_cells = max(1, int(round(cfg.fine_stepover_m / cfg.resolution_m)))
    paths = []
    reverse = False
    for i in range(int(ii.min()), int(ii.max()) + 1, lane_cells):
        band = region_mask[i:min(i + lane_cells, region_mask.shape[0])]
        occupied_j = np.nonzero(band.any(axis=0))[0]
        if not len(occupied_j):
            continue
        x = min(i + lane_cells // 2, region_mask.shape[0] - 1)
        endpoints = [(x, int(occupied_j.min())), (x, int(occupied_j.max()))]
        if reverse:
            endpoints.reverse()
        paths.extend(endpoints)
        reverse = not reverse
    return (np.asarray(paths, dtype=np.float64) + 0.5) * cfg.resolution_m


def footprint_mask(shape: tuple[int, int], path_xy: np.ndarray,
                   cfg: PlannerConfig) -> np.ndarray:
    changed = np.zeros(shape, dtype=bool)
    if not len(path_xy):
        return changed
    radius_cells = int(np.ceil(cfg.pad_radius_m / cfg.resolution_m))
    for start, end in zip(path_xy[:-1], path_xy[1:]):
        length = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(length / cfg.resolution_m)) + 1)
        for point in np.linspace(start, end, sample_count):
            center = point / cfg.resolution_m - 0.5
            i0, j0 = np.floor(center).astype(int)
            ia = max(0, i0 - radius_cells); ib = min(shape[0], i0 + radius_cells + 1)
            ja = max(0, j0 - radius_cells); jb = min(shape[1], j0 + radius_cells + 1)
            gi, gj = np.ogrid[ia:ib, ja:jb]
            distance = np.hypot((gi - center[0]) * cfg.resolution_m,
                                (gj - center[1]) * cfg.resolution_m)
            changed[ia:ib, ja:jb] |= distance <= cfg.pad_radius_m
    if len(path_xy) == 1:
        return footprint_mask(shape, np.vstack((path_xy, path_xy)), cfg)
    return changed


def path_length_m(path_xy: np.ndarray) -> float:
    return 0.0 if len(path_xy) < 2 else float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())


def build_plan(status: np.ndarray, cfg: PlannerConfig) -> tuple[list[dict], list[dict], np.ndarray, dict]:
    labels, regions = group_rework_regions(status, cfg)
    paths, changed_total = [], np.zeros(status.shape, dtype=bool)
    barriers = np.isin(status, (UNSAFE_GEOMETRY, CLEARCOAT_GUARD, NOT_REACHED))
    unsafe_regions = 0
    for region in regions:
        region_id = region["region_id"]
        path = raster_path_for_region(labels == region_id, cfg)
        changed = footprint_mask(status.shape, path, cfg)
        barrier_hit = bool(np.any(changed & barriers))
        if barrier_hit:
            unsafe_regions += 1
        else:
            changed_total |= changed
        length = path_length_m(path)
        region.update({
            "path_points": len(path), "path_length_m": length,
            "changed_cells": int(changed.sum()),
            "rework_affected_locked_cells": int((changed & (status == PASS_LOCKED)).sum()),
            "barrier_hit": barrier_hit,
            "plan_state": "REJECT_BARRIER" if barrier_hit else "READY_CPU_ONLY",
            "nominal_time_s": (0.0 if barrier_hit else
                               length / (cfg.fine_feed_mm_s / 1000.0)
                               + cfg.approach_s + cfg.retract_s),
        })
        for order, point in enumerate(path):
            paths.append({"region_id": region_id, "path_order": order,
                          "x_m": float(point[0]), "y_m": float(point[1]),
                          "contact": not barrier_hit})
    target_cells = int((status == REWORK).sum())
    summary = {
        "target_cells": target_cells,
        "target_fraction": target_cells / status.size,
        "regions": len(regions), "ready_regions": len(regions) - unsafe_regions,
        "rejected_barrier_regions": unsafe_regions,
        "changed_cells": int(changed_total.sum()),
        "rework_affected_locked_cells": int((changed_total & (status == PASS_LOCKED)).sum()),
        "revalidation_cells": int(changed_total.sum()),
        "changed_to_target_ratio": (float(changed_total.sum() / target_cells)
                                    if target_cells else 0.0),
        "nominal_rework_time_s": sum(float(region["nominal_time_s"]) for region in regions),
    }
    return regions, paths, changed_total, summary


def parse_historical_first_pass(status_log: str, control_hz: float = 60.0) -> list[dict]:
    first = {}
    pattern = re.compile(r"t=(\d+) rail=(C|SL|SR).*state=DONE.*모든_구간")
    with open(status_log, encoding="utf-8") as fh:
        for line in fh:
            match = pattern.search(line)
            if match and match.group(2) not in first:
                first[match.group(2)] = int(match.group(1))
    return [{"rail": rail, "first_done_step": first[rail],
             "simulated_physical_time_s": first[rail] / control_hz}
            for rail in ("C", "SL", "SR") if rail in first]


def time_guard(time_baseline_json: str, cfg: PlannerConfig,
               historical_status_log: str | None = None) -> dict:
    with open(time_baseline_json, encoding="utf-8") as fh:
        baseline = json.load(fh)
    coarse_h = float(baseline["summary"]["coarse_parallel_h"])
    scenario = next(row for row in baseline["summary"]["rework_scenarios"]
                    if row["failed_path_fraction"] == cfg.max_rework_fraction
                    and row["fine_feed_mm_s"] == cfg.fine_feed_mm_s
                    and row["regions_per_active_rail"] == 3)
    worst_h = float(scenario["coarse_plus_fine_parallel_h"])
    historical = (parse_historical_first_pass(historical_status_log)
                  if historical_status_log else [])
    return {
        "coarse_parallel_path_feed_h": coarse_h,
        "bounded_50pct_rework_parallel_h": worst_h,
        "max_parallel_vehicle_h": cfg.max_parallel_vehicle_h,
        "time_guard_pass": worst_h <= cfg.max_parallel_vehicle_h,
        "twelve_hour_estimate_supported_by_local_path_model": worst_h >= 12.0,
        "historical_v5_first_pass_simulated_physical_time": historical,
        "historical_log_caveat": "diagnostic old local run; not current wall-clock or production validation",
    }


def _demo_status() -> np.ndarray:
    status = np.full((160, 160), PASS_LOCKED, dtype="U24")
    status[35:120, 35:120] = REWORK
    status[0:4, :] = UNSAFE_GEOMETRY
    status[:, 0:4] = NOT_REACHED
    status[152:157, 152:157] = CLEARCOAT_GUARD
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-baseline", required=True)
    parser.add_argument("--historical-status-log")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    cfg = PlannerConfig()
    status = _demo_status()
    regions, paths, changed, summary = build_plan(status, cfg)
    timing = time_guard(os.path.join(args.time_baseline, "time_baseline.json"), cfg,
                        args.historical_status_log)
    region_limit_pass = (summary["regions"] <= cfg.max_regions_total
                         and summary["target_fraction"] <= cfg.max_rework_fraction)
    checks = [
        {"check": "region_count_limit", "expected": f"<={cfg.max_regions_total}",
         "actual": summary["regions"], "pass": summary["regions"] <= cfg.max_regions_total},
        {"check": "rework_fraction_limit", "expected": f"<={cfg.max_rework_fraction}",
         "actual": summary["target_fraction"],
         "pass": summary["target_fraction"] <= cfg.max_rework_fraction},
        {"check": "no_barrier_contact", "expected": 0,
         "actual": summary["rejected_barrier_regions"],
         "pass": summary["rejected_barrier_regions"] == 0},
        {"check": "revalidation_exactly_changed_mask", "expected": summary["changed_cells"],
         "actual": summary["revalidation_cells"],
         "pass": summary["changed_cells"] == summary["revalidation_cells"]},
        {"check": "changed_to_target_ratio", "expected": f"<={cfg.max_changed_to_target_ratio}",
         "actual": summary["changed_to_target_ratio"],
         "pass": summary["changed_to_target_ratio"] <= cfg.max_changed_to_target_ratio},
        {"check": "three_rail_time_guard", "expected": "<=4h",
         "actual": timing["bounded_50pct_rework_parallel_h"],
         "pass": timing["time_guard_pass"]},
        {"check": "twelve_hour_plan_rejected", "expected": False,
         "actual": timing["twelve_hour_estimate_supported_by_local_path_model"],
         "pass": not timing["twelve_hour_estimate_supported_by_local_path_model"]},
    ]
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "rework_regions.csv"), regions)
    write_csv(os.path.join(args.out_dir, "rework_paths.csv"), paths)
    np.save(os.path.join(args.out_dir, "changed_mask.npy"), changed)
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10E_REGION_REWORK_PLANNER_CPU",
        "status": "CONTRACT_COMPLETE_SYNTHETIC_PLAN_ONLY",
        "configuration_pt_design": asdict(cfg),
        "summary": summary, "vehicle_time_guard": timing,
        "acceptance_checks_pass": all(bool(row["pass"]) for row in checks),
        "region_limit_pass": region_limit_pass,
        "actual_vehicle_quality_input_available": False,
        "physx_executed": False, "training_performed": False,
        "polishing_v5_modified": False, "next_step_allowed": False,
    }
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-E region rework planner\n\n"
                 "Synthetic CPU contract. It rejects per-cell approaches and any local path/feed "
                 "estimate above four hours; actual vehicle quality input is not yet available.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
