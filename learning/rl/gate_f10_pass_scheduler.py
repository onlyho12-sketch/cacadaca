"""Gate F10-C CPU-only coarse-pass scheduler for existing C/SL/SR paths.

The scheduler consumes F10-B waypoint geometry and emits a deterministic,
continuous force/feed plan.  It never generates, deletes, reorders or writes
vehicle path arrays.  All limits introduced here are isolated PT-DESIGN values
pending later PhysX validation.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

RAILS = ("C", "SL", "SR")


@dataclass(frozen=True)
class SchedulerConfig:
    base_feed_mm_s: float = 12.7
    minimum_feed_ratio: float = 0.50
    top_flat_force_n: float = 8.0
    top_steep_force_n: float = 5.0
    side_flat_force_n: float = 6.0
    side_steep_force_n: float = 3.5
    risk_lookaround_m: float = 0.055
    entry_soft_start_m: float = 0.040
    entry_force_ratio: float = 0.35
    entry_force_floor_n: float = 2.0
    entry_feed_ratio: float = 0.50
    force_slew_n_s: float = 2.0
    feed_slew_mm_s2: float = 25.0
    path_continuity_max_step_m: float = 0.080
    control_hz: float = 60.0
    approach_s: float = 4.0
    retract_s: float = 1.5
    slide_settle_s: float = 1.5
    initial_home_s: float = 1.5


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def risk_envelope(rows: list[dict], radius_m: float) -> np.ndarray:
    """Non-causal pad-radius max envelope, split at path discontinuities."""
    n = len(rows)
    result = np.zeros(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and _truth(rows[end]["path_continuous"]):
            end += 1
        steps = np.array([0.0] + [float(rows[i]["path_step_m"]) for i in range(start + 1, end)])
        arc = np.cumsum(steps)
        risk = np.array([float(rows[i]["geometry_risk"]) for i in range(start, end)])
        for local_i, distance in enumerate(arc):
            result[start + local_i] = float(risk[np.abs(arc - distance) <= radius_m].max())
        start = end
    return result


def _slew(previous: float, target: float, rate_per_s: float, dt_s: float) -> float:
    delta = rate_per_s * max(dt_s, 1e-9)
    return float(np.clip(target, previous - delta, previous + delta))


def schedule_segment(rows: list[dict], cfg: SchedulerConfig) -> list[dict]:
    if not rows:
        return []
    envelope = risk_envelope(rows, cfg.risk_lookaround_m)
    output = []
    contact_group = 0
    in_contact = False
    entry_distance = 0.0
    previous_force = previous_feed = None
    for index, (source, risk_max) in enumerate(zip(rows, envelope)):
        unsafe = _truth(source["unsafe_geometry_pt_design"])
        discontinuity = index > 0 and not _truth(source["path_continuous"])
        if unsafe:
            output.append({
                **source, "source_row_index": index,
                "risk_envelope": risk_max, "schedule_state": "HOLD_REVIEW",
                "contact_group": "", "requires_reapproach": False,
                "entry_progress": 0.0, "force_target_unlimited_n": 0.0,
                "force_command_n": 0.0, "feed_target_unlimited_mm_s": 0.0,
                "feed_command_mm_s": 0.0, "contact_path_step_m": 0.0,
                "nominal_contact_dt_s": 0.0,
            })
            in_contact = False
            previous_force = previous_feed = None
            continue
        requires_reapproach = not in_contact or discontinuity
        if requires_reapproach:
            contact_group += 1
            entry_distance = 0.0
            previous_force = previous_feed = None
        step_m = 0.0 if requires_reapproach else float(source["path_step_m"])
        entry_distance += step_m
        entry_progress = (1.0 if cfg.entry_soft_start_m <= 0.0 else
                          float(np.clip(entry_distance / cfg.entry_soft_start_m, 0.0, 1.0)))
        shaped_risk = smoothstep01(risk_max)
        is_side = source["rail"] in ("SL", "SR")
        flat_force = cfg.side_flat_force_n if is_side else cfg.top_flat_force_n
        steep_force = cfg.side_steep_force_n if is_side else cfg.top_steep_force_n
        risk_force = flat_force + (steep_force - flat_force) * shaped_risk
        risk_feed = cfg.base_feed_mm_s * (1.0 - (1.0 - cfg.minimum_feed_ratio) * shaped_risk)
        entry_force_scale = cfg.entry_force_ratio + (1.0 - cfg.entry_force_ratio) * entry_progress
        entry_feed_scale = cfg.entry_feed_ratio + (1.0 - cfg.entry_feed_ratio) * entry_progress
        force_target = max(cfg.entry_force_floor_n, risk_force * entry_force_scale)
        feed_target = max(cfg.base_feed_mm_s * cfg.minimum_feed_ratio,
                          risk_feed * entry_feed_scale)
        if previous_force is None:
            force_command, feed_command = force_target, feed_target
            dt_s = 1.0 / cfg.control_hz
        else:
            mean_feed_m_s = max(1e-6, 0.5 * (previous_feed + feed_target) / 1000.0)
            dt_s = max(1.0 / cfg.control_hz, step_m / mean_feed_m_s)
            force_command = _slew(previous_force, force_target, cfg.force_slew_n_s, dt_s)
            feed_command = _slew(previous_feed, feed_target, cfg.feed_slew_mm_s2, dt_s)
        output.append({
            **source, "source_row_index": index,
            "risk_envelope": risk_max,
            "schedule_state": "POLISH_ENTRY" if entry_progress < 1.0 else "POLISH",
            "contact_group": contact_group,
            "requires_reapproach": requires_reapproach,
            "entry_progress": entry_progress,
            "force_target_unlimited_n": force_target,
            "force_command_n": force_command,
            "feed_target_unlimited_mm_s": feed_target,
            "feed_command_mm_s": feed_command,
            "contact_path_step_m": step_m,
            "nominal_contact_dt_s": dt_s,
        })
        in_contact = True
        previous_force, previous_feed = force_command, feed_command
    return output


def schedule_all(geometry_rows: list[dict], cfg: SchedulerConfig) -> list[dict]:
    result = []
    keys = []
    for row in geometry_rows:
        key = (row["rail"], int(row["segment"]))
        if key not in keys:
            keys.append(key)
    for rail, segment in keys:
        selected = [row for row in geometry_rows
                    if row["rail"] == rail and int(row["segment"]) == segment]
        selected.sort(key=lambda row: int(row["waypoint"]))
        result.extend(schedule_segment(selected, cfg))
    return result


def summarize(schedule: list[dict], time_rail_rows: list[dict], cfg: SchedulerConfig
              ) -> tuple[list[dict], list[dict], dict]:
    segment_rows, rail_rows = [], []
    for rail in RAILS:
        rail_selected = [row for row in schedule if row["rail"] == rail]
        for segment in sorted(set(int(row["segment"]) for row in rail_selected)):
            selected = [row for row in rail_selected if int(row["segment"]) == segment]
            polish = [row for row in selected if row["schedule_state"].startswith("POLISH")]
            groups = {int(row["contact_group"]) for row in polish}
            segment_rows.append({
                "rail": rail, "segment": segment, "waypoints": len(selected),
                "polish_waypoints": len(polish),
                "hold_review_waypoints": len(selected) - len(polish),
                "contact_groups": len(groups),
                "contact_path_length_m": sum(float(row["contact_path_step_m"]) for row in polish),
                "nominal_contact_time_s": sum(float(row["nominal_contact_dt_s"]) for row in polish),
                "force_command_min_n": min((float(row["force_command_n"]) for row in polish), default=0.0),
                "force_command_max_n": max((float(row["force_command_n"]) for row in polish), default=0.0),
                "feed_command_min_mm_s": min((float(row["feed_command_mm_s"]) for row in polish), default=0.0),
                "feed_command_max_mm_s": max((float(row["feed_command_mm_s"]) for row in polish), default=0.0),
            })
        time_row = next(row for row in time_rail_rows if row["rail"] == rail)
        segments = [row for row in segment_rows if row["rail"] == rail]
        groups = sum(int(row["contact_groups"]) for row in segments)
        contact_s = sum(float(row["nominal_contact_time_s"]) for row in segments)
        overhead_s = (cfg.initial_home_s + float(time_row["rail_move_s"])
                      + len(segments) * cfg.slide_settle_s
                      + groups * (cfg.approach_s + cfg.retract_s))
        rail_rows.append({
            "rail": rail, "segments": len(segments), "waypoints": len(rail_selected),
            "polish_waypoints": sum(int(row["polish_waypoints"]) for row in segments),
            "hold_review_waypoints": sum(int(row["hold_review_waypoints"]) for row in segments),
            "contact_groups": groups,
            "contact_path_length_m": sum(float(row["contact_path_length_m"]) for row in segments),
            "nominal_contact_time_s": contact_s,
            "nominal_overhead_s": overhead_s,
            "nominal_executable_subset_total_s": contact_s + overhead_s,
        })
    parallel = max(float(row["nominal_executable_subset_total_s"]) for row in rail_rows)
    serial = sum(float(row["nominal_executable_subset_total_s"]) for row in rail_rows)
    summary = {
        "waypoints": len(schedule),
        "polish_waypoints": sum(row["schedule_state"].startswith("POLISH") for row in schedule),
        "hold_review_waypoints": sum(row["schedule_state"] == "HOLD_REVIEW" for row in schedule),
        "contact_groups": sum(int(row["contact_groups"]) for row in rail_rows),
        "requires_reapproach_count": sum(_truth(row["requires_reapproach"]) for row in schedule),
        "nominal_executable_subset_parallel_h": parallel / 3600.0,
        "nominal_executable_subset_serial_h": serial / 3600.0,
        "bottleneck_rail": max(rail_rows, key=lambda row: float(
            row["nominal_executable_subset_total_s"]))["rail"],
        "workload_max_min_ratio": max(float(row["nominal_executable_subset_total_s"]) for row in rail_rows)
        / min(float(row["nominal_executable_subset_total_s"]) for row in rail_rows),
        "full_vehicle_time_status": "UNKNOWN_UNTIL_HOLD_REVIEW_WAYPOINTS_RESOLVED",
    }
    return segment_rows, rail_rows, summary


def acceptance_checks(source: list[dict], schedule: list[dict], cfg: SchedulerConfig) -> list[dict]:
    polish = [row for row in schedule if row["schedule_state"].startswith("POLISH")]
    order_ok = all((a["rail"], a["segment"], a["waypoint"], a["path_file"])
                   == (b["rail"], b["segment"], b["waypoint"], b["path_file"])
                   for a, b in zip(source, schedule))
    hold_zero = all(float(row["force_command_n"]) == 0.0 and float(row["feed_command_mm_s"]) == 0.0
                    for row in schedule if row["schedule_state"] == "HOLD_REVIEW")
    finite = all(np.isfinite(float(row[field])) for row in schedule for field in
                 ("risk_envelope", "force_command_n", "feed_command_mm_s",
                  "contact_path_step_m", "nominal_contact_dt_s"))
    force_range = all(cfg.entry_force_floor_n <= float(row["force_command_n"])
                      <= (cfg.side_flat_force_n if row["rail"] in ("SL", "SR")
                          else cfg.top_flat_force_n) for row in polish)
    feed_range = all(cfg.base_feed_mm_s * cfg.minimum_feed_ratio <= float(row["feed_command_mm_s"])
                     <= cfg.base_feed_mm_s for row in polish)
    force_slew_ok = feed_slew_ok = True
    for previous, current in zip(schedule, schedule[1:]):
        same_group = (previous["schedule_state"].startswith("POLISH")
                      and current["schedule_state"].startswith("POLISH")
                      and previous["rail"] == current["rail"]
                      and previous["segment"] == current["segment"]
                      and previous["contact_group"] == current["contact_group"])
        if same_group:
            dt_s = float(current["nominal_contact_dt_s"])
            force_slew_ok &= (abs(float(current["force_command_n"])
                                  - float(previous["force_command_n"]))
                              <= cfg.force_slew_n_s * dt_s + 1e-9)
            feed_slew_ok &= (abs(float(current["feed_command_mm_s"])
                                 - float(previous["feed_command_mm_s"]))
                             <= cfg.feed_slew_mm_s2 * dt_s + 1e-9)
    unsafe_hold_exact = sum(_truth(row["unsafe_geometry_pt_design"]) for row in source) == sum(
        row["schedule_state"] == "HOLD_REVIEW" for row in schedule)
    return [
        {"check": "source_row_count_preserved", "expected": len(source), "actual": len(schedule),
         "pass": len(source) == len(schedule)},
        {"check": "path_order_preserved", "expected": True, "actual": order_ok, "pass": order_ok},
        {"check": "hold_review_zero_contact_command", "expected": True,
         "actual": hold_zero, "pass": hold_zero},
        {"check": "finite_schedule", "expected": True, "actual": finite, "pass": finite},
        {"check": "force_within_local_v5_anchors", "expected": True,
         "actual": force_range, "pass": force_range},
        {"check": "feed_within_existing_policy_range", "expected": "6.35..12.7",
         "actual": feed_range, "pass": feed_range},
        {"check": "force_slew_within_candidate_limit", "expected": True,
         "actual": force_slew_ok, "pass": force_slew_ok},
        {"check": "feed_slew_within_candidate_limit", "expected": True,
         "actual": feed_slew_ok, "pass": feed_slew_ok},
        {"check": "unsafe_mask_maps_exactly_to_hold", "expected": True,
         "actual": unsafe_hold_exact, "pass": unsafe_hold_exact},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-result", required=True)
    parser.add_argument("--time-baseline", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    cfg = SchedulerConfig()
    geometry_csv = os.path.join(args.geometry_result, "waypoint_geometry.csv")
    time_csv = os.path.join(args.time_baseline, "rail_time_baseline.csv")
    geometry_rows = read_csv(geometry_csv)
    time_rows = read_csv(time_csv)
    schedule = schedule_all(geometry_rows, cfg)
    segment_rows, rail_rows, summary = summarize(schedule, time_rows, cfg)
    checks = acceptance_checks(geometry_rows, schedule, cfg)
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "waypoint_pass_schedule.csv"), schedule)
    write_csv(os.path.join(args.out_dir, "segment_schedule_summary.csv"), segment_rows)
    write_csv(os.path.join(args.out_dir, "rail_schedule_summary.csv"), rail_rows)
    write_csv(os.path.join(args.out_dir, "acceptance_checks.csv"), checks)
    inputs = [geometry_csv, os.path.join(args.geometry_result, "decision.json"), time_csv,
              os.path.join(args.time_baseline, "time_baseline.json")]
    write_csv(os.path.join(args.out_dir, "input_checksums.csv"),
              [{"path": os.path.abspath(path), "sha256": sha256(path)} for path in inputs])
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10C_COARSE_PASS_SCHEDULER_CPU",
        "status": "SCHEDULE_BUILT_HOLD_REVIEW_UNRESOLVED",
        "configuration_pt_design": asdict(cfg),
        "summary": summary,
        "acceptance_checks_pass": all(_truth(check["pass"]) for check in checks),
        "original_path_arrays_modified": False,
        "physx_executed": False, "training_performed": False,
        "polishing_v5_modified": False, "next_step_allowed": False,
        "limitations": [
            "force/feed/slew values are PT-DESIGN candidates pending PhysX validation",
            "HOLD_REVIEW waypoints are excluded from the nominal executable-subset time",
            "existing rail ownership and waypoint order are preserved; no load rebalancing mutation",
        ],
    }
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-C coarse-pass scheduler\n\n"
                 "CPU-only schedule over unchanged C/SL/SR waypoint order. HOLD_REVIEW points "
                 "must be resolved before any complete-vehicle time or execution claim.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
