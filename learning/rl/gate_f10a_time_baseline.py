"""Gate F10-A CPU-only vehicle path/time baseline.

Only the existing C/SL/SR path arrays and rail_config.json are read. The
reported time is a nominal kinematic estimate, not a measured PhysX cycle.
Inspection time remains explicitly unknown.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

RAILS = ("C", "SL", "SR")
DEFAULTS = {
    "feed_mm_s": 12.7,
    "rail_speed_m_s": 0.0018 * 3.0 * 60.0,
    "control_hz": 60.0,
    "initial_home_s": 90.0 / 60.0,
    "slide_settle_s": 90.0 / 60.0,
    "approach_s": 240.0 / 60.0,
    "retract_s": 90.0 / 60.0,
    "turn_s": 0.0,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def segment_files(scan_dir: str, rail: str) -> list[tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(scan_dir, f"path_{rail}*.npy")):
        match = re.search(rf"path_{rail}(\d+)\.npy$", os.path.basename(path))
        if match:
            out.append((int(match.group(1)), path))
    return sorted(out)


def polyline_length_m(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def count_reversals(points: np.ndarray) -> int:
    if len(points) < 3:
        return 0
    delta = np.diff(points, axis=0)
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-9
    direction = delta[valid] / norm[valid]
    if len(direction) < 2:
        return 0
    return int((np.sum(direction[:-1] * direction[1:], axis=1) < 0.0).sum())


def runtime_order(indices: list[int], stops: list[list[float]]) -> list[int]:
    """Mirror current v5: segment 0 first, then nearest unfinished Y/Z stop."""
    if not indices:
        return []
    first = 0 if 0 in indices else indices[0]
    order, remaining = [first], set(indices) - {first}
    while remaining:
        current = stops[order[-1]]
        nxt = min(remaining, key=lambda i: (
            float(np.hypot(stops[i][0] - current[0], stops[i][1] - current[1])), i))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def rework_scenarios(rail_rows: list[dict], params: dict) -> list[dict]:
    rows = []
    for fraction in (0.10, 0.20, 0.30, 0.50):
        for fine_feed in (8.0, 12.7):
            for region_count in (1, 3, 5):
                fine_by_rail, totals = [], []
                for rail in rail_rows:
                    fine_s = (rail["path_length_m"] * fraction / (fine_feed / 1000.0)
                              + region_count * (params["approach_s"] + params["retract_s"]))
                    fine_by_rail.append(fine_s)
                    totals.append(rail["coarse_total_s"] + fine_s)
                rows.append({
                    "failed_path_fraction": fraction,
                    "fine_feed_mm_s": fine_feed,
                    "regions_per_active_rail": region_count,
                    "fine_serial_s": sum(fine_by_rail),
                    "coarse_plus_fine_serial_h": sum(totals) / 3600.0,
                    "coarse_plus_fine_parallel_h": max(totals) / 3600.0 if totals else 0.0,
                })
    return rows


def analyse(scan_dir: str, params: dict) -> tuple[list[dict], list[dict], dict]:
    with open(os.path.join(scan_dir, "rail_config.json"), encoding="utf-8") as fh:
        rail_cfg = json.load(fh)
    seg_rows, rail_rows = [], []
    feed_m_s = params["feed_mm_s"] / 1000.0
    for rail in RAILS:
        stops = rail_cfg.get(rail, {}).get("yz_stops", [])
        files = dict(segment_files(scan_dir, rail))
        order = runtime_order(list(files), stops)
        prev_stop = None
        totals = {"path_length_m": 0.0, "polish_s": 0.0, "turn_s": 0.0,
                  "rail_move_s": 0.0, "segment_overhead_s": 0.0}
        waypoints = reversals = 0
        for route_pos, idx in enumerate(order):
            points = np.load(files[idx]).astype(float)
            length = polyline_length_m(points)
            turns = count_reversals(points)
            stop = stops[idx]
            move_m = (0.0 if prev_stop is None else
                      float(np.hypot(stop[0] - prev_stop[0], stop[1] - prev_stop[1])))
            polish_s = length / feed_m_s
            turn_s = turns * params["turn_s"]
            move_s = move_m / params["rail_speed_m_s"]
            overhead_s = params["slide_settle_s"] + params["approach_s"] + params["retract_s"]
            seg_rows.append({
                "rail": rail, "route_order": route_pos, "segment": idx,
                "waypoints": len(points), "path_length_m": length,
                "reversals_proxy": turns, "rail_move_m": move_m,
                "polish_s": polish_s, "turn_dwell_s": turn_s,
                "rail_move_s": move_s, "segment_overhead_s": overhead_s,
                "segment_total_s_excluding_initial_home": polish_s + turn_s + move_s + overhead_s,
            })
            totals["path_length_m"] += length
            totals["polish_s"] += polish_s
            totals["turn_s"] += turn_s
            totals["rail_move_s"] += move_s
            totals["segment_overhead_s"] += overhead_s
            waypoints += len(points)
            reversals += turns
            prev_stop = stop
        coarse = (params["initial_home_s"] + totals["polish_s"] + totals["turn_s"]
                  + totals["rail_move_s"] + totals["segment_overhead_s"])
        rail_rows.append({
            "rail": rail, "segments": len(order), "waypoints": waypoints,
            "path_length_m": totals["path_length_m"], "reversals_proxy": reversals,
            "polish_s": totals["polish_s"], "turn_dwell_s": totals["turn_s"],
            "rail_move_s": totals["rail_move_s"],
            "initial_home_s": params["initial_home_s"],
            "segment_overhead_s": totals["segment_overhead_s"],
            "coarse_total_s": coarse,
        })
    serial = sum(row["coarse_total_s"] for row in rail_rows)
    parallel = max((row["coarse_total_s"] for row in rail_rows), default=0.0)
    summary = {
        "total_segments": sum(row["segments"] for row in rail_rows),
        "total_waypoints": sum(row["waypoints"] for row in rail_rows),
        "total_path_length_m": sum(row["path_length_m"] for row in rail_rows),
        "coarse_serial_s": serial, "coarse_serial_h": serial / 3600.0,
        "coarse_parallel_s": parallel, "coarse_parallel_h": parallel / 3600.0,
        "bottleneck_rail": max(rail_rows, key=lambda row: row["coarse_total_s"])["rail"],
    }
    summary["rework_scenarios"] = rework_scenarios(rail_rows, params)
    return seg_rows, rail_rows, summary


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", default="scan_result/car")
    parser.add_argument("--out-dir", required=True)
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=value)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    params = {key: getattr(args, key) for key in DEFAULTS}
    seg_rows, rail_rows, summary = analyse(args.scan_dir, params)
    os.makedirs(args.out_dir)
    write_csv(os.path.join(args.out_dir, "segment_time_baseline.csv"), seg_rows)
    write_csv(os.path.join(args.out_dir, "rail_time_baseline.csv"), rail_rows)
    write_csv(os.path.join(args.out_dir, "rework_scenarios.csv"), summary["rework_scenarios"])
    selected = [path for rail in RAILS for _, path in segment_files(args.scan_dir, rail)]
    selected.append(os.path.join(args.scan_dir, "rail_config.json"))
    write_csv(os.path.join(args.out_dir, "input_checksums.csv"),
              [{"path": os.path.abspath(path), "sha256": _sha256(path)} for path in selected])
    document = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10A_VEHICLE_TIME_BASELINE_V2",
        "scan_dir": os.path.abspath(args.scan_dir),
        "runtime_constants_from_local_polishing_v5": params,
        "measured_from_data": ["path_length_m", "waypoints", "rail_move_m"],
        "inspection_time": "UNKNOWN_NOT_INCLUDED",
        "limitations": [
            "nominal path/feed calculation; not a measured PhysX cycle",
            "runtime IK/collision/skip/contact delays are not modeled",
            "reversals are a geometric proxy and current turn dwell is zero",
            "rework fractions and region counts are scenarios, not measurements",
        ],
        "summary": summary,
        "physx_executed": False, "training_performed": False, "inputs_modified": False,
    }
    with open(os.path.join(args.out_dir, "time_baseline.json"), "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-A vehicle time baseline v2\n\n"
                 "CPU-only nominal estimate from the existing C/SL/SR paths. "
                 "Inspection and runtime delay are unknown, so this is not a confirmed total cycle time.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{_sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
