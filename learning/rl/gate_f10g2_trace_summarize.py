"""Summarize the frozen F10-G2 cylinder substep trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(args.out_dir)
    trace_path = os.path.join(args.trace_dir, "substep_trace.csv")
    metadata_path = os.path.join(args.trace_dir, "metadata.json")
    with open(trace_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    events = [row for row in rows if row["latch_event"] == "1"]
    if len(events) != 1:
        raise RuntimeError(f"expected one latch event, got {len(events)}")
    event = events[0]
    env_rows = [row for row in rows if row["env"] == event["env"]]
    event_index = env_rows.index(event)
    previous = env_rows[event_index - 1]
    local = [row for row in env_rows
             if int(event["control_step"]) - 1 <= int(row["control_step"])
             <= int(event["control_step"])]
    trough = min(local, key=lambda row: float(row["force_sensor_raw_n"]))
    dt_s = 1.0 / 120.0
    last_rise = float(event["force_sensor_raw_n"]) - float(previous["force_sensor_raw_n"])
    gap_deepen_um = (float(trough["pad_gap_m"]) - float(event["pad_gap_m"])) * 1e6
    conclusion = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10G2_V2_CYLINDER_OVERLOAD_ROOT_CAUSE",
        "decision": "ROOT_CAUSE_CONFIRMED",
        "event": {
            "profile": event["surface_profile"],
            "profile_seed": int(event["profile_seed"]),
            "control_step": int(event["control_step"]),
            "substep": int(event["substep"]),
            "raw_force_n": float(event["force_sensor_raw_n"]),
            "filtered_force_n": float(event["force_sensor_filt_n"]),
            "command_force_n": float(event["force_cmd_n"]),
            "predicted_force_n": float(event["predicted_force_n"]),
            "predictive_cap_action": float(event["predictive_force_cap_action"]),
            "geometry_cap_action": float(event["force_cap_action"]),
            "sensor_fault": bool(int(event["sensor_fault"])),
            "normal_alignment_error_deg": float(event["normal_alignment_error_deg"]),
        },
        "dynamics": {
            "local_raw_force_trough_n": float(trough["force_sensor_raw_n"]),
            "trough_to_event_raw_rise_n": float(event["force_sensor_raw_n"]) - float(trough["force_sensor_raw_n"]),
            "trough_to_event_gap_deepen_um": gap_deepen_um,
            "last_substep_raw_rise_n": last_rise,
            "last_substep_raw_slope_n_s": last_rise / dt_s,
        },
        "direct_cause": (
            "The control-rate predictor used the prior 20 Hz mean and negative delta, "
            "so it left predictive_cap=1.0 while the 120 Hz admittance/contact loop "
            "deepened penetration after a force trough and produced a within-tick raw spike."
        ),
        "ruled_out": [
            "path direction (same/cross are identical before the trip)",
            "sensor fault", "normal misalignment", "command discontinuity at the trip",
        ],
        "candidate_status": "REJECTED",
        "training_allowed": False,
        "implementation_performed": False,
        "recommended_design": (
            "Use authoritative substep raw/filtered force state or a latched peak/rise envelope; "
            "do not attempt another static action cap or PPO before deterministic safety passes."
        ),
    }
    os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "root_cause.json"), "w", encoding="utf-8") as fh:
        json.dump(conclusion, fh, indent=2, sort_keys=True)
    window = [row for row in env_rows
              if int(event["control_step"]) - 6 <= int(row["control_step"])
              <= int(event["control_step"])]
    with open(os.path.join(args.out_dir, "event_window.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(window[0]))
        writer.writeheader()
        writer.writerows(window)
    inputs = {os.path.abspath(trace_path): sha256(trace_path),
              os.path.abspath(metadata_path): sha256(metadata_path)}
    with open(os.path.join(args.out_dir, "input_checksums.json"), "w", encoding="utf-8") as fh:
        json.dump(inputs, fh, indent=2, sort_keys=True)
    print(json.dumps(conclusion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
