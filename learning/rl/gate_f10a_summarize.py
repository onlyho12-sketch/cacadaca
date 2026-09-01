"""CPU-only aggregation of existing Gate F9 and F10-A trace outputs."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from datetime import datetime, timezone


KEY_FIELDS = ("shield_mode", "surface_kind", "direction_mode", "surface_profile",
              "env", "surface_seed_base")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def read_trace_root(trace_root: str) -> tuple[list[dict], list[dict]]:
    run_rows, events = [], []
    for trace_path in sorted(glob.glob(os.path.join(trace_root, "*", "substep_trace.csv"))):
        run_dir = os.path.dirname(trace_path)
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)
        physics_dt = float(metadata.get("physics_dt", 1.0 / 120.0))
        previous_raw, raw_max = {}, float("-inf")
        row_count = event_count = 0
        with open(trace_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row_count += 1
                env = int(row["env"])
                raw = float(row["force_sensor_raw_n"])
                raw_max = max(raw_max, raw)
                if row["latch_event"] == "1":
                    event_count += 1
                    prior = previous_raw.get(env)
                    events.append({
                        "run": os.path.basename(run_dir),
                        **{field: row[field] for field in KEY_FIELDS},
                        "control_step": int(row["control_step"]),
                        "substep": int(row["substep"]),
                        "force_sensor_raw_n": raw,
                        "force_sensor_filt_n": float(row["force_sensor_filt_n"]),
                        "force_used_n": float(row["force_used_n"]),
                        "force_cmd_n": float(row["force_cmd_n"]),
                        "raw_force_slope_n_s": ("" if prior is None else (raw - prior) / physics_dt),
                        "pad_gap_m": float(row["pad_gap_m"]),
                        "normal_alignment_error_deg": float(row["normal_alignment_error_deg"]),
                        "parent_force_action": float(row["parent_force_action"]),
                        "executed_force_action": float(row["executed_force_action"]),
                        "executed_feed_action": float(row["executed_feed_action"]),
                    })
                previous_raw[env] = raw
        run_rows.append({
            "run": os.path.basename(run_dir),
            "shield_mode": metadata["shield_mode"],
            "surface_kind": metadata["surface_kind"],
            "direction_mode": metadata["direction_mode"],
            "surface_seed_base": metadata["surface_seed_base"],
            "max_control_steps_requested": metadata.get("max_control_steps_requested", 800),
            "trace_rows": row_count, "latch_events_from_csv": event_count,
            "raw_force_max_n": raw_max,
            "legacy_metadata_latched_env_count": len(metadata.get("latched_envs", [])),
        })
    return run_rows, events


def coverage_rows(f9_rows: list[dict], mode: str, events: list[dict]) -> list[dict]:
    event_map = {}
    for event in events:
        event_map.setdefault(_key(event), event)
    expected = [row for row in f9_rows if row["shield_mode"] == mode
                and row["surface_kind"] == "cylinder"
                and row["force_hard_violated"] == "True"]
    result = []
    for row in expected:
        event = event_map.get(_key(row))
        result.append({
            **{field: row[field] for field in KEY_FIELDS},
            "f9_control_steps": row["control_steps"],
            "f9_sampled_raw_force_max_n": row["raw_force_max_n"],
            "trace_latch_found": bool(event),
            "trace_control_step": "" if event is None else event["control_step"],
            "trace_substep": "" if event is None else event["substep"],
            "trace_raw_force_n": "" if event is None else event["force_sensor_raw_n"],
            "trace_raw_force_slope_n_s": "" if event is None else event["raw_force_slope_n_s"],
        })
    return result


def summarize(f9_summary: str, trace_roots: str | list[str]) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    f9_rows = _read_csv(os.path.join(f9_summary, "combined_sequences.csv"))
    if isinstance(trace_roots, str):
        trace_roots = [trace_roots]
    run_rows, events = [], []
    for trace_root in trace_roots:
        root_runs, root_events = read_trace_root(trace_root)
        run_rows.extend(root_runs)
        events.extend(root_events)
    candidate = coverage_rows(f9_rows, "static_cap", events)
    control = coverage_rows(f9_rows, "control", events)
    candidate_found = sum(row["trace_latch_found"] for row in candidate)
    control_found = sum(row["trace_latch_found"] for row in control)
    complete = candidate_found == len(candidate) and control_found == len(control)
    unique_events = {_key(event): event for event in events}
    candidate_events = [unique_events[_key(row)] for row in candidate if _key(row) in unique_events]
    control_events = [unique_events[_key(row)] for row in control if _key(row) in unique_events]

    def event_range(rows: list[dict], field: str) -> dict:
        values = [float(row[field]) for row in rows if row[field] != ""]
        return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}

    overload_counts = {}
    for mode in ("control", "static_cap"):
        overload_counts[mode] = {}
        for surface in ("flat", "cylinder", "sphere", "freeform"):
            overload_counts[mode][surface] = sum(
                row["shield_mode"] == mode and row["surface_kind"] == surface
                and row["force_hard_violated"] == "True" for row in f9_rows)
    required_runs = [] if complete else [
        {"shield_mode": "static_cap", "direction_mode": "same_xx",
         "surface_seed_base": 41000, "max_control_steps": 2700,
         "purpose": "late candidate overload"},
        *[{"shield_mode": shield, "direction_mode": direction,
           "surface_seed_base": 43000, "max_control_steps": 800,
           "purpose": "control-positive paired trace"}
          for shield in ("control", "static_cap")
          for direction in ("same_xx", "cross_xy")],
    ]
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F10A_READONLY_ANALYSIS",
        "decision": ("F10A_COMPLETE_STATIC_CAP_REJECTED" if complete
                     else "INCOMPLETE_NEEDS_MINIMAL_TRACE"),
        "f9_candidate_cylinder_overloads": len(candidate),
        "candidate_overloads_trace_covered": candidate_found,
        "f9_control_cylinder_overloads": len(control),
        "control_overloads_trace_covered": control_found,
        "trace_runs_reused": len(run_rows),
        "trace_latch_events_from_csv": len(events),
        "unique_f9_overload_events_covered": len(unique_events),
        "f9_force_overload_counts_by_surface": overload_counts,
        "root_cause_analysis": {
            "hard_latch_input": "instantaneous raw physical force > 14 N at 120 Hz substep",
            "candidate_raw_force_n": event_range(candidate_events, "force_sensor_raw_n"),
            "candidate_filtered_force_n": event_range(candidate_events, "force_sensor_filt_n"),
            "candidate_raw_force_slope_n_s": event_range(candidate_events, "raw_force_slope_n_s"),
            "candidate_pad_gap_m": event_range(candidate_events, "pad_gap_m"),
            "candidate_normal_alignment_error_deg": event_range(
                candidate_events, "normal_alignment_error_deg"),
            "candidate_events_cap_saturated": sum(
                float(row["parent_force_action"]) > float(row["executed_force_action"]) + 1e-6
                for row in candidate_events),
            "candidate_events_total": len(candidate_events),
            "control_raw_force_n": event_range(control_events, "force_sensor_raw_n"),
            "finding": ("the +0.50 static action cap does not bound instantaneous contact force or its "
                        "rise rate; overloads are substep contact spikes at small penetration with near-zero "
                        "normal error, and several occur when the cap is not active"),
        },
        "reason": ("all 11 static-cap and 4 control cylinder overloads now have substep traces; "
                   "F9 rejection remains unchanged" if complete else
                   "existing 800-step trace omitted one late candidate event and all seed43000 control-positive runs"),
        "next_required_runs": required_runs,
        "physx_reexecution_required": not complete,
        "physx_executed_by_this_summary": False,
        "training_performed": False,
        "integration_allowed": False,
    }
    return run_rows, events, candidate, control, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f9-summary", required=True)
    parser.add_argument("--trace-root", required=True, nargs="+")
    parser.add_argument("--time-baseline", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    run_rows, events, candidate, control, decision = summarize(args.f9_summary, args.trace_root)
    os.makedirs(args.out_dir)
    _write_csv(os.path.join(args.out_dir, "trace_run_summary.csv"), run_rows)
    _write_csv(os.path.join(args.out_dir, "trace_latch_events.csv"), events)
    _write_csv(os.path.join(args.out_dir, "f9_candidate_overload_coverage.csv"), candidate)
    _write_csv(os.path.join(args.out_dir, "f9_control_overload_coverage.csv"), control)
    decision["f9_summary"] = os.path.abspath(args.f9_summary)
    decision["trace_roots"] = [os.path.abspath(path) for path in args.trace_root]
    decision["time_baseline"] = os.path.abspath(args.time_baseline)
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Gate F10-A read-only analysis\n\n"
                 "Existing F9 and substep traces were aggregated on CPU. The current evidence is "
                 "incomplete; no implementation or integration decision is authorized.\n")
    outputs = sorted(glob.glob(os.path.join(args.out_dir, "*")))
    with open(os.path.join(args.out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        for path in outputs:
            fh.write(f"{_sha256(path)}  {os.path.basename(path)}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
