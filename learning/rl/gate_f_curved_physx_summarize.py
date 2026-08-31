"""Aggregate Gate F4 curved PhysX diagnostics against frozen criteria."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any


EXPECTED_CANDIDATES = {
    "flat_cross": "flat",
    "cylinder_r0p60_cross": "cylinder",
    "sphere_r0p60_cross": "sphere",
    "freeform_s0_cross": "freeform",
}

# These PT-DESIGN diagnostic thresholds are frozen before any F4 PhysX run.
CRITERIA: tuple[dict[str, Any], ...] = (
    {"metric": "complete", "operator": "eq", "threshold": True},
    {"metric": "all_finite", "operator": "eq", "threshold": True},
    {"metric": "trace_all_finite", "operator": "eq", "threshold": True},
    {"metric": "force_hard_violated", "operator": "eq", "threshold": False},
    {"metric": "thermal_hard_violated", "operator": "eq", "threshold": False},
    {"metric": "unstable_hard_violated", "operator": "eq", "threshold": False},
    {"metric": "sensor_fault_control_steps", "operator": "eq", "threshold": 0},
    {"metric": "fallback_steps", "operator": "eq", "threshold": 0},
    {"metric": "no_contact_removal_errors", "operator": "eq", "threshold": 0},
    {"metric": "raw_hard_force_control_steps", "operator": "eq", "threshold": 0},
    {"metric": "post_first_contact_rate", "operator": "ge", "threshold": 0.95},
    {"metric": "force_sensor_filtered_steady_mean_n", "operator": "ge", "threshold": 2.0},
    {"metric": "force_sensor_filtered_steady_mean_n", "operator": "le", "threshold": 12.0},
    {"metric": "force_sensor_filtered_steady_std_n", "operator": "le", "threshold": 2.0},
    {"metric": "force_tracking_abs_error_steady_mean_n", "operator": "le", "threshold": 1.5},
    {"metric": "force_tracking_abs_error_steady_p95_n", "operator": "le", "threshold": 4.0},
    {"metric": "force_sensor_raw_max_n", "operator": "le", "threshold": 14.0},
    {"metric": "gap_tracking_abs_error_steady_p95_m", "operator": "le", "threshold": 0.004},
    {"metric": "roi_removal_mean_um", "operator": "gt", "threshold": 0.0},
    {"metric": "roi_coverage_fraction", "operator": "gt", "threshold": 0.0},
)


def _compare(observed: Any, operator: str, threshold: Any) -> bool:
    if observed is None:
        return False
    if operator == "eq":
        return observed == threshold
    if operator == "ge":
        return observed >= threshold
    if operator == "gt":
        return observed > threshold
    if operator == "le":
        return observed <= threshold
    raise ValueError(f"unsupported operator: {operator}")


def evaluate_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every frozen check; missing or null observations fail."""
    rows = []
    for index, criterion in enumerate(CRITERIA, start=1):
        metric = criterion["metric"]
        observed = summary.get(metric)
        rows.append({
            "candidate_id": summary.get("candidate_id", ""),
            "check_index": index,
            "metric": metric,
            "operator": criterion["operator"],
            "threshold": criterion["threshold"],
            "observed": observed,
            "passed": _compare(observed, criterion["operator"], criterion["threshold"]),
        })
    return rows


def frozen_criteria_document() -> dict[str, Any]:
    return {
        "gate": "F4",
        "status": "FROZEN_BEFORE_PHYSX_RUNS",
        "scope": "PT-DESIGN synthetic curved PhysX diagnostic; not real-cell validation",
        "candidate_rule": EXPECTED_CANDIDATES,
        "all_candidates_must_pass_all_checks": True,
        "criteria": list(CRITERIA),
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(input_dirs: list[str], out_dir: str) -> dict[str, Any]:
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    summaries = []
    sources = []
    for input_dir in input_dirs:
        summary_path = os.path.join(os.path.abspath(input_dir), "summary.json")
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append(summary)
        sources.append({"path": summary_path, "sha256": _sha256(summary_path)})
    candidate_ids = [row.get("candidate_id") for row in summaries]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ids must be unique")
    if set(candidate_ids) != set(EXPECTED_CANDIDATES):
        raise ValueError(
            f"expected candidate ids {sorted(EXPECTED_CANDIDATES)}, got {sorted(candidate_ids)}"
        )
    for summary in summaries:
        expected_kind = EXPECTED_CANDIDATES[summary["candidate_id"]]
        if summary.get("surface_kind") != expected_kind:
            raise ValueError(f"surface kind mismatch for {summary['candidate_id']}")

    check_rows = [check for summary in summaries for check in evaluate_summary(summary)]
    candidate_rows = []
    for summary in sorted(summaries, key=lambda row: row["candidate_id"]):
        checks = evaluate_summary(summary)
        candidate_rows.append({
            **summary,
            "checks_passed": sum(bool(row["passed"]) for row in checks),
            "checks_total": len(checks),
            "f4_candidate_pass": all(bool(row["passed"]) for row in checks),
        })
    overall_pass = all(row["f4_candidate_pass"] for row in candidate_rows)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "F4",
        "decision": "PASS" if overall_pass else "FAIL",
        "candidate_count": len(candidate_rows),
        "candidates_passed": sum(bool(row["f4_candidate_pass"]) for row in candidate_rows),
        "all_candidates_pass": overall_pass,
        "training_performed": False,
        "metric_status": "PT-DESIGN_SYNTHETIC_CURVED_PHYSX_DIAGNOSTIC",
        "next_step": (
            "F5 requires separate user approval; do not start automatically."
            if overall_pass else
            "Diagnose failed frozen checks in Gate F new files only; do not start F5."
        ),
        "sources": sources,
    }
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "acceptance_criteria.json"), "w", encoding="utf-8") as handle:
        json.dump(frozen_criteria_document(), handle, indent=2, sort_keys=True)
    _write_csv(os.path.join(out_dir, "candidate_summary.csv"), candidate_rows)
    _write_csv(os.path.join(out_dir, "acceptance_checks.csv"), check_rows)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.input_dir, os.path.abspath(args.out_dir)), indent=2))


if __name__ == "__main__":
    main()
