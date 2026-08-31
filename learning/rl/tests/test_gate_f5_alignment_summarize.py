"""CPU contract tests for frozen Gate F5 paired acceptance."""
from __future__ import annotations

from learning.rl.gate_f5_alignment_summarize import evaluate_pair
from learning.rl.gate_f_curved_physx_summarize import CRITERIA


def _base(kind: str, normal: bool) -> dict:
    row = {
        "candidate_id": kind,
        "surface_kind": kind,
        "normal_aligned_pad": normal,
        "roi_removal_mean_um": 0.21,
        "roi_coverage_fraction": 0.96,
        "force_sensor_filtered_steady_mean_n": 5.78,
        "force_tracking_abs_error_steady_p95_n": 1.0,
        "force_sensor_raw_max_n": 10.0,
        "gap_tracking_abs_error_steady_p95_m": 0.0037,
        "target_quaternion_norm_max_error": 1.0e-7,
        "target_quaternion_step_angle_max_deg": 0.2,
        "target_tilt_max_deg": 10.0 if kind != "flat" else 0.0,
        "normal_alignment_error_steady_p95_deg": 0.5 if kind != "flat" else 0.1,
    }
    for criterion in CRITERIA:
        metric, operator, threshold = (
            criterion["metric"], criterion["operator"], criterion["threshold"])
        row.setdefault(metric, threshold + 1.0 if operator == "gt" else threshold)
    return row


def test_curved_pair_passes_at_declared_values() -> None:
    rows = evaluate_pair(_base("sphere", False), _base("sphere", True), 4.0)
    assert all(row["passed"] for row in rows)


def test_alignment_ratio_failure_is_detected() -> None:
    normal = _base("sphere", True)
    normal["normal_alignment_error_steady_p95_deg"] = 1.1
    rows = evaluate_pair(_base("sphere", False), normal, 4.0)
    assert any(row["check"] == "curved_alignment_error_ratio" and not row["passed"]
               for row in rows)


def test_flat_parity_force_delta_failure_is_detected() -> None:
    normal = _base("flat", True)
    normal["force_sensor_filtered_steady_mean_n"] += 0.2
    rows = evaluate_pair(_base("flat", False), normal, 0.0)
    assert any(row["check"] == "flat_force_steady_mean_abs_delta_n" and not row["passed"]
               for row in rows)
