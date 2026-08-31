"""CPU contract tests for Gate F4 frozen diagnostic acceptance."""
from __future__ import annotations

from learning.rl.gate_f_curved_physx_summarize import CRITERIA, evaluate_summary


def _passing_summary() -> dict:
    summary = {"candidate_id": "flat_cross"}
    for criterion in CRITERIA:
        metric = criterion["metric"]
        operator = criterion["operator"]
        threshold = criterion["threshold"]
        if operator == "gt":
            summary[metric] = float(threshold) + 1.0
        else:
            summary[metric] = threshold
    return summary


def test_all_boundary_values_pass_frozen_contract() -> None:
    rows = evaluate_summary(_passing_summary())
    assert len(rows) == len(CRITERIA)
    assert all(row["passed"] for row in rows)


def test_missing_contact_metric_fails_closed() -> None:
    summary = _passing_summary()
    summary["force_tracking_abs_error_steady_p95_n"] = None
    rows = evaluate_summary(summary)
    failed = [row for row in rows if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["metric"] == "force_tracking_abs_error_steady_p95_n"


def test_each_safety_violation_fails() -> None:
    for metric in (
        "force_hard_violated",
        "thermal_hard_violated",
        "unstable_hard_violated",
    ):
        summary = _passing_summary()
        summary[metric] = True
        assert not all(row["passed"] for row in evaluate_summary(summary))
