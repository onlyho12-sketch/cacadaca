"""CPU tests for the Gate F10-C coarse-pass scheduler."""
from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f10_pass_scheduler as ps  # noqa: E402


def _row(i, risk=0.0, unsafe=False, continuous=True, step=0.01):
    return {"rail": "C", "segment": "0", "waypoint": str(i),
            "path_file": "path_C0.npy", "path_continuous": str(continuous),
            "path_step_m": str(0.0 if i == 0 else step), "geometry_risk": str(risk),
            "unsafe_geometry_pt_design": str(unsafe)}


def test_smoothstep_monotonic_and_bounded():
    values = [ps.smoothstep01(x / 10.0) for x in range(11)]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 1.0


def test_risk_derates_force_and_feed():
    cfg = ps.SchedulerConfig(entry_soft_start_m=0.0)
    low = ps.schedule_segment([_row(0, risk=0.0)], cfg)[0]
    high = ps.schedule_segment([_row(0, risk=1.0)], cfg)[0]
    assert float(high["force_command_n"]) < float(low["force_command_n"])
    assert float(high["feed_command_mm_s"]) < float(low["feed_command_mm_s"])


def test_entry_force_respects_existing_contact_advance_floor():
    out = ps.schedule_segment([_row(0, risk=1.0)], ps.SchedulerConfig())[0]
    assert float(out["force_command_n"]) >= 2.0


def test_unsafe_is_hold_with_zero_contact_commands():
    row = ps.schedule_segment([_row(0, unsafe=True)], ps.SchedulerConfig())[0]
    assert row["schedule_state"] == "HOLD_REVIEW"
    assert row["force_command_n"] == 0.0
    assert row["feed_command_mm_s"] == 0.0


def test_discontinuity_restarts_entry_group_without_deleting_point():
    rows = [_row(0), _row(1), _row(2, continuous=False, step=0.2), _row(3)]
    out = ps.schedule_segment(rows, ps.SchedulerConfig())
    assert len(out) == len(rows)
    assert out[0]["contact_group"] == 1
    assert out[2]["contact_group"] == 2
    assert out[2]["requires_reapproach"] is True
    assert out[2]["contact_path_step_m"] == 0.0


def test_force_slew_bound():
    previous, target, rate, dt = 2.0, 8.0, 2.0, 0.1
    assert ps._slew(previous, target, rate, dt) == pytest.approx(2.2)


def test_acceptance_preserves_source_order():
    source = [_row(0), _row(1)]
    schedule = ps.schedule_all(source, ps.SchedulerConfig())
    checks = ps.acceptance_checks(source, schedule, ps.SchedulerConfig())
    assert all(bool(row["pass"]) for row in checks)
