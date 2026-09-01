"""CPU-only tests for Gate F10-A trace coverage aggregation."""
from __future__ import annotations

import csv
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f10a_summarize as summary  # noqa: E402


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_coverage_uses_csv_event_not_legacy_metadata(tmp_path):
    trace_root = tmp_path / "traces"
    run = trace_root / "static_cap_cylinder_same_xx_seed1"
    run.mkdir(parents=True)
    metadata = {"shield_mode": "static_cap", "surface_kind": "cylinder",
                "direction_mode": "same_xx", "surface_seed_base": 1,
                "physics_dt": 0.01, "latched_envs": []}
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    base = {"shield_mode": "static_cap", "surface_kind": "cylinder",
            "direction_mode": "same_xx", "surface_profile": "profile", "env": "2",
            "surface_seed_base": "1", "control_step": "3", "substep": "0",
            "force_sensor_raw_n": "5", "force_sensor_filt_n": "4", "force_used_n": "4",
            "force_cmd_n": "6", "pad_gap_m": "0.01", "normal_alignment_error_deg": "2",
            "parent_force_action": "0.1", "executed_force_action": "0.2",
            "executed_feed_action": "0.3"}
    _write_csv(run / "substep_trace.csv", [dict(base, latch_event="0"),
                                             dict(base, force_sensor_raw_n="15", substep="1", latch_event="1")])
    runs, events = summary.read_trace_root(str(trace_root))
    assert runs[0]["legacy_metadata_latched_env_count"] == 0
    assert runs[0]["latch_events_from_csv"] == 1
    assert events[0]["raw_force_slope_n_s"] == 1000.0


def test_expected_coverage_key():
    row = {field: "x" for field in summary.KEY_FIELDS}
    row.update(shield_mode="static_cap", surface_kind="cylinder",
               force_hard_violated="True", control_steps="9", raw_force_max_n="8")
    event = {field: row[field] for field in summary.KEY_FIELDS}
    event.update(control_step=4, substep=2, force_sensor_raw_n=15,
                 raw_force_slope_n_s=100)
    covered = summary.coverage_rows([row], "static_cap", [event])
    assert covered[0]["trace_latch_found"] is True
