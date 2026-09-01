"""Unit tests for the Gate F10-A vehicle cycle-time baseline."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f10a_time_baseline as tb  # noqa: E402


def _make_scan(tmp_path, rails=("C",), segs=2, n=5):
    d = tmp_path / "scan"
    d.mkdir()
    cfg = {}
    for rail in rails:
        cfg[rail] = {"yz_stops": [[float(i), 0.0] for i in range(segs)]}
        for s in range(segs):
            # straight 1 m line along x, no reversals
            pts = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n), np.zeros(n)], axis=1)
            np.save(d / f"path_{rail}{s}.npy", pts.astype(np.float32))
    (d / "rail_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(d)


def test_polyline_length_and_reversals():
    straight = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    assert tb.polyline_length_m(straight) == pytest.approx(2.0)
    assert tb.count_reversals(straight) == 0
    back = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    assert tb.count_reversals(back) == 1
    assert tb.polyline_length_m(np.zeros((1, 3))) == 0.0


def test_feed_drives_polish_time(tmp_path):
    scan = _make_scan(tmp_path)
    params = dict(tb.DEFAULTS)
    params.update(turn_s=0.0, approach_s=0.0, retract_s=0.0,
                  slide_settle_s=0.0, initial_home_s=0.0, rail_speed_m_s=1e9)
    _seg, rails, summary = tb.analyse(scan, params)
    # two 1 m segments at 12.7 mm/s
    assert summary["total_path_length_m"] == pytest.approx(2.0)
    assert rails[0]["polish_s"] == pytest.approx(2.0 / 0.0127, rel=1e-6)


def test_parallel_is_bottleneck_rail(tmp_path):
    scan = _make_scan(tmp_path, rails=("C", "SL"), segs=2)
    _seg, rails, summary = tb.analyse(scan, dict(tb.DEFAULTS))
    assert summary["coarse_parallel_s"] == pytest.approx(
        max(r["coarse_total_s"] for r in rails))
    assert summary["coarse_serial_s"] == pytest.approx(
        sum(r["coarse_total_s"] for r in rails))
    assert summary["bottleneck_rail"] in ("C", "SL")


def test_rework_scenarios_increase_with_fraction(tmp_path):
    scan = _make_scan(tmp_path)
    _seg, _rails, s = tb.analyse(scan, dict(tb.DEFAULTS))
    scenarios = s["rework_scenarios"]
    def get(frac, feed, regions=1):
        return next(row for row in scenarios if row["failed_path_fraction"] == frac
                    and row["fine_feed_mm_s"] == feed
                    and row["regions_per_active_rail"] == regions)
    assert get(0.10, 8.0)["coarse_plus_fine_serial_h"] < get(0.50, 8.0)["coarse_plus_fine_serial_h"]
    assert get(0.20, 8.0)["coarse_plus_fine_serial_h"] > get(0.20, 12.7)["coarse_plus_fine_serial_h"]


def test_runtime_order_starts_zero_then_nearest():
    stops = [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    assert tb.runtime_order([0, 1, 2, 3], stops) == [0, 2, 3, 1]


def test_inputs_are_not_modified(tmp_path):
    scan = _make_scan(tmp_path)
    before = {f: os.path.getmtime(os.path.join(scan, f)) for f in os.listdir(scan)}
    tb.analyse(scan, dict(tb.DEFAULTS))
    after = {f: os.path.getmtime(os.path.join(scan, f)) for f in os.listdir(scan)}
    assert before == after
