"""CPU tests for Gate F6 zero-shot summary helpers."""
from __future__ import annotations

from learning.rl.gate_f6_zero_shot_summarize import _all4_area, _key, _truth


def test_truth_parses_csv_booleans() -> None:
    assert _truth(True)
    assert _truth("True")
    assert _truth("true")
    assert not _truth("False")


def test_pair_key_uses_direction_profile_and_seed() -> None:
    row = {"direction_mode": "cross_xy", "surface_profile": "factory_prepolish",
           "profile_seed": "31097"}
    assert _key(row) == ("cross_xy", "factory_prepolish", 31097)


def test_all4_area_uses_exact_cell_weighting() -> None:
    rows = [
        {"cell_count": "400", "all4_pass_area_pct": "100"},
        {"cell_count": "400", "all4_pass_area_pct": "50"},
    ]
    passed, total, percent = _all4_area(rows)
    assert (passed, total, percent) == (600, 800, 75.0)
