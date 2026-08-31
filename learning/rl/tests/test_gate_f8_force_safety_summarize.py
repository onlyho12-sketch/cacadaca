import json
import os

import pytest

from learning.rl.gate_f8_force_safety_summarize import (
    CANDIDATES, PARENT_SHA256, aggregate, load_runs)


GEOMETRIES = ("flat", "cylinder", "sphere", "freeform")
PROFILES = ("legacy_stress", "factory_prepolish", "factory_prepolish_deep_defect",
            "factory_prepolish_deep_stress")


def _sequence(arm, kind, direction, profile, *, overload=False, faults=0, all4=95.0,
              quality=True, flat_delta=0.0):
    safe = not overload
    return {
        "shield_mode": arm, "surface_kind": kind, "direction_mode": direction,
        "surface_profile": profile, "env": 0, "profile_seed": "33000",
        "outcome": "fail_force_overload" if overload else "success",
        "quality_ok": str(quality), "safety_ok": str(safe),
        "force_hard_violated": str(overload), "thermal_hard_violated": "False",
        "unstable_hard_violated": "False", "control_steps": 3000 + int(flat_delta),
        "contact_steps": 2500, "sensor_fault_steps": faults,
        "raw_force_max_n": 13.0 if overload else 9.5,
        "normal_alignment_error_max_deg": 0.4,
        "parent_force_action_mean": 0.4, "executed_force_action_mean": 0.3,
        "parent_feed_action_mean": 0.1, "executed_feed_action_mean": 0.1,
        "shield_step_fraction": 0.0 if arm == "control" else 1.0,
        "force_action_cap_mean": 1.0 if arm == "control" else 0.5,
        "force_action_cap_min": 1.0 if arm == "control" else 0.5,
        "gu_final": 92.0, "ra_final_um": 0.03, "rz_final_um": 0.12,
        "scratch_final_um": 0.01, "temperature_peak_c": 41.0,
        "final_all4_pass_area_pct": all4,
    }


def _write_run(root, arm, kind, direction, rows, *, meta_overrides=None):
    out = os.path.join(root, f"{arm}_{kind}_{direction}")
    os.makedirs(out)
    fields = list(rows[0])
    with open(os.path.join(out, "sequences.csv"), "w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(str(row[f]) for f in fields) + "\n")
    with open(os.path.join(out, "tile_area_fractions.csv"), "w", encoding="utf-8") as handle:
        handle.write("shield_mode,surface_kind,direction_mode,surface_profile,"
                     "profile_seed,tile_i,tile_j,cell_count,all4_pass_area_pct\n")
        for row in rows:
            for tile in range(25):
                handle.write(f"{arm},{kind},{direction},{row['surface_profile']},33000,"
                             f"{tile},0,400,{row['final_all4_pass_area_pct']}\n")
    meta = {"gate": "F8_FORCE_SAFETY", "shield_mode": arm, "surface_kind": kind,
            "direction_mode": direction, "training_performed": False,
            "observation_mode": "base14", "checkpoint_sha256": PARENT_SHA256,
            "all_finite": True, "completed": len(rows), "expected": len(rows),
            "physics_seed": 20260901, "policy_seed": 20260831, "surface_seed_base": 33000}
    meta.update(meta_overrides or {})
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    return out


def _arm_dirs(root, arm, direction, *, overload_kinds=(), faults=0, all4=95.0,
              quality=True, flat_overload=False):
    dirs = []
    for kind in GEOMETRIES:
        rows = []
        for profile in PROFILES:
            bad = kind in overload_kinds or (kind == "flat" and flat_overload)
            rows.append(_sequence(arm, kind, direction, profile, overload=bad,
                                  faults=faults if kind != "flat" else 0,
                                  all4=all4 if kind != "flat" else 96.0,
                                  quality=quality if kind != "flat" else True))
        dirs.append(_write_run(root, arm, kind, direction, rows))
    return dirs


def _stage1_dirs(tmp_path, **candidate_kwargs):
    root = str(tmp_path)
    dirs = _arm_dirs(root, "control", "same_xx", overload_kinds=("cylinder", "freeform"))
    dirs += _arm_dirs(root, "static_cap", "same_xx", **candidate_kwargs.get("static_cap", {}))
    dirs += _arm_dirs(root, "predictive_shield", "same_xx",
                      **candidate_kwargs.get("predictive_shield", {}))
    return dirs


def test_stage1_enters_both_candidates_and_prefers_less_intrusive(tmp_path):
    dirs = _stage1_dirs(tmp_path)
    decision = aggregate(dirs, str(tmp_path / "out"), "1")
    assert decision["decision"] == "PASS"
    assert decision["entered_candidates"] == list(CANDIDATES)
    assert decision["selected_candidate"] == "static_cap"
    assert decision["control_entry_force_overloads"] == 8


def test_stage1_tiebreak_prefers_better_quality(tmp_path):
    dirs = _stage1_dirs(tmp_path, predictive_shield={"all4": 99.0})
    decision = aggregate(dirs, str(tmp_path / "out"), "1")
    assert decision["selected_candidate"] == "predictive_shield"


def test_stage1_rejects_candidate_with_sensor_faults(tmp_path):
    dirs = _stage1_dirs(tmp_path, static_cap={"faults": 3})
    decision = aggregate(dirs, str(tmp_path / "out"), "1")
    assert decision["decision"] == "FAIL"
    assert decision["entered_candidates"] == ["predictive_shield"]
    assert decision["selected_candidate"] == "predictive_shield"


def test_stage1_rejects_candidate_without_overload_reduction(tmp_path):
    dirs = _stage1_dirs(tmp_path, static_cap={"overload_kinds": ("cylinder", "freeform")})
    decision = aggregate(dirs, str(tmp_path / "out"), "1")
    assert "static_cap" not in decision["entered_candidates"]


def test_stage1_flat_parity_mismatch_blocks_entry(tmp_path):
    dirs = _stage1_dirs(tmp_path, static_cap={"flat_overload": True})
    decision = aggregate(dirs, str(tmp_path / "out"), "1")
    assert "static_cap" not in decision["entered_candidates"]
    parity = (tmp_path / "out" / "flat_passthrough_parity.csv").read_text()
    assert "outcome" in parity


def _final_dirs(tmp_path, candidate="static_cap", **candidate_kwargs):
    root = str(tmp_path)
    dirs = []
    for direction in ("same_xx", "cross_xy"):
        dirs += _arm_dirs(root, "control", direction, overload_kinds=("cylinder", "freeform"))
        dirs += _arm_dirs(root, candidate, direction, **candidate_kwargs)
    return dirs


def test_final_gate_pass(tmp_path):
    dirs = _final_dirs(tmp_path)
    decision = aggregate(dirs, str(tmp_path / "out"), "final", "static_cap")
    assert decision["decision"] == "PASS"
    assert decision["candidate_force_overloads"] == 0
    assert decision["mean_curved_all4_area_drop_pp"] == pytest.approx(0.0)


def test_final_gate_fails_on_quality_drop(tmp_path):
    dirs = _final_dirs(tmp_path, all4=93.0)
    decision = aggregate(dirs, str(tmp_path / "out"), "final", "static_cap")
    assert decision["decision"] == "FAIL"
    assert decision["mean_curved_all4_area_drop_pp"] == pytest.approx(2.0)


def test_final_gate_fails_on_candidate_overload(tmp_path):
    dirs = _final_dirs(tmp_path, overload_kinds=("sphere",))
    decision = aggregate(dirs, str(tmp_path / "out"), "final", "static_cap")
    assert decision["decision"] == "FAIL"
    assert decision["candidate_force_overloads"] == 8


def test_load_runs_rejects_foreign_checkpoint(tmp_path):
    rows = [_sequence("control", "flat", "same_xx", profile) for profile in PROFILES]
    path = _write_run(str(tmp_path), "control", "flat", "same_xx", rows,
                      meta_overrides={"checkpoint_sha256": "0" * 64})
    with pytest.raises(ValueError, match="parent checkpoint"):
        load_runs([path])


def test_load_runs_rejects_training_runs(tmp_path):
    rows = [_sequence("control", "flat", "same_xx", profile) for profile in PROFILES]
    path = _write_run(str(tmp_path), "control", "flat", "same_xx", rows,
                      meta_overrides={"training_performed": True})
    with pytest.raises(ValueError, match="training_performed"):
        load_runs([path])


def test_aggregate_refuses_to_overwrite(tmp_path):
    dirs = _stage1_dirs(tmp_path)
    out = str(tmp_path / "out")
    aggregate(dirs, out, "1")
    with pytest.raises(FileExistsError):
        aggregate(dirs, out, "1")


def test_incomplete_arm_is_rejected(tmp_path):
    dirs = _stage1_dirs(tmp_path)
    with pytest.raises(ValueError, match="unique sequences|paired keys"):
        aggregate(dirs[:-1], str(tmp_path / "out2"), "1")
