"""Unit tests for the frozen Gate F9 multi-seed aggregation."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f9_summarize as f9  # noqa: E402


CONTROL_OVERLOADS = {("cylinder", "40000", "legacy_stress"),
                     ("freeform", "41000", "legacy_stress")}


def _row(arm: str, kind: str, dm: str, base: int, profile: str) -> dict[str, str]:
    row = {
        "shield_mode": arm, "surface_kind": kind, "direction_mode": dm,
        "surface_profile": profile, "env": "0", "profile_seed": str(base),
        "outcome": "success", "quality_ok": "True", "safety_ok": "True",
        "force_hard_violated": "False", "thermal_hard_violated": "False",
        "unstable_hard_violated": "False", "control_steps": "4000",
        "contact_steps": "3900", "sensor_fault_steps": "0", "raw_force_max_n": "9.5",
        "normal_alignment_error_max_deg": "0.5", "parent_force_action_mean": "0.7",
        "executed_force_action_mean": "0.5", "parent_feed_action_mean": "0.1",
        "executed_feed_action_mean": "0.1", "shield_step_fraction": "0.0",
        "force_action_cap_mean": "1.0", "force_action_cap_min": "1.0",
        "gu_final": "73.0", "ra_final_um": "0.10", "rz_final_um": "1.1",
        "scratch_final_um": "0.2", "temperature_peak_c": "45.0",
        "final_all4_pass_area_pct": "95.0",
    }
    if kind != "flat":
        if arm == f9.CANDIDATE:
            row.update({"final_all4_pass_area_pct": "94.8", "shield_step_fraction": "0.2",
                        "force_action_cap_mean": "0.5", "force_action_cap_min": "0.5"})
        elif (kind, str(base), profile) in CONTROL_OVERLOADS:
            row.update({"outcome": "fail_force_overload", "safety_ok": "False",
                        "force_hard_violated": "True", "control_steps": "1200"})
    return row


def _meta(arm: str, kind: str, dm: str, base: int, sha: str = f9.PARENT_SHA256) -> dict:
    return {"gate": "F8_FORCE_SAFETY", "shield_mode": arm, "surface_kind": kind,
            "direction_mode": dm, "surface_seed_base": base, "physics_seed": 20260901,
            "policy_seed": 20260831, "checkpoint_sha256": sha,
            "observation_mode": "base14", "training_performed": False,
            "all_finite": True, "completed": 4, "expected": 4}


def _write_run(directory: str, meta: dict, rows: list[dict[str, str]]) -> None:
    os.makedirs(directory)
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    with open(os.path.join(directory, "sequences.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tile_fields = ("shield_mode", "surface_kind", "direction_mode", "surface_profile",
                   "env", "profile_seed", "tile_id")
    with open(os.path.join(directory, "tile_area_fractions.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tile_fields)
        writer.writeheader()
        for row in rows:
            for tile in range(f9.TILES_PER_SEQUENCE):
                writer.writerow({field: row[field] for field in tile_fields[:-1]}
                                | {"tile_id": str(tile)})


def build_all(root, seq_mutator=None, meta_sha=f9.PARENT_SHA256) -> list[str]:
    run_dirs = []
    for base in f9.SEED_BASES:
        for dm in f9.DIRECTIONS:
            for kind in f9.GEOMETRIES:
                for arm in f9.ARMS:
                    rows = [_row(arm, kind, dm, base, profile) for profile in f9.PROFILES]
                    if seq_mutator:
                        seq_mutator(rows, arm, kind, dm, base)
                    directory = os.path.join(
                        str(root), f"gate_f9_{arm}_{kind}_{dm.split('_')[0]}_seed{base}")
                    _write_run(directory, _meta(arm, kind, dm, base, meta_sha), rows)
                    run_dirs.append(directory)
    return run_dirs


def _plan(root) -> str:
    path = os.path.join(str(root), "acceptance_criteria.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"gate": f9.GATE, "frozen_before_execution": True,
                   "parent_sha256": f9.PARENT_SHA256,
                   "surface_seed_bases": list(f9.SEED_BASES)}, handle)
    return path


def test_aggregate_pass(tmp_path):
    run_dirs = build_all(tmp_path / "runs")
    decision = f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])
    assert decision["decision"] == "PASS"
    assert decision["integration_candidate"] is True
    assert decision["sequences"] == 256 and decision["tile_rows"] == 6400
    assert decision["control_entry_curved_safety_pass"] == 60
    assert decision["candidate_entry_curved_safety_pass"] == 64
    assert decision["candidate_force_overloads"] == 0
    assert decision["control_force_overloads"] == 4
    for name in ("decision.json", "acceptance_checks.csv", "censoring_summary.csv",
                 "seed_summary.csv", "flat_passthrough_parity.csv", "checksums.sha256"):
        assert os.path.isfile(tmp_path / "out" / name)
    with open(tmp_path / "out" / "censoring_summary.csv", newline="") as handle:
        censoring = {row["surface_kind"]: row for row in csv.DictReader(handle)}
    assert censoring["cylinder"]["censored_pairs"] == "2"
    assert censoring["freeform"]["censored_pairs"] == "2"
    assert censoring["flat"]["censored_pairs"] == "0"
    assert censoring["cylinder"]["uncensored_pairs"] == "30"


def test_candidate_overload_fails(tmp_path):
    def mutate(rows, arm, kind, dm, base):
        if arm == f9.CANDIDATE and kind == "cylinder" and dm == "same_xx" and base == 42000:
            rows[0].update({"force_hard_violated": "True", "safety_ok": "False"})
    run_dirs = build_all(tmp_path / "runs", mutate)
    decision = f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])
    assert decision["decision"] == "FAIL"
    assert decision["candidate_force_overloads"] == 1


def test_no_control_failure_fails_by_design(tmp_path):
    def mutate(rows, arm, kind, dm, base):
        for row in rows:
            row.update({"outcome": "success", "safety_ok": "True",
                        "force_hard_violated": "False", "control_steps": "4000"})
    run_dirs = build_all(tmp_path / "runs", mutate)
    decision = f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])
    assert decision["decision"] == "FAIL"
    assert decision["control_entry_curved_safety_pass"] == 64
    assert decision["candidate_entry_curved_safety_pass"] == 64


def test_flat_parity_mismatch_fails(tmp_path):
    def mutate(rows, arm, kind, dm, base):
        if arm == f9.CANDIDATE and kind == "flat" and dm == "cross_xy" and base == 41000:
            rows[1]["gu_final"] = "72.9"
    run_dirs = build_all(tmp_path / "runs", mutate)
    decision = f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])
    assert decision["decision"] == "FAIL"


def test_curved_all4_drop_fails(tmp_path):
    def mutate(rows, arm, kind, dm, base):
        if arm == f9.CANDIDATE and kind != "flat":
            for row in rows:
                row["final_all4_pass_area_pct"] = "93.5"
    run_dirs = build_all(tmp_path / "runs", mutate)
    decision = f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])
    assert decision["decision"] == "FAIL"
    assert decision["mean_curved_all4_area_drop_pp"] == pytest.approx(1.5)


def test_wrong_checkpoint_hash_rejected(tmp_path):
    run_dirs = build_all(tmp_path / "runs", meta_sha="0" * 64)
    with pytest.raises(ValueError, match="checkpoint hash"):
        f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])


def test_missing_run_rejected(tmp_path):
    run_dirs = build_all(tmp_path / "runs")[:-1]
    with pytest.raises(ValueError, match="64 official runs"):
        f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])


def test_broken_seed_pairing_rejected(tmp_path):
    def mutate(rows, arm, kind, dm, base):
        if arm == f9.CANDIDATE and kind == "sphere" and dm == "same_xx" and base == 43000:
            rows[2]["profile_seed"] = "43999"
    run_dirs = build_all(tmp_path / "runs", mutate)
    with pytest.raises(ValueError, match="pair"):
        f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [])


def test_duplicate_verified_and_mismatch_rejected(tmp_path):
    run_dirs = build_all(tmp_path / "runs")
    duplicate = str(tmp_path / "dup_run")
    shutil.copytree(run_dirs[0], duplicate)
    decision = f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out"), [duplicate])
    assert decision["decision"] == "PASS"
    assert decision["duplicates_excluded"] == 1
    assert decision["sequences"] == 256
    with open(os.path.join(duplicate, "sequences.csv"), "a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="differs from official"):
        f9.aggregate(_plan(tmp_path), run_dirs, str(tmp_path / "out2"), [duplicate])


def test_refuses_to_overwrite(tmp_path):
    run_dirs = build_all(tmp_path / "runs")
    out = tmp_path / "out"
    os.makedirs(out)
    with pytest.raises(FileExistsError):
        f9.aggregate(_plan(tmp_path), run_dirs, str(out), [])
