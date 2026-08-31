"""Aggregate Gate F9 multi-seed paired validation against the frozen plan.

Applies the acceptance criteria frozen in
``gate_f9_multi_seed_plan_20260901_072500/acceptance_criteria.json`` to the 64
Gate F9 evaluation runs (control vs static_cap, 4 geometries x 2 directions x
4 independent surface seed bases).  No threshold in this file may be edited
after PhysX results exist.  Reuses the seed-agnostic helpers of the frozen F8
summarizer; the F8 module itself is not modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl.gate_f8_force_safety_summarize import (  # noqa: E402
    CONTROL, CURVED, ENTRY_CURVED, FLAT_PARITY_FIELDS, GEOMETRIES,
    PARENT_SHA256, PROFILES, _Checks, _key, _read_csv, _stats, _truth,
    _write_csv, flat_parity_rows, geometry_rows, paired_all4_drop_pp)

GATE = "F9_MULTI_SEED_PAIRED_VALIDATION"
CANDIDATE = "static_cap"
ARMS = (CONTROL, CANDIDATE)
DIRECTIONS = ("same_xx", "cross_xy")
SEED_BASES = (40000, 41000, 42000, 43000)
FIXED_SEEDS = {"physics_seed": 20260901, "policy_seed": 20260831}
SEQUENCES_PER_RUN = 4
TILES_PER_SEQUENCE = 25
EXPECTED_RUNS = 64
EXPECTED_SEQUENCES = 256
EXPECTED_TILE_ROWS = 6400
MAX_CURVED_ALL4_DROP_PP = 1.0
HARD_FIELDS = ("force_hard_violated", "thermal_hard_violated", "unstable_hard_violated")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metadata(directory: str) -> dict[str, Any]:
    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("gate") != "F8_FORCE_SAFETY":
        raise ValueError(f"{directory}: not a force-safety evaluation run")
    if meta.get("training_performed") is not False:
        raise ValueError(f"{directory}: training_performed must be false")
    if meta.get("observation_mode") != "base14":
        raise ValueError(f"{directory}: observation_mode must be base14")
    if meta.get("checkpoint_sha256") != PARENT_SHA256:
        raise ValueError(f"{directory}: unexpected parent checkpoint hash")
    if meta.get("all_finite") is not True:
        raise ValueError(f"{directory}: metadata reports non-finite outputs")
    if int(meta.get("completed", -1)) != int(meta.get("expected", -2)):
        raise ValueError(f"{directory}: incomplete run")
    for field, expected in FIXED_SEEDS.items():
        if int(meta.get(field, -1)) != expected:
            raise ValueError(f"{directory}: {field} must be {expected}")
    if int(meta.get("surface_seed_base", -1)) not in SEED_BASES:
        raise ValueError(f"{directory}: surface_seed_base must be one of {SEED_BASES}")
    if meta.get("shield_mode") not in ARMS:
        raise ValueError(f"{directory}: shield_mode must be one of {ARMS}")
    if meta.get("direction_mode") not in DIRECTIONS:
        raise ValueError(f"{directory}: unexpected direction_mode")
    if meta.get("surface_kind") not in GEOMETRIES:
        raise ValueError(f"{directory}: unexpected surface_kind")
    return meta


def _config_key(meta: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(meta["shield_mode"]), str(meta["surface_kind"]),
            str(meta["direction_mode"]), int(meta["surface_seed_base"]))


def load_runs(run_dirs: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]],
                                            list[dict[str, Any]]]:
    sequences: list[dict[str, str]] = []
    tiles: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for directory in run_dirs:
        meta = _load_metadata(directory)
        config = _config_key(meta)
        if config in seen:
            raise ValueError(f"{directory}: duplicate configuration {config} in official runs")
        seen.add(config)
        meta["run_dir"] = directory
        metadata.append(meta)
        base = str(meta["surface_seed_base"])
        rows = _read_csv(os.path.join(directory, "sequences.csv"))
        if len(rows) != SEQUENCES_PER_RUN:
            raise ValueError(f"{directory}: expected {SEQUENCES_PER_RUN} sequences, got {len(rows)}")
        for row in rows:
            row["surface_seed_base"] = base
        sequences.extend(rows)
        tile_rows = _read_csv(os.path.join(directory, "tile_area_fractions.csv"))
        if len(tile_rows) != SEQUENCES_PER_RUN * TILES_PER_SEQUENCE:
            raise ValueError(f"{directory}: expected "
                             f"{SEQUENCES_PER_RUN * TILES_PER_SEQUENCE} tile rows, "
                             f"got {len(tile_rows)}")
        for row in tile_rows:
            row["surface_seed_base"] = base
        tiles.extend(tile_rows)
    if len(metadata) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} official runs, got {len(metadata)}")
    return sequences, tiles, metadata


def validate_pairing(sequences: list[dict[str, str]]) -> None:
    expected_configs = {(arm, kind, dm, str(base)) for arm in ARMS for kind in GEOMETRIES
                        for dm in DIRECTIONS for base in SEED_BASES}
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in sequences:
        grouped.setdefault((row["shield_mode"], row["surface_kind"],
                            row["direction_mode"], row["surface_seed_base"]), []).append(row)
    if set(grouped) != expected_configs:
        raise ValueError("official sequences do not cover exactly the 64 frozen configurations")
    for config, rows in grouped.items():
        if sorted(row["surface_profile"] for row in rows) != sorted(PROFILES):
            raise ValueError(f"{config}: profiles incomplete")
    for arm in ARMS:
        keys = [_key(row) for row in sequences if row["shield_mode"] == arm]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{arm}: pair keys are not unique; profile seeds collide")
    control_keys = {_key(row) for row in sequences if row["shield_mode"] == CONTROL}
    candidate_keys = {_key(row) for row in sequences if row["shield_mode"] == CANDIDATE}
    if control_keys != candidate_keys:
        raise ValueError("control/static_cap pair keys do not match")
    for kind, dm, base in ((k, d, str(b)) for k in GEOMETRIES for d in DIRECTIONS
                           for b in SEED_BASES):
        pairs = {}
        for row in sequences:
            if (row["surface_kind"], row["direction_mode"], row["surface_seed_base"]) == \
                    (kind, dm, base):
                pairs.setdefault((row["surface_profile"], row["shield_mode"]),
                                 row["profile_seed"])
        for profile in PROFILES:
            if pairs.get((profile, CONTROL)) != pairs.get((profile, CANDIDATE)):
                raise ValueError(f"seed pairing broken for {kind}/{dm}/base {base}/{profile}")


def _hard_violated(row: dict[str, str]) -> bool:
    return any(_truth(row[field]) for field in HARD_FIELDS)


def censoring_rows(sequences: list[dict[str, str]]) -> list[dict[str, Any]]:
    control = {_key(row): row for row in sequences if row["shield_mode"] == CONTROL}
    rows = []
    for kind in GEOMETRIES:
        deltas, censored, control_hard, candidate_hard = [], 0, 0, 0
        control_steps_all, candidate_steps_all, pair_count = [], [], 0
        for row in sequences:
            if row["shield_mode"] != CANDIDATE or row["surface_kind"] != kind:
                continue
            reference = control[_key(row)]
            pair_count += 1
            control_steps_all.append(int(reference["control_steps"]))
            candidate_steps_all.append(int(row["control_steps"]))
            ref_hard, cand_hard = _hard_violated(reference), _hard_violated(row)
            control_hard += int(ref_hard)
            candidate_hard += int(cand_hard)
            if ref_hard or cand_hard:
                censored += 1
            else:
                deltas.append(int(row["control_steps"]) - int(reference["control_steps"]))
        rows.append({
            "surface_kind": kind,
            "pairs": pair_count,
            "censored_pairs": censored,
            "uncensored_pairs": len(deltas),
            "control_hard_violations": control_hard,
            "candidate_hard_violations": candidate_hard,
            "raw_mean_control_steps_control_censored": float(np.mean(control_steps_all)),
            "raw_mean_control_steps_candidate_censored": float(np.mean(candidate_steps_all)),
            "uncensored_mean_step_delta_candidate_minus_control":
                float(np.mean(deltas)) if deltas else float("nan"),
        })
    return rows


def seed_rows(sequences: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for base in SEED_BASES:
            for scope, kinds in (("flat", ("flat",)), ("curved", CURVED)):
                group = [row for row in sequences if row["shield_mode"] == arm
                         and row["surface_seed_base"] == str(base)
                         and row["surface_kind"] in kinds]
                rows.append({"shield_mode": arm, "surface_seed_base": base,
                             "scope": scope, **_stats(group)})
    return rows


def evaluate(sequences: list[dict[str, str]], tiles: list[dict[str, str]]
             ) -> tuple[_Checks, dict[str, Any], list[dict[str, Any]]]:
    checks = _Checks()
    control_entry = _stats([row for row in sequences if row["shield_mode"] == CONTROL
                            and row["surface_kind"] in ENTRY_CURVED])
    candidate_entry = _stats([row for row in sequences if row["shield_mode"] == CANDIDATE
                              and row["surface_kind"] in ENTRY_CURVED])
    control_flat = _stats([row for row in sequences if row["shield_mode"] == CONTROL
                           and row["surface_kind"] == "flat"])
    candidate_flat = _stats([row for row in sequences if row["shield_mode"] == CANDIDATE
                             and row["surface_kind"] == "flat"])
    candidate_all = _stats([row for row in sequences if row["shield_mode"] == CANDIDATE])
    drop = paired_all4_drop_pp(sequences, CANDIDATE, CURVED, DIRECTIONS)
    parity = flat_parity_rows(sequences, CANDIDATE, DIRECTIONS)
    all_faults = sum(int(row["sensor_fault_steps"]) for row in sequences)
    checks.add("completion", "sequences", len(sequences), "eq", EXPECTED_SEQUENCES)
    checks.add("completion", "tile_rows", len(tiles), "eq", EXPECTED_TILE_ROWS)
    checks.add("final", "entry_curved_safety_pass", candidate_entry["safety_pass"], "gt",
               control_entry["safety_pass"])
    checks.add("final", "candidate_force_overloads", candidate_all["force_overloads"], "eq", 0)
    checks.add("final", "flat_safety_pass", candidate_flat["safety_pass"], "ge",
               control_flat["safety_pass"])
    checks.add("final", "flat_quality_pass", candidate_flat["quality_pass"], "ge",
               control_flat["quality_pass"])
    checks.add("final", "mean_curved_all4_area_drop_pp", drop, "le", MAX_CURVED_ALL4_DROP_PP)
    checks.add("final", "sensor_fault_steps", all_faults, "eq", 0)
    checks.add("final", "flat_action_passthrough_parity",
               all(row["identical"] for row in parity), "eq", True)
    control_curved = _stats([row for row in sequences if row["shield_mode"] == CONTROL
                             and row["surface_kind"] in CURVED])
    candidate_curved = _stats([row for row in sequences if row["shield_mode"] == CANDIDATE
                               and row["surface_kind"] in CURVED])
    detail = {
        "candidate": CANDIDATE,
        "control_entry_curved_safety_pass": control_entry["safety_pass"],
        "candidate_entry_curved_safety_pass": candidate_entry["safety_pass"],
        "control_curved_safety_pass": control_curved["safety_pass"],
        "candidate_curved_safety_pass": candidate_curved["safety_pass"],
        "control_curved_sequences": control_curved["sequences"],
        "control_force_overloads": _stats([row for row in sequences
                                           if row["shield_mode"] == CONTROL])["force_overloads"],
        "candidate_force_overloads": candidate_all["force_overloads"],
        "mean_curved_all4_area_drop_pp": drop,
        "curved_all4_area_delta_pp_candidate_minus_control":
            candidate_curved["mean_final_all4_area_pct"]
            - control_curved["mean_final_all4_area_pct"],
    }
    return checks, detail, parity


def verify_duplicates(duplicate_dirs: list[str],
                      metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    official = {_config_key(meta): meta["run_dir"] for meta in metadata}
    rows = []
    for directory in duplicate_dirs:
        meta = _load_metadata(directory)
        config = _config_key(meta)
        if config not in official:
            raise ValueError(f"{directory}: duplicate has no official counterpart {config}")
        duplicate_sha = _sha256(os.path.join(directory, "sequences.csv"))
        official_sha = _sha256(os.path.join(official[config], "sequences.csv"))
        if duplicate_sha != official_sha:
            raise ValueError(f"{directory}: duplicate sequences.csv differs from official "
                             f"{official[config]}; human review required")
        rows.append({"duplicate_dir": directory, "official_dir": official[config],
                     "sequences_sha256": duplicate_sha, "hash_match": True})
    return rows


def aggregate(plan_path: str, run_dirs: list[str], out_dir: str,
              duplicate_dirs: list[str]) -> dict[str, Any]:
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("gate") != GATE or plan.get("frozen_before_execution") is not True:
        raise ValueError("plan file is not the frozen Gate F9 plan")
    if plan.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("plan parent hash does not match frozen constant")
    if tuple(plan.get("surface_seed_bases", ())) != SEED_BASES:
        raise ValueError("plan seed bases do not match frozen constant")
    sequences, tiles, metadata = load_runs(run_dirs)
    validate_pairing(sequences)
    checks, detail, parity = evaluate(sequences, tiles)
    duplicates = verify_duplicates(duplicate_dirs, metadata)
    geometry = geometry_rows(sequences, ARMS, DIRECTIONS)
    censoring = censoring_rows(sequences)
    seeds = seed_rows(sequences)
    passed = checks.passed()
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": GATE,
        "decision": "PASS" if passed else "FAIL",
        "integration_candidate": passed,
        "completion_pass": checks.passed("completion"),
        "runs": len(metadata),
        "sequences": len(sequences),
        "tile_rows": len(tiles),
        "duplicates_excluded": len(duplicates),
        "parent_sha256": PARENT_SHA256,
        "surface_seed_bases": list(SEED_BASES),
        "training_performed": False,
        "ppo_performed": False,
        "champion_promoted": False,
        "release_modified": False,
        "production_readiness": False,
        "recommendation": (
            "Gate F9 PASS: static_cap(+0.50) is an integration candidate only; "
            "integration/release requires a separately approved stage."
            if passed else
            "Gate F9 FAIL: report as-is; do not tune thresholds, seeds, or the cap."),
        **detail,
    }
    os.makedirs(out_dir)
    _write_csv(os.path.join(out_dir, "combined_sequences.csv"), sequences)
    _write_csv(os.path.join(out_dir, "combined_tile_area_fractions.csv"), tiles)
    _write_csv(os.path.join(out_dir, "arm_geometry_summary.csv"), geometry)
    _write_csv(os.path.join(out_dir, "seed_summary.csv"), seeds)
    _write_csv(os.path.join(out_dir, "flat_passthrough_parity.csv"), parity)
    _write_csv(os.path.join(out_dir, "censoring_summary.csv"), censoring)
    _write_csv(os.path.join(out_dir, "acceptance_checks.csv"), checks.rows)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    readme = [
        "# Gate F9 multi-seed paired validation",
        "",
        f"- Decision: **{decision['decision']}** "
        f"(integration candidate: {decision['integration_candidate']})",
        f"- Official sample: {decision['runs']} runs, {decision['sequences']} sequences, "
        f"{decision['tile_rows']} tile rows; duplicates excluded: "
        f"{decision['duplicates_excluded']}",
        f"- Entry curved (cylinder+freeform) safety: control "
        f"{detail['control_entry_curved_safety_pass']} -> candidate "
        f"{detail['candidate_entry_curved_safety_pass']}",
        f"- Curved (cyl+sph+free) safety: control {detail['control_curved_safety_pass']}"
        f"/{detail['control_curved_sequences']} -> candidate "
        f"{detail['candidate_curved_safety_pass']}/{detail['control_curved_sequences']}",
        f"- Force overloads: control {detail['control_force_overloads']} -> candidate "
        f"{detail['candidate_force_overloads']}",
        f"- Mean paired curved all4 drop: {detail['mean_curved_all4_area_drop_pp']:.6f} pp "
        f"(limit {MAX_CURVED_ALL4_DROP_PP})",
        "- Curved control times are censored by early hard terminations; see "
        "censoring_summary.csv before comparing control steps.",
        "- Training/PPO/promotion/release changes: none. A PASS is an integration "
        "candidate only; integration/release is a separately approved stage.",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(readme))
    source_rows = [{"role": "plan", "path": os.path.abspath(plan_path),
                    "sha256": _sha256(plan_path)}]
    for meta in metadata:
        for name in ("metadata.json", "sequences.csv", "tile_area_fractions.csv"):
            path = os.path.join(meta["run_dir"], name)
            source_rows.append({"role": "official", "path": os.path.abspath(path),
                                "sha256": _sha256(path)})
    for row in duplicates:
        path = os.path.join(row["duplicate_dir"], "sequences.csv")
        source_rows.append({"role": "duplicate", "path": os.path.abspath(path),
                            "sha256": row["sequences_sha256"]})
    _write_csv(os.path.join(out_dir, "source_checksums.csv"), source_rows)
    names = ("combined_sequences.csv", "combined_tile_area_fractions.csv",
             "arm_geometry_summary.csv", "seed_summary.csv", "flat_passthrough_parity.csv",
             "censoring_summary.csv", "acceptance_checks.csv", "decision.json", "README.md",
             "source_checksums.csv")
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{_sha256(os.path.join(out_dir, name))}  {name}\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-list", required=True,
                        help="text file with one official run directory per line")
    parser.add_argument("--duplicates", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    with open(args.run_list, encoding="utf-8") as handle:
        run_dirs = [line.strip() for line in handle if line.strip()]
    decision = aggregate(args.plan, run_dirs, args.out_dir, args.duplicates)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
