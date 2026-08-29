#!/usr/bin/env python3
"""Bridge the BMW 150-cell inspection contract to the imported RL twin.

The inspection project stores scratch as a dimensionless severity in [0, 1],
while ``learning/vehicle_export`` expects and returns a scratch depth in um.
This bridge keeps both representations and records the deterministic mapping.

The imported twin's GU value is a multi-factor literature proxy.  The existing
inspection CSV requires a normalized optical-looking field and reconstructs GU
with the 25--78 anchor.  For contract compatibility only, this bridge stores
the inverse anchor value in ``relative_gloss_*_not_gu``.  It is explicitly
tagged as encoded twin output, not an RTX optical measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SCRATCH_MIN_UM = 0.05
SCRATCH_MAX_UM = 2.0
GU_LOW = 25.0
GU_HIGH = 78.0
GU_TARGET = 70.0
RA_RL_MAX_UM = 0.20
RA_TOYOTA_MIN_UM = 0.070
RA_TOYOTA_MAX_UM = 0.100
RZ_MAX_UM = 2.0
CLEARCOAT_MIN_UM = 35.0

LEARNING_INPUT_COLUMNS = (
    "region_id", "region_name", "cell_id",
    "position_x_m", "position_y_m", "position_z_m",
    "normal_x", "normal_y", "normal_z",
    "init_roughness_um", "init_scratch_um", "init_ra_um", "init_rz_um",
    "init_clearcoat_um", "init_gu_proxy", "surface_seed",
)

INSPECTION_REQUIRED = (
    "episode_id", "scenario_seed", "region_id", "region_label", "cell_id",
    "grid_row", "grid_column",
    "position_x_m", "position_y_m", "position_z_m",
    "normal_x", "normal_y", "normal_z",
    "roughness_before", "scratch_before", "ra_before_um", "rz_before_um",
    "clearcoat_before_um", "gu_proxy_before",
)

LEARNING_RESULT_REQUIRED = (
    "region_id", "cell_id", "force_n", "rpm", "feed_speed_mm_s",
    "step_over_ratio", "pass_count", "policy_action_force", "policy_action_feed",
    "roughness_before", "roughness_after",
    "scratch_before_um", "scratch_after_um",
    "ra_before_um", "ra_after_um", "rz_before_um", "rz_after_um",
    "clearcoat_initial_um", "clearcoat_removed_um", "clearcoat_remaining_min_um",
    "gu_proxy_before", "gu_proxy_after", "overall_pass", "failure_reason",
    "surface_seed", "process_time_s", "episode_completed", "evaluation_mode",
)


def read_csv(path: Path, required=()):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is missing")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return list(reader.fieldnames), rows


def finite(row, name, identity):
    try:
        value = float(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{identity}: {name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{identity}: {name} must be finite")
    return value


def boolean(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def scratch_severity_to_um(severity: float) -> float:
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"scratch severity must be in [0, 1], got {severity}")
    if severity <= 1.0e-12:
        return 0.0
    return SCRATCH_MIN_UM + severity * (SCRATCH_MAX_UM - SCRATCH_MIN_UM)


def scratch_um_to_severity(depth_um: float) -> float:
    if depth_um <= 1.0e-12:
        return 0.0
    return min(1.0, max(0.0, (depth_um - SCRATCH_MIN_UM) /
                            (SCRATCH_MAX_UM - SCRATCH_MIN_UM)))


def gu_to_anchor_relative(gu: float) -> float:
    return min(1.0, max(0.0, (gu - GU_LOW) / (GU_HIGH - GU_LOW)))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_metadata(policy_id: str):
    """Return truthful metadata without changing the imported exporter."""
    if policy_id == "bc_champion":
        return {
            "policy_type": "BC champion (behavior cloning)",
            "evaluation_mode": "flat_trained_bc_policy_inference",
        }
    if policy_id.startswith("terminal_ppo"):
        return {
            "policy_type": "terminal-reward PPO evaluation candidate",
            "evaluation_mode": "flat_trained_terminal_ppo_policy_inference",
        }
    return {
        "policy_type": f"policy checkpoint ({policy_id})",
        "evaluation_mode": "flat_trained_policy_inference",
    }


def correct_learning_summary(summary_path: Path, policy_id: str, checkpoint: Path):
    """Correct exporter metadata that is hard-coded to BC for every checkpoint."""
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = policy_metadata(policy_id)
    summary.update({
        "policy_id": policy_id,
        "policy_type": metadata["policy_type"],
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256(Path(checkpoint)),
        "evaluation_mode": metadata["evaluation_mode"],
        "metadata_corrected_by_bridge": True,
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def validate_complete_grid(rows):
    keyed = set()
    regions = defaultdict(set)
    for row in rows:
        identity = (row["region_id"].strip(), int(row["grid_row"]), int(row["grid_column"]))
        if identity in keyed:
            raise ValueError(f"duplicate BMW cell: {identity}")
        keyed.add(identity)
        regions[identity[0]].add(identity[1:])
    expected = {(r, c) for r in range(1, 6) for c in range(1, 6)}
    if len(rows) != 150 or len(regions) != 6:
        raise ValueError(f"expected 6 regions/150 cells, got {len(regions)}/{len(rows)}")
    for region, cells in regions.items():
        if cells != expected:
            raise ValueError(f"{region}: incomplete 5x5 grid")
    return sorted(regions)


def to_learning(input_csv: Path, output_csv: Path):
    _, rows = read_csv(input_csv, INSPECTION_REQUIRED)
    regions = validate_complete_grid(rows)
    converted = []
    scratch_depths = []
    for row in rows:
        region = row["region_id"].strip()
        grid_row = int(row["grid_row"])
        grid_column = int(row["grid_column"])
        identity = (region, grid_row, grid_column)
        severity = finite(row, "scratch_before", identity)
        scratch_um = scratch_severity_to_um(severity)
        scratch_depths.append(scratch_um)
        scenario_seed = int(float(row["scenario_seed"]))
        surface_seed = scenario_seed + (grid_row - 1) * 5 + (grid_column - 1)
        ra_um = finite(row, "ra_before_um", identity)
        # Rq estimate from Ra for a Gaussian profile.  The imported exporter
        # currently recomputes Rq from its synthesized patch and does not use
        # this input for dynamics, but a physically dimensioned value is kept.
        rq_estimate_um = ra_um * math.sqrt(math.pi / 2.0)
        converted.append({
            "region_id": region,
            "region_name": row["region_label"].strip() or region,
            "cell_id": row["cell_id"].strip(),
            "position_x_m": row["position_x_m"],
            "position_y_m": row["position_y_m"],
            "position_z_m": row["position_z_m"],
            "normal_x": row["normal_x"],
            "normal_y": row["normal_y"],
            "normal_z": row["normal_z"],
            "init_roughness_um": f"{rq_estimate_um:.9f}",
            "init_scratch_um": f"{scratch_um:.9f}",
            "init_ra_um": row["ra_before_um"],
            "init_rz_um": row["rz_before_um"],
            "init_clearcoat_um": row["clearcoat_before_um"],
            "init_gu_proxy": row["gu_proxy_before"],
            "surface_seed": surface_seed,
        })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_INPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(converted)
    summary = {
        "status": "bmw_150_converted_to_learning_twin_input",
        "source_csv": str(Path(input_csv).resolve()),
        "output_csv": str(output_csv.resolve()),
        "regions": regions,
        "cell_count": len(converted),
        "scratch_mapping": {
            "source": "dimensionless defect severity [0,1]",
            "destination": "synthetic scratch depth [um]",
            "formula": "0 if severity=0 else 0.05 + severity*(2.0-0.05)",
            "evidence": "0.05--2.0 um literature range; interpolation is PT-DESIGN",
            "minimum_mapped_um": min(scratch_depths),
            "maximum_mapped_um": max(scratch_depths),
        },
        "surface_seed_formula": "scenario_seed + (grid_row-1)*5 + (grid_column-1)",
        "important": (
            "The imported policy was trained on flat 120x120 mm patches. Vehicle position "
            "and normal are preserved, but curvature generalization is not validated."
        ),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_csv, summary_path, summary


def from_learning(initial_csv: Path, learning_csv: Path, output_csv: Path,
                  policy_id: str, checkpoint: Path, learning_summary: Path | None = None):
    _, initial_rows = read_csv(initial_csv, INSPECTION_REQUIRED)
    _, result_rows = read_csv(learning_csv, LEARNING_RESULT_REQUIRED)
    validate_complete_grid(initial_rows)
    if len(result_rows) != 150:
        raise ValueError(f"{learning_csv}: expected 150 result rows, got {len(result_rows)}")
    source = {row["cell_id"].strip(): row for row in initial_rows}
    if len(source) != 150:
        raise ValueError("initial CSV cell_id values are not unique")

    output = []
    seen = set()
    checkpoint = Path(checkpoint)
    ck_hash = checkpoint_sha256(checkpoint)
    metadata = policy_metadata(policy_id)
    for result in result_rows:
        cell_id = result["cell_id"].strip()
        if cell_id not in source:
            raise ValueError(f"unknown result cell_id: {cell_id}")
        if cell_id in seen:
            raise ValueError(f"duplicate result cell_id: {cell_id}")
        seen.add(cell_id)
        initial = source[cell_id]
        identity = cell_id
        gu_before = finite(result, "gu_proxy_before", identity)
        gu_after = finite(result, "gu_proxy_after", identity)
        scratch_before_um = finite(result, "scratch_before_um", identity)
        scratch_after_um = finite(result, "scratch_after_um", identity)
        clearcoat_before = finite(result, "clearcoat_initial_um", identity)
        clearcoat_mean_removed = finite(result, "clearcoat_removed_um", identity)
        clearcoat_mean_after = clearcoat_before - clearcoat_mean_removed
        clearcoat_min_after = finite(result, "clearcoat_remaining_min_um", identity)
        # The vehicle renderer has one clearcoat value per cell.  Use the
        # safety-critical minimum thickness, not the patch mean, so a local
        # breakthrough risk cannot be hidden by spatial averaging.
        clearcoat_after = clearcoat_min_after
        clearcoat_removed = clearcoat_before - clearcoat_after
        if clearcoat_after < 0.0 or clearcoat_removed < -1.0e-9:
            raise ValueError(f"{cell_id}: negative clearcoat remaining")

        output.append({
            "data_origin": f"synthetic_learning_twin_{policy_id}",
            "episode_id": f"{initial['episode_id']}_{policy_id}",
            "cell_id": cell_id,
            "grid_row": initial["grid_row"],
            "grid_column": initial["grid_column"],
            "position_x_m": initial["position_x_m"],
            "position_y_m": initial["position_y_m"],
            "position_z_m": initial["position_z_m"],
            "normal_x": initial["normal_x"],
            "normal_y": initial["normal_y"],
            "normal_z": initial["normal_z"],
            "force_n": result["force_n"],
            "rpm": result["rpm"],
            "feed_mm_s": result["feed_speed_mm_s"],
            "step_over_ratio": result["step_over_ratio"],
            "pass_count": result["pass_count"],
            # Rq in um is bounded below 1 for this model and is retained as the
            # inspection renderer's [0,1] roughness control value.
            "roughness_before": result["roughness_before"],
            "roughness_after": result["roughness_after"],
            "scratch_before": f"{scratch_um_to_severity(scratch_before_um):.9f}",
            "scratch_after": f"{scratch_um_to_severity(scratch_after_um):.9f}",
            "ra_before_um": result["ra_before_um"],
            "ra_after_um": result["ra_after_um"],
            "rz_before_um": result["rz_before_um"],
            "rz_after_um": result["rz_after_um"],
            "clearcoat_before_um": f"{clearcoat_before:.9f}",
            "clearcoat_after_um": f"{clearcoat_after:.9f}",
            "clearcoat_removed_um": f"{clearcoat_removed:.9f}",
            "relative_gloss_before_not_gu": f"{gu_to_anchor_relative(gu_before):.12f}",
            "relative_gloss_after_not_gu": f"{gu_to_anchor_relative(gu_after):.12f}",
            "region_id": initial["region_id"],
            "region_label": initial["region_label"],
            "gu_proxy_before": f"{gu_before:.9f}",
            "gu_proxy_after": f"{gu_after:.9f}",
            "policy_id": policy_id,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": ck_hash,
            "policy_action_force": result["policy_action_force"],
            "policy_action_feed": result["policy_action_feed"],
            "scratch_before_um_twin": f"{scratch_before_um:.9f}",
            "scratch_after_um_twin": f"{scratch_after_um:.9f}",
            "clearcoat_min_after_um": f"{clearcoat_min_after:.9f}",
            "clearcoat_mean_after_um_twin": f"{clearcoat_mean_after:.9f}",
            "clearcoat_mean_removed_um_twin": f"{clearcoat_mean_removed:.9f}",
            "twin_overall_pass": result["overall_pass"],
            "twin_failure_reason": result["failure_reason"],
            "surface_seed": result["surface_seed"],
            "process_time_s": result["process_time_s"],
            "episode_completed": result["episode_completed"],
            "evaluation_mode": metadata["evaluation_mode"],
            "optical_field_provenance": "inverse_encoded_from_multifactor_gu_proxy_not_rtx",
            "source_scratch_severity": initial["scratch_before"],
            "source_ra_before_um": initial["ra_before_um"],
            "source_rz_before_um": initial["rz_before_um"],
            "source_gu_proxy_before": initial["gu_proxy_before"],
        })

    if seen != set(source):
        raise ValueError(f"missing result cells: {sorted(set(source) - seen)[:10]}")
    output.sort(key=lambda row: (
        row["region_id"], int(row["grid_row"]), int(row["grid_column"])
    ))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = summarize_policy_rows(output, policy_id, checkpoint, ck_hash)
    summary["source_learning_csv"] = str(Path(learning_csv).resolve())
    summary["output_contract_csv"] = str(output_csv.resolve())
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if learning_summary is not None:
        correct_learning_summary(learning_summary, policy_id, checkpoint)
    return output_csv, summary_path, summary


def summarize_policy_rows(rows, policy_id, checkpoint, ck_hash=None):
    def values(name):
        return [float(row[name]) for row in rows]

    gu_after = values("gu_proxy_after")
    ra_after = values("ra_after_um")
    rz_after = values("rz_after_um")
    cc_min = values("clearcoat_min_after_um")
    scratch_before = values("scratch_before_um_twin")
    scratch_after = values("scratch_after_um_twin")
    gu_ok = [v >= GU_TARGET for v in gu_after]
    ra_rl_ok = [v <= RA_RL_MAX_UM for v in ra_after]
    ra_toyota_ok = [RA_TOYOTA_MIN_UM <= v <= RA_TOYOTA_MAX_UM for v in ra_after]
    rz_ok = [v <= RZ_MAX_UM for v in rz_after]
    cc_ok = [v >= CLEARCOAT_MIN_UM for v in cc_min]
    scratch_ok = [after < before or before < 0.05
                  for before, after in zip(scratch_before, scratch_after)]
    combined_rl = [all(flags) for flags in zip(gu_ok, ra_rl_ok, rz_ok, cc_ok, scratch_ok)]
    combined_toyota = [all(flags) for flags in zip(gu_ok, ra_toyota_ok, rz_ok, cc_ok, scratch_ok)]

    return {
        "status": "bmw_150_learning_policy_inference_converted",
        "policy_id": policy_id,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": ck_hash or checkpoint_sha256(Path(checkpoint)),
        "cell_count": len(rows),
        "evaluation_mode": sorted({row["evaluation_mode"] for row in rows}),
        "episode_completed_count": sum(boolean(row["episode_completed"]) for row in rows),
        "mean": {
            "gu_proxy_before": sum(values("gu_proxy_before")) / len(rows),
            "gu_proxy_after": sum(gu_after) / len(rows),
            "scratch_before_um": sum(scratch_before) / len(rows),
            "scratch_after_um": sum(scratch_after) / len(rows),
            "ra_after_um": sum(ra_after) / len(rows),
            "rz_after_um": sum(rz_after) / len(rows),
            "clearcoat_min_after_um": sum(cc_min) / len(rows),
            "force_n": sum(values("force_n")) / len(rows),
            "feed_mm_s": sum(values("feed_mm_s")) / len(rows),
        },
        "minimum_clearcoat_after_um": min(cc_min),
        "pass_counts": {
            "gu_proxy_ge_70": sum(gu_ok),
            "ra_le_0_20_um_rl_project": sum(ra_rl_ok),
            "ra_0_070_to_0_100_um_toyota_anchored": sum(ra_toyota_ok),
            "rz_le_2_0_um": sum(rz_ok),
            "clearcoat_min_ge_35_um": sum(cc_ok),
            "scratch_improved": sum(scratch_ok),
            "combined_rl_project": sum(combined_rl),
            "combined_toyota_anchored": sum(combined_toyota),
        },
        "gu_compatibility_encoding": {
            "formula": "relative=(multi_factor_gu_proxy-25)/(78-25), clipped to [0,1]",
            "is_rtx_optical_measurement": False,
        },
        "curved_generalization_validated": False,
        "synthetic_disclaimer": "All quality outputs are literature-model synthetic values, not measurements.",
    }


def compare(bc_summary: Path, terminal_summary: Path, output_json: Path):
    bc = json.loads(Path(bc_summary).read_text(encoding="utf-8"))
    terminal = json.loads(Path(terminal_summary).read_text(encoding="utf-8"))
    metrics = ("gu_proxy_after", "scratch_after_um", "ra_after_um", "rz_after_um",
               "clearcoat_min_after_um", "force_n", "feed_mm_s")
    result = {
        "status": "paired_bmw_150_policy_comparison",
        "same_cell_count": bc["cell_count"] == terminal["cell_count"] == 150,
        "bc": bc,
        "terminal_ppo": terminal,
        "terminal_minus_bc_mean": {
            name: terminal["mean"][name] - bc["mean"][name] for name in metrics
        },
        "decision_note": (
            "This comparison does not replace the official champion. BC remains the recorded "
            "champion; terminal PPO is an evaluation candidate."
        ),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_json, result


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("to-learning")
    prepare.add_argument("--input-csv", type=Path, required=True)
    prepare.add_argument("--output-csv", type=Path, required=True)
    restore = commands.add_parser("from-learning")
    restore.add_argument("--initial-csv", type=Path, required=True)
    restore.add_argument("--learning-csv", type=Path, required=True)
    restore.add_argument("--output-csv", type=Path, required=True)
    restore.add_argument("--policy-id", required=True)
    restore.add_argument("--checkpoint", type=Path, required=True)
    restore.add_argument("--learning-summary", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--bc-summary", type=Path, required=True)
    comparison.add_argument("--terminal-summary", type=Path, required=True)
    comparison.add_argument("--output-json", type=Path, required=True)
    return root


def main():
    args = parser().parse_args()
    if args.command == "to-learning":
        csv_path, summary_path, summary = to_learning(args.input_csv, args.output_csv)
        print(f"[learning bridge] input: {summary['cell_count']} cells / {len(summary['regions'])} regions")
        print(f"[learning bridge] CSV: {csv_path}")
        print(f"[learning bridge] summary: {summary_path}")
    elif args.command == "from-learning":
        csv_path, summary_path, summary = from_learning(
            args.initial_csv, args.learning_csv, args.output_csv,
            args.policy_id, args.checkpoint, args.learning_summary,
        )
        print(f"[learning bridge] policy={args.policy_id} cells={summary['cell_count']}")
        print(f"[learning bridge] GU>=70: {summary['pass_counts']['gu_proxy_ge_70']}/150")
        print(f"[learning bridge] RL combined: {summary['pass_counts']['combined_rl_project']}/150")
        print(f"[learning bridge] CSV: {csv_path}")
        print(f"[learning bridge] summary: {summary_path}")
    else:
        output, result = compare(args.bc_summary, args.terminal_summary, args.output_json)
        print(f"[learning bridge] paired comparison: {output}")
        print(json.dumps(result["terminal_minus_bc_mean"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
