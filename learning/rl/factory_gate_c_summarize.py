"""Combine corrected factory Gate C results with preserved legacy evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone

import numpy as np


GROUP_METRICS = (
    "profile_gu_mean", "roi_total_ra_um", "roi_total_rz_um",
    "roi_fine_ra_um", "roi_fine_rz_um", "roi_removal_mean_um",
    "roi_removal_std_um", "roi_removal_cv", "roi_removal_waviness_ra_um",
    "roi_removal_waviness_std_um", "roi_center_edge_delta_um",
    "roi_center_edge_ratio", "roi_coverage_fraction", "roi_scratch_max_um",
    "roi_clearcoat_min_um", "force_used_mean_n", "force_used_max_n",
    "feed_cmd_mean_mm_s", "fallback_steps",
)
HIGHER_BETTER = {
    "profile_gu_mean", "roi_coverage_fraction", "roi_clearcoat_min_um",
}
UNIFIED_GROUP_FIELDS = (
    "surface_profile", "direction_mode", "n", "success_count",
    "quality_ok_count", "safety_ok_count", "gu70_count",
    "ra_le_0p20_count", "rz_le_2p0_count", "scratch_improved_count",
    "profile_gu_mean_mean", "roi_total_ra_um_mean", "roi_total_rz_um_mean",
    "roi_fine_ra_um_mean", "roi_fine_rz_um_mean", "roi_removal_mean_um_mean",
    "roi_removal_std_um_mean", "roi_removal_cv_mean",
    "roi_removal_waviness_ra_um_mean", "roi_removal_waviness_std_um_mean",
    "roi_center_edge_delta_um_mean", "roi_center_edge_ratio_mean",
    "roi_coverage_fraction_mean", "roi_scratch_max_um_mean",
    "roi_clearcoat_min_um_mean", "force_used_mean_n_mean",
    "force_used_max_n_mean", "feed_cmd_mean_mm_s_mean", "fallback_steps_mean",
)


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_finite(rows: list[dict], label: str) -> None:
    for index, row in enumerate(rows):
        for key, value in row.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                raise RuntimeError(f"{label} row={index} field={key} is not finite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--legacy_summary_dir")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    combined = {name: [] for name in ("initial_diagnostics", "sequences", "passes", "tiles")}
    metadata = []
    for run_dir_arg in args.run_dirs:
        run_dir = os.path.abspath(run_dir_arg)
        with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("training_performed") is not False:
            raise RuntimeError(f"training flag is not false: {run_dir}")
        metadata.append(meta)
        for name in combined:
            rows = _read(os.path.join(run_dir, f"{name}.csv"))
            _assert_finite(rows, f"{run_dir}/{name}")
            if name == "initial_diagnostics":
                for row in rows:
                    row.setdefault("path_mode", meta["path"]["path_mode"])
                    row.setdefault("direction_mode", meta["path"]["direction_mode"])
            combined[name].extend({"source_run": run_dir, **row} for row in rows)

    initial_by_key = {
        (row["surface_profile"], row.get("direction_mode", "same_xx"), int(row["profile_seed"])): row
        for row in combined["initial_diagnostics"]
    }
    sequences_by_key = {
        (row["surface_profile"], row["direction_mode"], int(row["profile_seed"])): row
        for row in combined["sequences"]
    }
    passes_by_key = {
        (row["surface_profile"], row["direction_mode"], int(row["profile_seed"])): row
        for row in combined["passes"]
    }
    if len(sequences_by_key) != len(combined["sequences"]):
        raise RuntimeError("duplicate profile/direction/seed sequence key")

    groups = sorted({(key[0], key[1]) for key in sequences_by_key})
    group_rows = []
    for profile, direction in groups:
        sequence_rows = [row for key, row in sequences_by_key.items()
                         if key[:2] == (profile, direction)]
        pass_rows = [row for key, row in passes_by_key.items()
                     if key[:2] == (profile, direction)]
        row = {
            "surface_profile": profile,
            "direction_mode": direction,
            "n": len(sequence_rows),
            "success_count": sum(item["outcome"] == "success" for item in sequence_rows),
            "quality_ok_count": sum(item["quality_ok"] == "True" for item in sequence_rows),
            "safety_ok_count": sum(item["safety_ok"] == "True" for item in sequence_rows),
            "gu70_count": sum(float(item["gu_final"]) >= 70.0 for item in sequence_rows),
            "ra_le_0p20_count": sum(float(item["ra_final_um"]) <= 0.20 for item in sequence_rows),
            "rz_le_2p0_count": sum(float(item["rz_final_um"]) <= 2.0 for item in sequence_rows),
            "scratch_improved_count": sum(
                float(item["scratch_final_um"]) < float(item["scratch_before_um"])
                for item in sequence_rows),
        }
        for metric in GROUP_METRICS:
            if pass_rows and metric in pass_rows[0]:
                values = np.asarray([float(item[metric]) for item in pass_rows])
                row[f"{metric}_mean"] = float(values.mean())
                initial_rows = [
                    initial_by_key[(profile, direction, int(item["profile_seed"]))]
                    for item in pass_rows
                ]
                if metric in initial_rows[0]:
                    before = np.asarray([float(item[metric]) for item in initial_rows])
                    row[f"{metric}_initial_mean"] = float(before.mean())
                    row[f"{metric}_change_mean"] = float((values - before).mean())
        group_rows.append(row)

    paired_rows = []
    paired_summary_rows = []
    initial_exact = {}
    profiles = sorted({row["surface_profile"] for row in combined["sequences"]})
    for profile in profiles:
        cross = {key[2]: row for key, row in passes_by_key.items()
                 if key[:2] == (profile, "cross_xy")}
        same = {key[2]: row for key, row in passes_by_key.items()
                if key[:2] == (profile, "same_xx")}
        if not cross or not same:
            continue
        if set(cross) != set(same):
            raise RuntimeError(f"paired seeds differ for {profile}")
        cross_initial = {
            key[2]: row for key, row in initial_by_key.items()
            if key[:2] == (profile, "cross_xy")
        }
        same_initial = {
            key[2]: row for key, row in initial_by_key.items()
            if key[:2] == (profile, "same_xx")
        }
        if set(cross_initial) != set(same_initial):
            raise RuntimeError(f"initial paired seeds differ for {profile}")
        ignored = {"source_run", "path_mode", "direction_mode"}
        initial_exact[profile] = all(
            all(cross_initial[seed][name] == same_initial[seed][name]
                for name in cross_initial[seed] if name not in ignored)
            for seed in cross_initial
        )
        if not initial_exact[profile]:
            raise RuntimeError(f"initial diagnostic equality failed for {profile}")
        for seed in sorted(cross):
            row = {"surface_profile": profile, "profile_seed": seed}
            for metric in GROUP_METRICS:
                if metric in cross[seed] and metric in same[seed]:
                    c, s = float(cross[seed][metric]), float(same[seed][metric])
                    row[f"{metric}_cross"] = c
                    row[f"{metric}_same"] = s
                    row[f"{metric}_cross_minus_same"] = c - s
            paired_rows.append(row)
        block = [row for row in paired_rows if row["surface_profile"] == profile]
        summary = {"surface_profile": profile, "n": len(block)}
        for metric in GROUP_METRICS:
            delta_name = f"{metric}_cross_minus_same"
            if delta_name not in block[0]:
                continue
            delta = np.asarray([float(row[delta_name]) for row in block])
            summary[f"{metric}_cross_minus_same_mean"] = float(delta.mean())
            if metric in HIGHER_BETTER:
                summary[f"{metric}_cross_better_count"] = int((delta > 0.0).sum())
            else:
                summary[f"{metric}_cross_better_count"] = int((delta < 0.0).sum())
        paired_summary_rows.append(summary)

    for name, rows in combined.items():
        _write(os.path.join(out_dir, f"combined_{name}.csv"), rows)
    _write(os.path.join(out_dir, "group_summary.csv"), group_rows)
    _write(os.path.join(out_dir, "paired_direction_deltas.csv"), paired_rows)
    _write(os.path.join(out_dir, "paired_direction_summary.csv"), paired_summary_rows)

    unified_groups = []
    unified_paired = []
    legacy_sequence_count = 0
    legacy_summary_dir = None
    if args.legacy_summary_dir:
        legacy_summary_dir = os.path.abspath(args.legacy_summary_dir)
        legacy_groups = [
            row for row in _read(os.path.join(legacy_summary_dir, "group_summary.csv"))
            if row["surface_profile"] == "legacy_stress"
        ]
        legacy_paired = [
            row for row in _read(os.path.join(
                legacy_summary_dir, "paired_direction_summary.csv"))
            if row["surface_profile"] == "legacy_stress"
        ]
        if len(legacy_groups) != 2 or len(legacy_paired) != 1:
            raise RuntimeError("expected exactly two legacy groups and one paired summary")
        legacy_pairing = json.load(open(
            os.path.join(legacy_summary_dir, "pairing_validation.json"),
            encoding="utf-8"))
        if (legacy_pairing.get("training_performed") is not False
                or not legacy_pairing["initial_exact_cross_vs_same"]["legacy_stress"]):
            raise RuntimeError("preserved legacy Gate C pairing validation failed")
        legacy_sequence_count = sum(int(row["n"]) for row in legacy_groups)
        unified_groups.extend({
            "evidence_source": "preserved_legacy_gate_c",
            **{name: row.get(name, "") for name in UNIFIED_GROUP_FIELDS},
        } for row in legacy_groups)
        unified_paired.extend({
            "evidence_source": "preserved_legacy_gate_c", **row,
        } for row in legacy_paired)
    unified_groups.extend({
        "evidence_source": "corrected_factory_gate_c",
        **{name: row.get(name, "") for name in UNIFIED_GROUP_FIELDS},
    } for row in group_rows)
    unified_paired.extend({
        "evidence_source": "corrected_factory_gate_c", **row,
    } for row in paired_summary_rows)
    _write(os.path.join(out_dir, "all_profile_group_summary.csv"), unified_groups)
    # Legacy and factory paired CSVs have a common core but not necessarily the
    # same historical extra columns.  Keep a compact cross-profile comparison.
    paired_common = (
        "surface_profile", "n",
        "profile_gu_mean_cross_minus_same_mean",
        "roi_total_ra_um_cross_minus_same_mean",
        "roi_total_rz_um_cross_minus_same_mean",
        "roi_fine_ra_um_cross_minus_same_mean",
        "roi_fine_rz_um_cross_minus_same_mean",
        "roi_removal_std_um_cross_minus_same_mean",
        "roi_removal_cv_cross_minus_same_mean",
        "roi_removal_waviness_ra_um_cross_minus_same_mean",
        "roi_removal_waviness_std_um_cross_minus_same_mean",
        "roi_center_edge_delta_um_cross_minus_same_mean",
        "roi_center_edge_ratio_cross_minus_same_mean",
        "roi_coverage_fraction_cross_minus_same_mean",
        "roi_scratch_max_um_cross_minus_same_mean",
        "force_used_max_n_cross_minus_same_mean",
    )
    _write(os.path.join(out_dir, "all_profile_paired_direction_summary.csv"), [
        {"evidence_source": row["evidence_source"],
         **{name: row.get(name, "") for name in paired_common}}
        for row in unified_paired
    ])
    with open(os.path.join(out_dir, "pairing_validation.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "factory_initial_exact_cross_vs_same": initial_exact,
            "factory_seed_exact_cross_vs_same": True,
            "legacy_summary_dir": legacy_summary_dir,
            "legacy_sequence_count": legacy_sequence_count,
            "factory_sequence_count": len(combined["sequences"]),
            "all_profile_sequence_count": legacy_sequence_count + len(combined["sequences"]),
            "training_performed": False,
        }, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "summary_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_directories": [os.path.abspath(path) for path in args.run_dirs],
            "run_metadata": metadata,
            "training_performed": False,
            "sequence_count": len(combined["sequences"]),
            "paired_row_count": len(paired_rows),
            "factory_initial_exact_cross_vs_same": initial_exact,
            "legacy_summary_dir": legacy_summary_dir,
            "legacy_sequence_count": legacy_sequence_count,
            "all_profile_sequence_count": legacy_sequence_count + len(combined["sequences"]),
        }, handle, indent=2, sort_keys=True)
    print(f"factory summary: {len(combined['sequences'])} sequences, wrote {out_dir}")


if __name__ == "__main__":
    main()
