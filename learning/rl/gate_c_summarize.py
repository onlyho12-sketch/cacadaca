"""Combine the four Gate C2 runs into paired, auditable CSV summaries."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np


RUN_KEYS = (
    ("legacy_stress", "cross_xy"),
    ("legacy_stress", "same_xx"),
    ("new_car_mild", "cross_xy"),
    ("new_car_mild", "same_xx"),
)
METRICS = (
    "profile_gu_mean", "roi_total_ra_um", "roi_total_rz_um",
    "roi_fine_ra_um", "roi_fine_rz_um", "roi_removal_mean_um",
    "roi_removal_std_um", "roi_removal_cv", "roi_removal_waviness_ra_um",
    "roi_removal_waviness_std_um", "roi_center_edge_delta_um",
    "roi_center_edge_ratio", "roi_coverage_fraction", "roi_scratch_max_um",
    "roi_clearcoat_min_um", "profile_mar_severity_remaining_mean",
    "profile_q_mar_mean", "force_used_mean_n", "force_used_max_n",
    "feed_cmd_mean_mm_s", "fallback_steps",
)
HIGHER_BETTER = {"profile_gu_mean", "roi_coverage_fraction", "profile_q_mar_mean",
                 "roi_clearcoat_min_um"}


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_cross", required=True)
    parser.add_argument("--legacy_same", required=True)
    parser.add_argument("--newcar_cross", required=True)
    parser.add_argument("--newcar_same", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    dirs = {
        ("legacy_stress", "cross_xy"): os.path.abspath(args.legacy_cross),
        ("legacy_stress", "same_xx"): os.path.abspath(args.legacy_same),
        ("new_car_mild", "cross_xy"): os.path.abspath(args.newcar_cross),
        ("new_car_mild", "same_xx"): os.path.abspath(args.newcar_same),
    }
    data = {}
    all_sequences, all_passes, all_tiles = [], [], []
    for key, directory in dirs.items():
        initial = _read(os.path.join(directory, "initial_diagnostics.csv"))
        sequences = _read(os.path.join(directory, "sequences.csv"))
        passes = _read(os.path.join(directory, "passes.csv"))
        tiles = _read(os.path.join(directory, "tiles.csv"))
        metadata = json.load(open(os.path.join(directory, "metadata.json"), encoding="utf-8"))
        if not (len(initial) == len(sequences) == len(passes) == 16 and len(tiles) == 800):
            raise RuntimeError(f"unexpected row counts for {key}")
        data[key] = {"initial": initial, "sequences": sequences,
                     "passes": passes, "tiles": tiles, "metadata": metadata}
        all_sequences.extend(sequences); all_passes.extend(passes)
        for row in tiles:
            all_tiles.append({"surface_profile": key[0], "direction_mode": key[1], **row})

    seeds = [data[key]["metadata"]["profile_seeds"] for key in RUN_KEYS]
    seed_exact = all(value == seeds[0] for value in seeds[1:])
    initial_exact = {}
    for profile in ("legacy_stress", "new_car_mild"):
        cross = sorted(data[(profile, "cross_xy")]["initial"], key=lambda row: int(row["env"]))
        same = sorted(data[(profile, "same_xx")]["initial"], key=lambda row: int(row["env"]))
        initial_exact[profile] = all(
            all(a[name] == b[name] for name in a if name != "env")
            for a, b in zip(cross, same))
    if not seed_exact or not all(initial_exact.values()):
        raise RuntimeError("paired seed or initial-state equality failed")

    group_rows = []
    for profile, direction in RUN_KEYS:
        block = data[(profile, direction)]
        initial = {int(row["env"]): row for row in block["initial"]}
        passes = {int(row["env"]): row for row in block["passes"]}
        sequences = block["sequences"]
        row = {
            "surface_profile": profile, "direction_mode": direction,
            "n": 16,
            "success_count": sum(item["outcome"] == "success" for item in sequences),
            "quality_ok_count": sum(item["quality_ok"] == "True" for item in sequences),
            "safety_ok_count": sum(item["safety_ok"] == "True" for item in sequences),
            "gu70_count": sum(float(item["gu_final"]) >= 70.0 for item in sequences),
            "ra_le_0p20_count": sum(float(item["ra_final_um"]) <= 0.20 for item in sequences),
            "rz_le_2p0_count": sum(float(item["rz_final_um"]) <= 2.0 for item in sequences),
            "scratch_improved_count": sum(
                float(item["scratch_final_um"]) < float(item["scratch_before_um"])
                or float(item["scratch_before_um"]) < 0.05 for item in sequences),
        }
        for metric in METRICS:
            final = np.asarray([float(passes[i][metric]) for i in range(16)])
            row[f"{metric}_mean"] = float(final.mean())
            if metric in initial[0]:
                before = np.asarray([float(initial[i][metric]) for i in range(16)])
                row[f"{metric}_initial_mean"] = float(before.mean())
                row[f"{metric}_change_mean"] = float((final - before).mean())
        group_rows.append(row)

    paired_rows, paired_summary = [], []
    for profile in ("legacy_stress", "new_car_mild"):
        cross = {int(row["env"]): row for row in data[(profile, "cross_xy")]["passes"]}
        same = {int(row["env"]): row for row in data[(profile, "same_xx")]["passes"]}
        for env_id in range(16):
            row = {"surface_profile": profile, "env": env_id,
                   "profile_seed": int(cross[env_id]["profile_seed"])}
            for metric in METRICS:
                c = float(cross[env_id][metric]); s = float(same[env_id][metric])
                row[f"{metric}_cross"] = c
                row[f"{metric}_same"] = s
                row[f"{metric}_cross_minus_same"] = c - s
            paired_rows.append(row)
        summary = {"surface_profile": profile, "n": 16}
        rows = [row for row in paired_rows if row["surface_profile"] == profile]
        for metric in METRICS:
            delta = np.asarray([row[f"{metric}_cross_minus_same"] for row in rows])
            summary[f"{metric}_cross_minus_same_mean"] = float(delta.mean())
            if metric in HIGHER_BETTER:
                summary[f"{metric}_cross_better_count"] = int((delta > 0.0).sum())
            else:
                summary[f"{metric}_cross_better_count"] = int((delta < 0.0).sum())
        paired_summary.append(summary)

    _write(os.path.join(out_dir, "group_summary.csv"), group_rows)
    _write(os.path.join(out_dir, "paired_direction_deltas.csv"), paired_rows)
    _write(os.path.join(out_dir, "paired_direction_summary.csv"), paired_summary)
    _write(os.path.join(out_dir, "combined_sequences.csv"), all_sequences)
    _write(os.path.join(out_dir, "combined_passes.csv"), all_passes)
    _write(os.path.join(out_dir, "combined_tiles.csv"), all_tiles)
    with open(os.path.join(out_dir, "pairing_validation.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed_exact_across_four_runs": seed_exact,
            "profile_seeds": seeds[0],
            "initial_exact_cross_vs_same": initial_exact,
            "run_directories": {f"{p}__{d}": path for (p, d), path in dirs.items()},
            "training_performed": False,
        }, handle, indent=2, sort_keys=True)
    print(f"[Gate C] wrote {out_dir}")
    print(f"[Gate C] seeds exact={seed_exact} initial exact={initial_exact}")


if __name__ == "__main__":
    main()
