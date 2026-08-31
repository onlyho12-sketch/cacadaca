"""Build Gate D inspection-extension evidence from preserved Gate C runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone

import numpy as np

from learning.rl.env.gate_d_observation import (
    GateDObservationTargets,
    encode_inspection_extension,
    feature_schema_rows,
    observation_feature_names,
)


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs=2, required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    feature_names = observation_feature_names("spatial120")[14:]
    targets = GateDObservationTargets(max_passes=1)
    vector_rows = []
    run_metadata = []
    for run_arg in args.run_dirs:
        run_dir = os.path.abspath(run_arg)
        with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("training_performed") is not False:
            raise RuntimeError(f"training flag is not false: {run_dir}")
        direction = metadata["path"]["direction_mode"]
        run_metadata.append(metadata)
        diagnostics = _read(os.path.join(run_dir, "initial_diagnostics.csv"))
        passes = _read(os.path.join(run_dir, "passes.csv"))
        tiles = _read(os.path.join(run_dir, "tiles.csv"))
        for inspection_stage, source_rows, tile_stage, inspection_pass in (
            ("initial", diagnostics, "initial", 0),
            ("after_pass", passes, "after_pass", 1),
        ):
            stage_tiles = [row for row in tiles
                           if row["stage"] == tile_stage
                           and int(row["pass"]) == inspection_pass]
            by_env = {}
            for row in stage_tiles:
                by_env.setdefault(int(row["env"]), []).append(row)
            for row in source_rows:
                env_id = int(row["env"])
                block = by_env[env_id]
                if len(block) != 25:
                    raise RuntimeError(
                        f"expected 25 {inspection_stage} tiles for env {env_id}")
                maps = {name: np.zeros((5, 5), dtype=float) for name in (
                    "scratch_max_um", "removal_mean_um", "under_fraction", "over_fraction")}
                for tile in block:
                    x, y = int(tile["tile_x"]), int(tile["tile_y"])
                    for name in maps:
                        maps[name][x, y] = float(tile[name])
                extension = encode_inspection_extension(
                    row, maps, inspection_pass, targets)
                vector = {
                    "source_run": run_dir,
                    "direction_mode": direction,
                    "inspection_stage": inspection_stage,
                    "inspection_pass": inspection_pass,
                    "surface_profile": row["surface_profile"],
                    "profile_seed": int(row["profile_seed"]),
                }
                vector.update({name: float(value)
                               for name, value in zip(feature_names, extension)})
                vector_rows.append(vector)

    if len(vector_rows) != 192:
        raise RuntimeError(
            f"expected 192 direction/stage/profile/seed vectors, got {len(vector_rows)}")
    by_key = {
        (row["direction_mode"], row["inspection_stage"],
         row["surface_profile"], row["profile_seed"]): row
        for row in vector_rows
    }
    directions = sorted({row["direction_mode"] for row in vector_rows})
    if directions != ["cross_xy", "same_xx"]:
        raise RuntimeError(f"unexpected directions: {directions}")
    paired_exact = True
    for profile in sorted({row["surface_profile"] for row in vector_rows}):
        seeds = sorted({row["profile_seed"] for row in vector_rows
                        if row["surface_profile"] == profile})
        if len(seeds) != 16:
            raise RuntimeError(f"expected 16 seeds for {profile}")
        for seed in seeds:
            cross = by_key[("cross_xy", "initial", profile, seed)]
            same = by_key[("same_xx", "initial", profile, seed)]
            paired_exact &= all(cross[name] == same[name] for name in feature_names)
    if not paired_exact:
        raise RuntimeError("Gate D extension pairing is not exact")

    summary_rows = []
    after_pass_exact_count = 0
    for profile in sorted({row["surface_profile"] for row in vector_rows}):
        seeds = sorted({row["profile_seed"] for row in vector_rows
                        if row["surface_profile"] == profile})
        for seed in seeds:
            cross = by_key[("cross_xy", "after_pass", profile, seed)]
            same = by_key[("same_xx", "after_pass", profile, seed)]
            after_pass_exact_count += int(
                all(cross[name] == same[name] for name in feature_names))
        for direction in directions:
            for inspection_stage in ("initial", "after_pass"):
                block = [row for row in vector_rows
                         if row["surface_profile"] == profile
                         and row["direction_mode"] == direction
                         and row["inspection_stage"] == inspection_stage]
                for feature in feature_names:
                    values = np.asarray([float(row[feature]) for row in block])
                    if not np.isfinite(values).all():
                        raise RuntimeError(
                            f"non-finite values in "
                            f"{profile}/{direction}/{inspection_stage}/{feature}")
                    summary_rows.append({
                        "surface_profile": profile,
                        "direction_mode": direction,
                        "inspection_stage": inspection_stage,
                        "feature_name": feature,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    })
    if not all(math.isfinite(float(row[k])) for row in summary_rows
               for k in ("mean", "std", "min", "max")):
        raise RuntimeError("non-finite Gate D feature summary")

    _write(os.path.join(out_dir, "feature_schema.csv"), feature_schema_rows())
    _write(os.path.join(out_dir, "inspection_extension_vectors.csv"), vector_rows)
    _write(os.path.join(out_dir, "feature_summary.csv"), summary_rows)
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_dirs": [os.path.abspath(item) for item in args.run_dirs],
            "vector_count": len(vector_rows),
            "extension_dim": len(feature_names),
            "initial_paired_cross_same_exact": paired_exact,
            "after_pass_exact_pair_count": after_pass_exact_count,
            "after_pass_pair_count": 48,
            "all_finite": True,
            "training_performed": False,
            "normalization_status": "PT-DESIGN_NOT_MEASURED_DISTRIBUTION",
            "run_metadata": run_metadata,
        }, handle, indent=2, sort_keys=True)
    print(
        f"Gate D observation validation: {len(vector_rows)} vectors x "
        f"{len(feature_names)} extension features, initial paired exact, all finite")


if __name__ == "__main__":
    main()
