"""Write auditable CPU distributions for legacy and factory pre-polish profiles."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from learning.polytwin.factory_surface_profiles import (
    FACTORY_PROFILE_IDS,
    FACTORY_SPECS,
    LEGACY_STRESS,
    PROFILE_IDS,
    make_factory_centered_patch,
)
from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel
from learning.polytwin.roughness_metrics import ra_um, rz_um
from learning.rl.env.planar_roi_diagnostics import (
    PlanarRoiGeometry,
    centered_roi_slices,
    make_centered_roi_patch,
    surface_view,
)


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _states_exact(left, right) -> bool:
    if vars(left).keys() != vars(right).keys():
        return False
    for name in vars(left):
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                return False
        elif a != b:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    geometry = PlanarRoiGeometry()
    sl = centered_roi_slices(geometry)
    gloss = LiteratureGlossProxyModel()
    rows = []
    legacy_exact_count = 0
    range_violation_count = 0
    for profile_id in PROFILE_IDS:
        for seed in range(args.seeds):
            state, metadata = make_factory_centered_patch(
                profile_id, geometry.map_size_m, geometry.roi_size_m,
                geometry.resolution_m, seed)
            exact = ""
            if profile_id == LEGACY_STRESS:
                exact = _states_exact(make_centered_roi_patch(geometry, seed), state)
                legacy_exact_count += int(exact)
            arrays_finite = all(
                np.isfinite(value).all()
                for value in vars(state).values() if isinstance(value, np.ndarray))
            roi = surface_view(state, sl)
            base_height = roi.initial_micro_height_um + roi.initial_scratch_depth_um
            gu = gloss.evaluate(roi)["summary"]
            shallow = metadata.get("shallow_segments", [])
            deep = metadata.get("deep_segments", [])
            if profile_id in FACTORY_SPECS:
                spec = FACTORY_SPECS[profile_id]
                valid = (
                    spec.base_ra_min_um <= metadata["base_ra_target_um"] <= spec.base_ra_max_um
                    and spec.shallow_scratch_count_min
                    <= metadata["shallow_scratch_count"]
                    <= spec.shallow_scratch_count_max
                    and metadata["deep_scratch_count"] == spec.deep_scratch_count
                    and all(spec.shallow_depth_min_um <= item["depth_um"]
                            <= spec.shallow_depth_max_um for item in shallow)
                    and all(float(spec.deep_depth_min_um) <= item["depth_um"]
                            <= float(spec.deep_depth_max_um) for item in deep)
                    and all(spec.scratch_length_min_m <= item["length_m"]
                            <= spec.scratch_length_max_m for item in (*shallow, *deep))
                )
                range_violation_count += int(not valid)
                if not valid:
                    raise RuntimeError(f"profile range violation: {profile_id} seed={seed}")
            row = {
                "surface_profile": profile_id,
                "profile_seed": seed,
                "design_status": metadata["design_status"],
                "legacy_exact": exact,
                "arrays_finite": arrays_finite,
                "surface_state_field_count": len(vars(state)),
                "base_ra_target_um": metadata["base_ra_target_um"],
                "roi_base_ra_um": ra_um(base_height),
                "roi_total_ra_um": ra_um(roi.initial_micro_height_um),
                "roi_total_rz_um": rz_um(roi.initial_micro_height_um),
                "roi_scratch_max_um": float(roi.initial_scratch_depth_um.max()),
                "shallow_scratch_count": metadata.get("shallow_scratch_count", ""),
                "deep_scratch_count": metadata.get("deep_scratch_count", ""),
                "sampled_shallow_depths_um_json": json.dumps(
                    [item["depth_um"] for item in shallow]),
                "sampled_deep_depths_um_json": json.dumps(
                    [item["depth_um"] for item in deep]),
                "sampled_scratch_lengths_m_json": json.dumps(
                    [item["length_m"] for item in (*shallow, *deep)]),
                "gu_mean": gu["gu_mean"],
                "gu_p10": gu["gu_p10"],
                "gu_min": gu["gu_min"],
                "clearcoat_min_um": float(roi.initial_clearcoat_um.min()),
            }
            if not arrays_finite or any(
                    not math.isfinite(float(value))
                    for key, value in row.items()
                    if key not in {"surface_profile", "design_status", "legacy_exact",
                                   "sampled_shallow_depths_um_json",
                                   "sampled_deep_depths_um_json",
                                   "sampled_scratch_lengths_m_json"}
                    and value != ""):
                raise RuntimeError(f"non-finite profile row: {profile_id} seed={seed}")
            rows.append(row)

    if legacy_exact_count != args.seeds:
        raise RuntimeError(f"legacy exact mismatch: {legacy_exact_count}/{args.seeds}")

    summaries = []
    for profile_id in PROFILE_IDS:
        block = [row for row in rows if row["surface_profile"] == profile_id]
        numeric = (
            "base_ra_target_um", "roi_base_ra_um", "roi_total_ra_um",
            "roi_total_rz_um", "roi_scratch_max_um", "gu_mean", "gu_p10",
            "gu_min", "clearcoat_min_um",
        )
        summary = {
            "surface_profile": profile_id,
            "n": len(block),
            "finite_count": sum(row["arrays_finite"] for row in block),
            "legacy_exact_count": legacy_exact_count if profile_id == LEGACY_STRESS else "",
            "scratch_count_0": sum(row["shallow_scratch_count"] == 0 for row in block),
            "scratch_count_1": sum(row["shallow_scratch_count"] == 1 for row in block),
            "scratch_count_2": sum(row["shallow_scratch_count"] == 2 for row in block),
        }
        for name in numeric:
            values = np.asarray([float(row[name]) for row in block])
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_min"] = float(values.min())
            summary[f"{name}_max"] = float(values.max())
        summaries.append(summary)

    _write_csv(os.path.join(out_dir, "profile_samples.csv"), rows)
    _write_csv(os.path.join(out_dir, "profile_summary.csv"), summaries)
    with open(os.path.join(out_dir, "parameter_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "design_status": "PT-DESIGN_NO_MEASURED_FACTORY_DISTRIBUTION",
            "legacy_generator": "ESTABLISHED_GATE_B_GENERATOR_UNCHANGED",
            "factory_specs": {key: asdict(value) for key, value in FACTORY_SPECS.items()},
            "surface_state_extended": False,
            "mar_density_or_severity_created": False,
            "ra_rz_gu_formulas_changed": False,
            "clearcoat_removal_temperature_models_changed": False,
            "seeds_per_profile": args.seeds,
            "legacy_exact_count": legacy_exact_count,
            "range_violation_count": range_violation_count,
        }, handle, indent=2, sort_keys=True)
    print(f"factory validation: legacy exact {legacy_exact_count}/{args.seeds}")
    print(f"factory validation: {len(rows)} rows finite, wrote {out_dir}")
    print(f"factory validation: range violations {range_violation_count}")


if __name__ == "__main__":
    main()
