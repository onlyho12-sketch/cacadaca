"""Write Gate B2 initial-distribution and legacy-reproduction evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from learning.polytwin import config as C
from learning.polytwin.mar_optics import (
    DEFAULT_MAR_OPTICS,
    evaluate_profile_gloss,
    mar_quality_cell_map,
)
from learning.polytwin.roughness_metrics import ra_um, rz_um
from learning.polytwin.surface_profiles import (
    LEGACY_STRESS,
    NEW_CAR_MILD,
    NEW_CAR_MILD_SPEC,
    make_surface_profile,
)
from learning.polytwin.surface_state import make_flat_patch


def _state_equal(left, right) -> bool:
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


def _fingerprint(sample) -> str:
    digest = hashlib.sha256()
    for name in sorted(vars(sample.state)):
        value = getattr(sample.state, name)
        digest.update(name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    digest.update(sample.mar_density.tobytes())
    digest.update(sample.mar_severity_initial.tobytes())
    return digest.hexdigest()


def _summarize(rows: list[dict], profile: str) -> dict:
    selected = [row for row in rows if row["surface_profile"] == profile]
    numeric = [
        "base_micro_ra_um", "base_micro_rz_um", "initial_total_ra_um",
        "initial_total_rz_um", "individual_scratch_count",
        "individual_scratch_max_um", "mar_density_mean", "mar_severity_mean",
        "q_mar_mean", "initial_gu_mean", "initial_gu_min",
        "clearcoat_min_um", "clearcoat_mean_um", "clearcoat_max_um",
    ]
    out = {"n": len(selected)}
    for name in numeric:
        values = np.asarray([
            float(row[name]) for row in selected if str(row[name]).strip() != ""
        ])
        if not values.size:
            out[name] = {"available": False, "reason": "not retained by legacy state"}
            continue
        out[name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=256)
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    rows = []
    all_legacy_exact = True
    for seed in range(args.seed_base, args.seed_base + args.num_seeds):
        for profile in (LEGACY_STRESS, NEW_CAR_MILD):
            sample = make_surface_profile(profile, (0.20, 0.20), 0.002, seed)
            state = sample.state
            exact = ""
            if profile == LEGACY_STRESS:
                # Compare the generator output before quality evaluation updates
                # the cached residual-scratch diagnostic in-place.
                direct = make_flat_patch((0.20, 0.20), 0.002, seed=seed)
                exact = _state_equal(direct, state)
                all_legacy_exact &= bool(exact)
            gloss = evaluate_profile_gloss(sample)
            q_mar = mar_quality_cell_map(sample, state.cumulative_removal_um)
            values = [
                sample.base_micro_height_um, state.initial_micro_height_um,
                state.initial_scratch_depth_um, sample.mar_density,
                sample.mar_severity_initial, q_mar, state.initial_clearcoat_um,
                gloss["gu_map"],
            ]
            finite = all(np.isfinite(value).all() for value in values)
            bounds_ok = bool(
                C.CLEARCOAT_MIN_UM - 1e-12 <= state.initial_clearcoat_um.min()
                and state.initial_clearcoat_um.max() <= C.CLEARCOAT_MAX_UM + 1e-12
            )
            if profile == NEW_CAR_MILD:
                spec = NEW_CAR_MILD_SPEC
                sampled_count = int(sample.metadata["individual_scratch_count"])
                bounds_ok &= bool(
                    spec.base_micro_ra_min_um
                    <= float(sample.metadata["base_micro_ra_target_um"])
                    <= spec.base_micro_ra_max_um
                    and min(spec.individual_scratch_count_values) <= sampled_count
                    <= max(spec.individual_scratch_count_values)
                    and 0.0 <= state.initial_scratch_depth_um.min()
                    and state.initial_scratch_depth_um.max()
                    <= spec.visible_scratch_depth_max_um + 1e-12
                    and 0.0 <= sample.mar_density.min()
                    and sample.mar_density.max() <= spec.mar_density_max + 1e-12
                    and 0.0 <= sample.mar_severity_initial.min()
                    and sample.mar_severity_initial.max()
                    <= spec.mar_severity_max + 1e-12
                )
            rows.append({
                "surface_profile": profile, "seed": seed,
                "design_status": sample.metadata["design_status"],
                "legacy_exact_match": exact,
                "declared_bounds_ok": bounds_ok,
                "base_micro_ra_target_um": sample.metadata.get(
                    "base_micro_ra_target_um", C.RA_TARGET_UM),
                "base_micro_ra_um": ra_um(sample.base_micro_height_um),
                "base_micro_rz_um": rz_um(sample.base_micro_height_um),
                "initial_total_ra_um": ra_um(state.initial_micro_height_um),
                "initial_total_rz_um": rz_um(state.initial_micro_height_um),
                "individual_scratch_count": sample.metadata.get(
                    "individual_scratch_count", ""),
                "individual_scratch_count_design_min": (
                    4 if profile == LEGACY_STRESS else 0),
                "individual_scratch_count_design_max": (
                    12 if profile == LEGACY_STRESS else 3),
                "individual_scratch_max_um": float(state.initial_scratch_depth_um.max()),
                "individual_scratch_affected_fraction": float(state.defect_mask.mean()),
                "mar_density_mean": float(sample.mar_density.mean()),
                "mar_density_max": float(sample.mar_density.max()),
                "mar_severity_mean": float(sample.mar_severity_initial.mean()),
                "mar_severity_max": float(sample.mar_severity_initial.max()),
                "q_mar_mean": float(q_mar.mean()),
                "initial_gu_mean": gloss["summary"]["gu_mean"],
                "initial_gu_p10": gloss["summary"]["gu_p10"],
                "initial_gu_min": gloss["summary"]["gu_min"],
                "clearcoat_min_um": float(state.initial_clearcoat_um.min()),
                "clearcoat_mean_um": float(state.initial_clearcoat_um.mean()),
                "clearcoat_max_um": float(state.initial_clearcoat_um.max()),
                "all_finite": finite,
                "state_fingerprint_sha256": _fingerprint(sample),
            })

    csv_path = os.path.join(out_dir, "initial_profile_distributions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_finite = all(bool(row["all_finite"]) for row in rows)
    all_bounds_ok = all(bool(row["declared_bounds_ok"]) for row in rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_seeds_per_profile": args.num_seeds, "seed_base": args.seed_base,
        "patch_size_m": [0.20, 0.20], "resolution_m": 0.002,
        "legacy_exact_all": all_legacy_exact, "all_finite": all_finite,
        "all_declared_bounds_ok": all_bounds_ok,
        "legacy_stress": _summarize(rows, LEGACY_STRESS),
        "new_car_mild": _summarize(rows, NEW_CAR_MILD),
        "new_car_mild_spec": asdict(NEW_CAR_MILD_SPEC),
        "mar_optics_config": asdict(DEFAULT_MAR_OPTICS),
        "caveat": ("new_car_mild scratch count/depth/mixture and mar distributions are "
                    "PT-DESIGN because no measured joint distribution is available."),
    }
    with open(os.path.join(out_dir, "initial_profile_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "parameter_provenance.json"),
              "w", encoding="utf-8") as handle:
        json.dump({
            "new_car_mild_spec": asdict(NEW_CAR_MILD_SPEC),
            "mar_optics_config": asdict(DEFAULT_MAR_OPTICS),
        }, handle, indent=2, sort_keys=True)

    print(f"[Gate B2] wrote {out_dir}")
    print(f"[Gate B2] legacy exact={all_legacy_exact} seeds={args.num_seeds}")
    print(f"[Gate B2] all finite={all_finite} rows={len(rows)}")
    print(f"[Gate B2] all declared bounds={all_bounds_ok}")
    if not all_legacy_exact or not all_finite or not all_bounds_ok:
        raise RuntimeError("Gate B2 initial profile validation failed")


if __name__ == "__main__":
    main()
