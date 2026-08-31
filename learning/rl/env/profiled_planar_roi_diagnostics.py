"""Pure diagnostics for Gate B2 profiled planar ROI surfaces."""
from __future__ import annotations

import copy

import numpy as np

from learning.polytwin.mar_optics import (
    DEFAULT_MAR_OPTICS,
    evaluate_profile_gloss,
    mar_quality_cell_map,
    remaining_mar_severity,
)
from learning.polytwin.roughness_metrics import ra_um, rz_um
from learning.polytwin.surface_profiles import SurfaceProfileSample

from .planar_roi_diagnostics import (
    PlanarRoiGeometry,
    _tile_slices,
    centered_roi_slices,
    diagnose_roi,
    surface_view,
)


def profile_sample_view(sample: SurfaceProfileSample,
                        slices: tuple[slice, slice]) -> SurfaceProfileSample:
    view = SurfaceProfileSample(
        state=surface_view(sample.state, slices),
        profile_id=sample.profile_id,
        base_micro_height_um=sample.base_micro_height_um[slices],
        mar_density=sample.mar_density[slices],
        mar_severity_initial=sample.mar_severity_initial[slices],
        metadata=copy.deepcopy(sample.metadata),
    )
    view.validate()
    return view


def diagnose_profile_roi(sample: SurfaceProfileSample, geometry: PlanarRoiGeometry,
                         waviness_sigma_m: float = 0.010,
                         under_over_fraction: float = 0.20,
                         tiles: tuple[int, int] = (5, 5)) -> dict:
    slices = centered_roi_slices(geometry)
    roi = profile_sample_view(sample, slices)
    base = diagnose_roi(
        sample.state, geometry, waviness_sigma_m=waviness_sigma_m,
        under_over_fraction=under_over_fraction, tiles=tiles)
    gloss = evaluate_profile_gloss(roi, tiles=tiles)
    mar_remaining = remaining_mar_severity(
        roi, roi.state.cumulative_removal_um, DEFAULT_MAR_OPTICS)
    q_mar_cell = mar_quality_cell_map(
        roi, roi.state.cumulative_removal_um, DEFAULT_MAR_OPTICS)

    scalars = {
        "surface_profile": sample.profile_id,
        "profile_seed": int(sample.metadata["profile_seed"]),
        "profile_base_micro_ra_um": ra_um(roi.base_micro_height_um),
        "profile_base_micro_rz_um": rz_um(roi.base_micro_height_um),
        "profile_individual_scratch_initial_max_um": float(
            roi.state.initial_scratch_depth_um.max()),
        "profile_individual_scratch_affected_fraction": float(
            roi.state.defect_mask.mean()),
        "profile_mar_density_mean": float(roi.mar_density.mean()),
        "profile_mar_density_max": float(roi.mar_density.max()),
        "profile_mar_severity_initial_mean": float(roi.mar_severity_initial.mean()),
        "profile_mar_severity_remaining_mean": float(mar_remaining.mean()),
        "profile_mar_severity_remaining_max": float(mar_remaining.max()),
        "profile_q_mar_mean": float(q_mar_cell.mean()),
        "profile_q_mar_min": float(q_mar_cell.min()),
        "profile_gu_mean": float(gloss["summary"]["gu_mean"]),
        "profile_gu_p10": float(gloss["summary"]["gu_p10"]),
        "profile_gu_min": float(gloss["summary"]["gu_min"]),
        "profile_gu_std": float(gloss["summary"]["gu_std"]),
        "profile_gloss_pass": bool(gloss["summary"]["gloss_pass"]),
        "profile_mar_model_id": DEFAULT_MAR_OPTICS.model_id,
        "profile_mar_evidence_tag": DEFAULT_MAR_OPTICS.evidence_tag,
        **base["scalars"],
    }

    maps = {name: values.copy() for name, values in base["tile_maps"].items()}
    maps.update({name: np.zeros(tiles, dtype=float) for name in (
        "base_micro_ra_um", "base_micro_rz_um", "individual_scratch_initial_max_um",
        "mar_density_mean", "mar_severity_initial_mean",
        "mar_severity_remaining_mean", "q_mar_mean",
    )})
    for sx, sy, i, j in _tile_slices(roi.state.shape, tiles):
        maps["base_micro_ra_um"][i, j] = ra_um(roi.base_micro_height_um[sx, sy])
        maps["base_micro_rz_um"][i, j] = rz_um(roi.base_micro_height_um[sx, sy])
        maps["individual_scratch_initial_max_um"][i, j] = (
            roi.state.initial_scratch_depth_um[sx, sy].max())
        maps["mar_density_mean"][i, j] = roi.mar_density[sx, sy].mean()
        maps["mar_severity_initial_mean"][i, j] = (
            roi.mar_severity_initial[sx, sy].mean())
        maps["mar_severity_remaining_mean"][i, j] = mar_remaining[sx, sy].mean()
        maps["q_mar_mean"][i, j] = q_mar_cell[sx, sy].mean()
    maps["profile_gu"] = np.asarray(gloss["gu_map"], dtype=float)
    return {"scalars": scalars, "tile_maps": maps,
            "profile_metadata": copy.deepcopy(sample.metadata)}
