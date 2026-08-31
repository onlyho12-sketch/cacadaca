"""Synthetic sub-grid mar optics for Gate B2.

Mar density/severity are unresolved optical state.  They never modify
`micro_height_um`, individual scratch maps, or the existing Ra/Rz calculation.
All transfer parameters in this module are PT-DESIGN, not measured GU data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gloss_proxy import (
    GOOD_REFINISH_ANCHOR_GU,
    HIGH_GLOSS_ANCHOR_GU,
    TARGET_GU,
    LiteratureGlossProxyModel,
    _tile_slices,
    gu_from_relative,
)
from .surface_profiles import LEGACY_STRESS, NEW_CAR_MILD, SurfaceProfileSample


@dataclass(frozen=True)
class MarOpticsConfig:
    model_id: str = "pt_design_subgrid_mar_optics_v1"
    removal_characteristic_um: float = 0.25
    scatter_gain: float = 8.0
    geometric_weight: float = 1.0
    evidence_tag: str = "PT-DESIGN_NO_MEASURED_MAR_GU_CALIBRATION"


DEFAULT_MAR_OPTICS = MarOpticsConfig()


def remaining_mar_severity(sample: SurfaceProfileSample,
                           cumulative_removal_um: np.ndarray,
                           cfg: MarOpticsConfig = DEFAULT_MAR_OPTICS) -> np.ndarray:
    if cumulative_removal_um.shape != sample.mar_severity_initial.shape:
        raise ValueError("removal and mar maps must have the same shape")
    attenuation = np.exp(-np.clip(cumulative_removal_um, 0.0, None)
                         / max(cfg.removal_characteristic_um, 1e-12))
    return sample.mar_severity_initial * attenuation


def mar_quality_cell_map(sample: SurfaceProfileSample,
                         cumulative_removal_um: np.ndarray,
                         cfg: MarOpticsConfig = DEFAULT_MAR_OPTICS) -> np.ndarray:
    severity = remaining_mar_severity(sample, cumulative_removal_um, cfg)
    exposure = np.clip(sample.mar_density * severity, 0.0, 1.0)
    return np.exp(-cfg.scatter_gain * exposure)


def tile_mean_map(values: np.ndarray, tiles: tuple[int, int] = (5, 5)) -> np.ndarray:
    out = np.zeros(tiles, dtype=float)
    for sx, sy, i, j in _tile_slices(values.shape, tiles):
        out[i, j] = float(values[sx, sy].mean())
    return out


def evaluate_profile_gloss(sample: SurfaceProfileSample, tiles: tuple[int, int] = (5, 5),
                           cfg: MarOpticsConfig = DEFAULT_MAR_OPTICS) -> dict:
    """Evaluate legacy unchanged or new-car GU with an explicit q_mar term."""
    base = LiteratureGlossProxyModel().evaluate(sample.state, tiles=tiles)
    if sample.profile_id == LEGACY_STRESS:
        base["term_maps"]["q_mar"] = np.ones(tiles, dtype=float)
        base["summary"].update({
            "surface_profile": LEGACY_STRESS,
            "mar_optics_used": False,
            "mar_optics_model_id": None,
            "upper_anchor_gu": GOOD_REFINISH_ANCHOR_GU,
        })
        return base
    if sample.profile_id != NEW_CAR_MILD:
        raise ValueError(f"unsupported profile {sample.profile_id!r}")

    q_mar_cell = mar_quality_cell_map(sample, sample.state.cumulative_removal_um, cfg)
    q_mar = tile_mean_map(q_mar_cell, tiles)
    terms = base["term_maps"]
    gloss_cfg = LiteratureGlossProxyModel().cfg
    weights = {
        "q_ra": gloss_cfg.w_ra,
        "q_scratch": gloss_cfg.w_scratch,
        "q_uniformity": gloss_cfg.w_uniformity,
        "q_clearcoat": gloss_cfg.w_clearcoat,
        "q_mar": cfg.geometric_weight,
    }
    geometric_weight_sum = sum(weights.values())
    log_q = np.zeros(tiles, dtype=float)
    for name, weight in weights.items():
        values = q_mar if name == "q_mar" else terms[name]
        log_q += weight * np.log(np.maximum(values, 1e-6))
    log_q /= geometric_weight_sum
    log_q += gloss_cfg.w_thermal * np.log(np.maximum(terms["q_thermal"], 1e-6))
    q_total = np.exp(log_q)
    gu_map = gu_from_relative(q_total, HIGH_GLOSS_ANCHOR_GU)
    term_maps = {**terms, "q_mar": q_mar, "q_total": q_total}
    summary = {
        **base["summary"],
        "gu_mean": float(gu_map.mean()),
        "gu_p10": float(np.percentile(gu_map, 10)),
        "gu_std": float(gu_map.std()),
        "gu_min": float(gu_map.min()),
        "surface_profile": NEW_CAR_MILD,
        "mar_optics_used": True,
        "mar_optics_model_id": cfg.model_id,
        "mar_optics_evidence_tag": cfg.evidence_tag,
        "upper_anchor_gu": HIGH_GLOSS_ANCHOR_GU,
        "gloss_pass": bool(
            gu_map.mean() >= TARGET_GU
            and np.percentile(gu_map, 10) >= gloss_cfg.gu_p10_limit
            and gu_map.std() <= gloss_cfg.gu_std_limit
            and gu_map.min() >= gloss_cfg.gu_min_limit),
        "band_counts": {
            "target_pass": int((gu_map >= 70).sum()),
            "partial": int(((gu_map >= 60) & (gu_map < 70)).sum()),
            "low": int(((gu_map >= 30) & (gu_map < 60)).sum()),
            "severe_defect": int((gu_map < 30).sum()),
        },
    }
    return {"gu_map": gu_map, "term_maps": term_maps, "summary": summary,
            "mar_severity_remaining": remaining_mar_severity(
                sample, sample.state.cumulative_removal_um, cfg),
            "q_mar_cell": q_mar_cell}
