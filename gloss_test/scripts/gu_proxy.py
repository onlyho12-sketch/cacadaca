"""Literature-anchored 20-degree GU proxy utilities.

This module deliberately does not claim Gloss Meter calibration.  It maps the
existing pristine-normalized optical response to a simulation-only GU scale
anchored to values reported by Ulbrich et al. (Coatings 2021, 11, 1320).
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LiteratureGuProxyConfig:
    """Versioned anchors and decision threshold for the optical-only v1 model."""

    model_id: str = "literature_gu_proxy_optical_linear_v1"
    source_doi: str = "10.3390/coatings11111320"
    defective_anchor_gu: float = 25.0
    good_refinish_anchor_gu: float = 78.0
    high_gloss_vehicle_anchor_gu: float = 89.0
    target_gu: float = 70.0
    relative_gloss_min: float = 0.0
    relative_gloss_max: float = 1.0
    actual_gloss_meter_calibrated: bool = False
    output_metric: str = "predicted_20deg_gu_literature_proxy"

    def metadata(self):
        payload = asdict(self)
        payload.update({
            "is_measured_gu": False,
            "evidence_tag": "L-DERIVED",
            "mapping_formula": (
                "defective_anchor_gu + clipped_relative_gloss * "
                "(good_refinish_anchor_gu - defective_anchor_gu)"
            ),
            "meaning": (
                "Simulation-only 20-degree GU proxy anchored to literature; "
                "not a Gloss Meter measurement"
            ),
        })
        return payload


def relative_gloss_to_gu_proxy(relative_gloss, config=None):
    """Map scalar/array pristine-relative gloss to the literature GU proxy."""
    config = config or LiteratureGuProxyConfig()
    values = np.asarray(relative_gloss, dtype=float)
    clipped = np.clip(
        values,
        config.relative_gloss_min,
        config.relative_gloss_max,
    )
    unit = (
        (clipped - config.relative_gloss_min)
        / (config.relative_gloss_max - config.relative_gloss_min)
    )
    mapped = config.defective_anchor_gu + unit * (
        config.good_refinish_anchor_gu - config.defective_anchor_gu
    )
    if np.ndim(relative_gloss) == 0:
        return float(mapped)
    return mapped


def gu_proxy_passes(gu_proxy, config=None):
    """Return target pass flags for scalar/array proxy GU values."""
    config = config or LiteratureGuProxyConfig()
    passed = np.asarray(gu_proxy, dtype=float) >= config.target_gu
    if np.ndim(gu_proxy) == 0:
        return bool(passed)
    return passed

