"""Isaac-free Gate D observation schema and inspection encoder.

The extension is deliberately inspection-cached: initial inspection values are
used during pass 1, and the most recent pass-end inspection is used afterwards.
No full-surface diagnostic is recomputed at control frequency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BASE14 = "base14"
GLOBAL20 = "global20"
SPATIAL120 = "spatial120"
OBSERVATION_MODES = (BASE14, GLOBAL20, SPATIAL120)
BASE_OBSERVATION_DIM = 14
GLOBAL_FEATURE_DIM = 6
TILE_SHAPE = (5, 5)
TILE_CHANNELS = (
    "scratch_max_um",
    "removal_mean_um",
    "under_fraction",
    "over_fraction",
)

BASE_FEATURE_NAMES = (
    "force_mean_norm",
    "force_error_norm",
    "force_delta_norm",
    "feed_cmd_norm",
    "path_progress",
    "local_scratch_mean_norm",
    "local_scratch_max_norm",
    "local_removal_norm",
    "local_clearcoat_margin_norm",
    "previous_force_action",
    "previous_feed_action",
    "local_temperature_mean_norm",
    "local_temperature_peak_norm",
    "local_thermal_damage_norm",
)
GLOBAL_FEATURE_NAMES = (
    "inspection_pass_norm",
    "gu_gap_norm",
    "ra_gap_norm",
    "rz_gap_norm",
    "clearcoat_margin_norm",
    "center_edge_removal_delta_norm",
)


@dataclass(frozen=True)
class GateDObservationTargets:
    target_gu: float = 70.0
    ra_limit_um: float = 0.20
    rz_limit_um: float = 2.0
    clearcoat_safety_limit_um: float = 30.0
    max_passes: int = 6

    def validate(self) -> None:
        values = (
            self.target_gu,
            self.ra_limit_um,
            self.rz_limit_um,
            self.clearcoat_safety_limit_um,
        )
        if not all(np.isfinite(values)) or min(values) <= 0.0:
            raise ValueError("Gate D targets must be finite and positive")
        if self.max_passes <= 0:
            raise ValueError("max_passes must be positive")


def observation_dim(mode: str) -> int:
    if mode == BASE14:
        return BASE_OBSERVATION_DIM
    if mode == GLOBAL20:
        return BASE_OBSERVATION_DIM + GLOBAL_FEATURE_DIM
    if mode == SPATIAL120:
        return (BASE_OBSERVATION_DIM + GLOBAL_FEATURE_DIM
                + len(TILE_CHANNELS) * TILE_SHAPE[0] * TILE_SHAPE[1])
    raise ValueError(f"unknown Gate D observation mode: {mode!r}")


def observation_feature_names(mode: str) -> tuple[str, ...]:
    names = list(BASE_FEATURE_NAMES)
    if mode in (GLOBAL20, SPATIAL120):
        names.extend(GLOBAL_FEATURE_NAMES)
    elif mode != BASE14:
        raise ValueError(f"unknown Gate D observation mode: {mode!r}")
    if mode == SPATIAL120:
        for channel in TILE_CHANNELS:
            for tile_x, tile_y in np.ndindex(TILE_SHAPE):
                names.append(f"tile_{channel}_x{tile_x}_y{tile_y}_norm")
    if len(names) != observation_dim(mode):
        raise RuntimeError("Gate D feature schema length mismatch")
    return tuple(names)


def _finite_float(mapping: dict, key: str) -> float:
    value = float(mapping[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite Gate D scalar {key}={value}")
    return value


def encode_inspection_extension(
    scalars: dict,
    tile_maps: dict,
    inspection_pass: int,
    targets: GateDObservationTargets,
) -> np.ndarray:
    """Return the 106 normalized values appended by ``spatial120``.

    Normalization constants are PT-DESIGN feature scales, not measured process
    distributions. Values are clipped only to bound future policy inputs.
    """
    targets.validate()
    if inspection_pass < 0:
        raise ValueError("inspection_pass must be non-negative")
    gu = _finite_float(scalars, "profile_gu_mean")
    ra = _finite_float(scalars, "roi_total_ra_um")
    rz = _finite_float(scalars, "roi_total_rz_um")
    clearcoat = _finite_float(scalars, "roi_clearcoat_min_um")
    center_edge = _finite_float(scalars, "roi_center_edge_delta_um")
    global_values = np.asarray([
        np.clip(inspection_pass / targets.max_passes, 0.0, 1.0),
        np.clip((targets.target_gu - gu) / 10.0, -3.0, 3.0),
        np.clip((ra - targets.ra_limit_um) / targets.ra_limit_um, -1.0, 5.0),
        np.clip((rz - targets.rz_limit_um) / targets.rz_limit_um, -1.0, 5.0),
        np.clip((clearcoat - targets.clearcoat_safety_limit_um) / 20.0, -1.0, 2.0),
        np.clip(center_edge / 1.0, -2.0, 2.0),
    ], dtype=np.float32)

    tile_values = []
    for channel in TILE_CHANNELS:
        values = np.asarray(tile_maps[channel], dtype=np.float64)
        if values.shape != TILE_SHAPE:
            raise ValueError(
                f"Gate D tile channel {channel} shape={values.shape}, expected={TILE_SHAPE}")
        if not np.isfinite(values).all():
            raise ValueError(f"Gate D tile channel {channel} contains NaN/Inf")
        if channel == "scratch_max_um":
            normalized = np.clip(values / 2.0, 0.0, 2.0)
        elif channel == "removal_mean_um":
            normalized = np.clip(values / 5.0, 0.0, 2.0)
        else:
            normalized = np.clip(values, 0.0, 1.0)
        tile_values.extend(normalized.reshape(-1).tolist())
    extension = np.concatenate([
        global_values,
        np.asarray(tile_values, dtype=np.float32),
    ])
    if extension.shape != (106,) or not np.isfinite(extension).all():
        raise RuntimeError("invalid Gate D inspection extension")
    return extension


def compose_observation(base14: np.ndarray, extension106: np.ndarray,
                        mode: str) -> np.ndarray:
    base = np.asarray(base14, dtype=np.float32)
    extension = np.asarray(extension106, dtype=np.float32)
    if base.shape[-1] != BASE_OBSERVATION_DIM:
        raise ValueError(f"base observation must end in 14 values, got {base.shape}")
    if extension.shape[-1] != 106:
        raise ValueError(f"inspection extension must end in 106 values, got {extension.shape}")
    if mode == BASE14:
        out = base.copy()
    elif mode == GLOBAL20:
        out = np.concatenate([base, extension[..., :GLOBAL_FEATURE_DIM]], axis=-1)
    elif mode == SPATIAL120:
        out = np.concatenate([base, extension], axis=-1)
    else:
        raise ValueError(f"unknown Gate D observation mode: {mode!r}")
    if out.shape[-1] != observation_dim(mode) or not np.isfinite(out).all():
        raise RuntimeError("invalid composed Gate D observation")
    return out


def feature_schema_rows() -> list[dict]:
    rows = []
    for index, name in enumerate(observation_feature_names(SPATIAL120)):
        if index < BASE_OBSERVATION_DIM:
            group, status = "established_base14", "ESTABLISHED_UNCHANGED"
        elif index < BASE_OBSERVATION_DIM + GLOBAL_FEATURE_DIM:
            group, status = "inspection_global", "PT-DESIGN_NORMALIZATION"
        else:
            group, status = "inspection_tile_5x5", "PT-DESIGN_NORMALIZATION"
        rows.append({
            "feature_index": index,
            "feature_name": name,
            "feature_group": group,
            "base14_included": index < observation_dim(BASE14),
            "global20_included": index < observation_dim(GLOBAL20),
            "spatial120_included": True,
            "design_status": status,
        })
    return rows
