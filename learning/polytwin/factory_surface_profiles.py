"""PT-DESIGN factory pre-polish surface profiles using the established SurfaceState.

Only initial roughness and scratch-generation parameter ranges differ by profile.
No optical mar layer or new per-cell state is introduced.  Ra/Rz/GU, clearcoat,
removal, and thermal calculations remain the established implementations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import config as C
from .roughness_metrics import ra_um
from .surface_state import SurfaceState, _distance_to_segment, make_flat_patch

LEGACY_STRESS = "legacy_stress"
FACTORY_PREPOLISH = "factory_prepolish"
FACTORY_PREPOLISH_DEEP_DEFECT = "factory_prepolish_deep_defect"
FACTORY_PREPOLISH_DEEP_STRESS = "factory_prepolish_deep_stress"

FACTORY_PROFILE_IDS = (
    FACTORY_PREPOLISH,
    FACTORY_PREPOLISH_DEEP_DEFECT,
    FACTORY_PREPOLISH_DEEP_STRESS,
)
PROFILE_IDS = (LEGACY_STRESS, *FACTORY_PROFILE_IDS)


@dataclass(frozen=True)
class FactorySurfaceSpec:
    """Synthetic test ranges; these are not measured production distributions."""

    profile_id: str
    base_ra_min_um: float = 0.10
    base_ra_max_um: float = 0.12
    shallow_scratch_count_min: int = 0
    shallow_scratch_count_max: int = 2
    shallow_depth_min_um: float = 0.10
    shallow_depth_max_um: float = 0.60
    scratch_length_min_m: float = 0.010
    scratch_length_max_m: float = 0.080
    scratch_width_m: float = C.SCRATCH_WIDTH_M
    deep_depth_min_um: float | None = None
    deep_depth_max_um: float | None = None
    design_status: str = "PT-DESIGN_NO_MEASURED_FACTORY_DISTRIBUTION"

    @property
    def deep_scratch_count(self) -> int:
        return int(self.deep_depth_min_um is not None)


FACTORY_SPECS = {
    FACTORY_PREPOLISH: FactorySurfaceSpec(profile_id=FACTORY_PREPOLISH),
    FACTORY_PREPOLISH_DEEP_DEFECT: FactorySurfaceSpec(
        profile_id=FACTORY_PREPOLISH_DEEP_DEFECT,
        deep_depth_min_um=0.60,
        deep_depth_max_um=1.00,
    ),
    FACTORY_PREPOLISH_DEEP_STRESS: FactorySurfaceSpec(
        profile_id=FACTORY_PREPOLISH_DEEP_STRESS,
        deep_depth_min_um=1.00,
        deep_depth_max_um=1.50,
    ),
}


def _scratch_grid(size_m: tuple[float, float], resolution_m: float):
    nx = int(round(size_m[0] / resolution_m))
    ny = int(round(size_m[1] / resolution_m))
    x = (np.arange(nx) + 0.5) * resolution_m
    y = (np.arange(ny) + 0.5) * resolution_m
    return np.meshgrid(x, y, indexing="ij")


def _sample_segment(
    xx: np.ndarray,
    yy: np.ndarray,
    size_m: tuple[float, float],
    rng: np.random.Generator,
    depth_min_um: float,
    depth_max_um: float,
    length_min_m: float,
    length_max_m: float,
    width_m: float,
) -> tuple[np.ndarray, dict]:
    """Use the established independent center/angle/segment/Gaussian-groove formula."""
    depth_um = float(rng.uniform(depth_min_um, depth_max_um))
    length_m = float(rng.uniform(length_min_m, length_max_m))
    angle_rad = float(rng.uniform(0.0, np.pi))
    center_x_m = float(rng.uniform(0.0, size_m[0]))
    center_y_m = float(rng.uniform(0.0, size_m[1]))
    half = 0.5 * length_m * np.asarray([np.cos(angle_rad), np.sin(angle_rad)])
    distance = _distance_to_segment(
        xx,
        yy,
        (center_x_m - half[0], center_y_m - half[1]),
        (center_x_m + half[0], center_y_m + half[1]),
    )
    groove = depth_um * np.exp(-(distance / width_m) ** 2)
    return groove, {
        "depth_um": depth_um,
        "length_m": length_m,
        "angle_rad": angle_rad,
        "center_x_m": center_x_m,
        "center_y_m": center_y_m,
    }


def _factory_scratch_map(
    roi_size_m: tuple[float, float],
    resolution_m: float,
    seed: int,
    spec: FactorySurfaceSpec,
) -> tuple[np.ndarray, dict]:
    xx, yy = _scratch_grid(roi_size_m, resolution_m)
    scratches = np.zeros_like(xx, dtype=float)

    # All factory profiles share the same shallow state for a given profile seed.
    shallow_rng = np.random.default_rng(seed + 3_000_017)
    shallow_count = int(shallow_rng.integers(
        spec.shallow_scratch_count_min, spec.shallow_scratch_count_max + 1))
    shallow_segments = []
    for _ in range(shallow_count):
        groove, sampled = _sample_segment(
            xx, yy, roi_size_m, shallow_rng,
            spec.shallow_depth_min_um, spec.shallow_depth_max_um,
            spec.scratch_length_min_m, spec.scratch_length_max_m,
            spec.scratch_width_m,
        )
        scratches = np.maximum(scratches, groove)
        shallow_segments.append(sampled)

    deep_segments = []
    if spec.deep_scratch_count:
        # The two deep profiles share geometry and a unit severity draw.  Only the
        # declared depth interval differs, isolating controllable vs stress tails.
        deep_rng = np.random.default_rng(seed + 4_000_037)
        groove, sampled = _sample_segment(
            xx, yy, roi_size_m, deep_rng,
            float(spec.deep_depth_min_um), float(spec.deep_depth_max_um),
            spec.scratch_length_min_m, spec.scratch_length_max_m,
            spec.scratch_width_m,
        )
        scratches = np.maximum(scratches, groove)
        deep_segments.append(sampled)

    return scratches, {
        "shallow_scratch_count": shallow_count,
        "deep_scratch_count": spec.deep_scratch_count,
        "shallow_segments": shallow_segments,
        "deep_segments": deep_segments,
    }


def _centered_slices(
    map_size_m: tuple[float, float],
    roi_size_m: tuple[float, float],
    resolution_m: float,
) -> tuple[slice, slice]:
    full = np.rint(np.asarray(map_size_m) / resolution_m).astype(int)
    roi = np.rint(np.asarray(roi_size_m) / resolution_m).astype(int)
    if np.any(roi > full) or np.any((full - roi) % 2):
        raise ValueError("ROI must fit and be centered on cell boundaries")
    start = (full - roi) // 2
    return (
        slice(int(start[0]), int(start[0] + roi[0])),
        slice(int(start[1]), int(start[1] + roi[1])),
    )


def _legacy_centered_patch(
    map_size_m: tuple[float, float],
    roi_size_m: tuple[float, float],
    resolution_m: float,
    seed: int,
) -> SurfaceState:
    """Byte-for-byte equivalent construction to Gate B make_centered_roi_patch."""
    sl = _centered_slices(map_size_m, roi_size_m, resolution_m)
    outer = make_flat_patch(
        map_size_m, resolution_m, seed=seed + 1_000_003, with_scratches=False)
    defect_source = make_flat_patch(
        roi_size_m, resolution_m, seed=seed, with_scratches=True)
    scratches = defect_source.initial_scratch_depth_um
    outer.micro_height_um[sl] -= scratches
    outer.initial_micro_height_um[sl] -= scratches
    outer.initial_scratch_depth_um[sl] = scratches
    outer.residual_scratch_depth_um[sl] = scratches
    outer.defect_mask[sl] = defect_source.defect_mask
    outer.healthy_mask[sl] = ~defect_source.defect_mask
    return outer


def make_factory_centered_patch(
    profile_id: str,
    map_size_m: tuple[float, float] = (0.32, 0.32),
    roi_size_m: tuple[float, float] = (0.20, 0.20),
    resolution_m: float = 0.002,
    seed: int = 0,
) -> tuple[SurfaceState, dict]:
    """Create an established SurfaceState plus scalar provenance metadata."""
    if profile_id == LEGACY_STRESS:
        state = _legacy_centered_patch(map_size_m, roi_size_m, resolution_m, seed)
        return state, {
            "profile_id": LEGACY_STRESS,
            "profile_seed": seed,
            "design_status": "ESTABLISHED_GATE_B_GENERATOR_UNCHANGED",
            "base_ra_target_um": C.RA_TARGET_UM,
            "scratch_count_range": [C.SCRATCH_COUNT_MIN, C.SCRATCH_COUNT_MAX],
            "scratch_depth_range_um": [C.SCRATCH_DEPTH_MIN_UM, C.SCRATCH_DEPTH_MAX_UM],
        }
    if profile_id not in FACTORY_SPECS:
        raise ValueError(f"unknown profile_id={profile_id!r}; expected one of {PROFILE_IDS}")

    spec = FACTORY_SPECS[profile_id]
    parameter_rng = np.random.default_rng(seed + 5_000_041)
    target_ra_um = float(parameter_rng.uniform(spec.base_ra_min_um, spec.base_ra_max_um))
    state = make_flat_patch(
        map_size_m,
        resolution_m,
        seed=seed + 2_000_003,
        target_ra_um=target_ra_um,
        with_scratches=False,
    )
    sl = _centered_slices(map_size_m, roi_size_m, resolution_m)
    # The central ROI is the declared quality area.  Scale the continuous full-map
    # base field once so its ROI Ra, evaluated by the established detrended formula,
    # equals the sampled PT-DESIGN target without introducing a boundary seam.
    roi_ra_before_scale = ra_um(state.initial_micro_height_um[sl])
    base_scale = target_ra_um / max(roi_ra_before_scale, 1e-12)
    state.micro_height_um *= base_scale
    state.initial_micro_height_um *= base_scale
    scratches, scratch_metadata = _factory_scratch_map(
        roi_size_m, resolution_m, seed, spec)
    state.micro_height_um[sl] -= scratches
    state.initial_micro_height_um[sl] -= scratches
    state.initial_scratch_depth_um[sl] = scratches
    state.residual_scratch_depth_um[sl] = scratches
    state.defect_mask[sl] = scratches > (0.5 * C.SCRATCH_DEPTH_MIN_UM)
    state.healthy_mask[:] = ~state.defect_mask
    state.seed = seed
    metadata = {
        "profile_id": profile_id,
        "profile_seed": seed,
        "design_status": spec.design_status,
        "base_state_seed": seed + 2_000_003,
        "shallow_scratch_seed": seed + 3_000_017,
        "deep_scratch_seed": seed + 4_000_037 if spec.deep_scratch_count else None,
        "parameter_seed": seed + 5_000_041,
        "base_ra_target_um": target_ra_um,
        "base_ra_roi_before_scale_um": roi_ra_before_scale,
        "base_height_scale": base_scale,
        "spec": asdict(spec),
        **scratch_metadata,
    }
    return state, metadata
