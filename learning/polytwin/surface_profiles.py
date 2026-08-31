"""Isolated surface profiles for Gate B2.

`legacy_stress` is an adapter around the established `make_flat_patch` and must
remain array-identical to it.  `new_car_mild` is a separate PT-DESIGN profile.
Sub-grid swirl/mar is stored as optical state and is never carved into the
2 mm quality height grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config as C
from .surface_state import SurfaceState, make_flat_patch

LEGACY_STRESS = "legacy_stress"
NEW_CAR_MILD = "new_car_mild"
PROFILE_IDS = (LEGACY_STRESS, NEW_CAR_MILD)


@dataclass(frozen=True)
class NewCarMildSpec:
    """PT-DESIGN ranges, not measured new-car defect distributions."""

    profile_id: str = NEW_CAR_MILD
    design_status: str = "PT-DESIGN_NO_MEASURED_DISTRIBUTION"
    base_micro_ra_min_um: float = 0.035
    base_micro_ra_max_um: float = 0.065
    individual_scratch_count_values: tuple[int, ...] = (0, 1, 2, 3)
    individual_scratch_count_probabilities: tuple[float, ...] = (0.45, 0.35, 0.15, 0.05)
    light_scratch_mixture_fraction: float = 0.80
    light_scratch_depth_min_um: float = 0.05
    light_scratch_depth_max_um: float = 0.20
    visible_scratch_depth_min_um: float = 0.20
    visible_scratch_depth_max_um: float = 0.60
    individual_scratch_length_min_m: float = 0.010
    individual_scratch_length_max_m: float = 0.080
    individual_scratch_width_m: float = 0.001
    mar_density_mean_min: float = 0.03
    mar_density_mean_max: float = 0.10
    mar_density_max: float = 0.20
    mar_severity_mean_min: float = 0.04
    mar_severity_mean_max: float = 0.16
    mar_severity_max: float = 0.25
    mar_density_correlation_m: float = 0.012
    mar_severity_correlation_m: float = 0.018
    mar_spatial_variation: float = 0.45
    tags: dict[str, str] = field(default_factory=lambda: {
        "profile": "PT-DESIGN",
        "base_micro_ra_range": "PT-DESIGN",
        "scratch_count_and_mixture": "PT-DESIGN_NO_MEASURED_DISTRIBUTION",
        "scratch_depth_range": "PT-DESIGN_NO_MEASURED_DISTRIBUTION",
        "scratch_geometry": "PT-DESIGN_NO_MEASURED_DISTRIBUTION",
        "mar_density_and_severity": "PT-DESIGN_NO_MEASURED_DISTRIBUTION",
        "clearcoat_range": "existing L-DERIVED 40-50 um",
    })


NEW_CAR_MILD_SPEC = NewCarMildSpec()


@dataclass
class SurfaceProfileSample:
    state: SurfaceState
    profile_id: str
    base_micro_height_um: np.ndarray
    mar_density: np.ndarray
    mar_severity_initial: np.ndarray
    metadata: dict

    def validate(self) -> None:
        shape = self.state.shape
        for name in ("base_micro_height_um", "mar_density", "mar_severity_initial"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != surface shape {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if np.any(self.mar_density < 0.0) or np.any(self.mar_density > 1.0):
            raise ValueError("mar_density must be within [0, 1]")
        if np.any(self.mar_severity_initial < 0.0) or np.any(self.mar_severity_initial > 1.0):
            raise ValueError("mar_severity_initial must be within [0, 1]")


def _distance_to_segment(xx: np.ndarray, yy: np.ndarray,
                         p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    direction = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    length_sq = float(direction @ direction)
    if length_sq < 1e-12:
        return np.hypot(xx - p0[0], yy - p0[1])
    t = ((xx - p0[0]) * direction[0] + (yy - p0[1]) * direction[1]) / length_sq
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(xx - (p0[0] + t * direction[0]),
                    yy - (p0[1] + t * direction[1]))


def _correlated_unit_field(shape: tuple[int, int], correlation_m: float,
                           resolution_m: float, rng: np.random.Generator) -> np.ndarray:
    sigma = max(correlation_m / resolution_m, 0.5)
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    field -= field.mean()
    scale = max(float(field.std()), 1e-12)
    return np.clip(field / scale, -2.0, 2.0) / 2.0


def _make_individual_scratch_map(size_m: tuple[float, float], resolution_m: float,
                                 rng: np.random.Generator,
                                 spec: NewCarMildSpec) -> tuple[np.ndarray, dict]:
    nx = int(round(size_m[0] / resolution_m))
    ny = int(round(size_m[1] / resolution_m))
    x = (np.arange(nx) + 0.5) * resolution_m
    y = (np.arange(ny) + 0.5) * resolution_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    scratch_map = np.zeros((nx, ny), dtype=float)
    count = int(rng.choice(spec.individual_scratch_count_values,
                           p=spec.individual_scratch_count_probabilities))
    sampled_depths = []
    sampled_lengths = []
    for _ in range(count):
        if rng.random() < spec.light_scratch_mixture_fraction:
            depth = float(rng.uniform(spec.light_scratch_depth_min_um,
                                      spec.light_scratch_depth_max_um))
            component = "light"
        else:
            depth = float(rng.uniform(spec.visible_scratch_depth_min_um,
                                      spec.visible_scratch_depth_max_um))
            component = "visible"
        length = float(rng.uniform(spec.individual_scratch_length_min_m,
                                   spec.individual_scratch_length_max_m))
        angle = float(rng.uniform(0.0, np.pi))
        cx = float(rng.uniform(0.0, size_m[0]))
        cy = float(rng.uniform(0.0, size_m[1]))
        half = 0.5 * length * np.asarray([np.cos(angle), np.sin(angle)])
        distance = _distance_to_segment(
            xx, yy, (cx - half[0], cy - half[1]), (cx + half[0], cy + half[1]))
        groove = depth * np.exp(-(distance / spec.individual_scratch_width_m) ** 2)
        scratch_map = np.maximum(scratch_map, groove)
        sampled_depths.append({"component": component, "depth_um": depth})
        sampled_lengths.append(length)
    return scratch_map, {
        "individual_scratch_count": count,
        "sampled_scratch_depths": sampled_depths,
        "sampled_scratch_lengths_m": sampled_lengths,
    }


def _make_mar_maps(shape: tuple[int, int], resolution_m: float, seed: int,
                   spec: NewCarMildSpec) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    density_mean = float(rng.uniform(spec.mar_density_mean_min, spec.mar_density_mean_max))
    severity_mean = float(rng.uniform(spec.mar_severity_mean_min, spec.mar_severity_mean_max))
    density_field = _correlated_unit_field(
        shape, spec.mar_density_correlation_m, resolution_m, rng)
    severity_field = _correlated_unit_field(
        shape, spec.mar_severity_correlation_m, resolution_m, rng)
    density = np.clip(
        density_mean * (1.0 + spec.mar_spatial_variation * density_field),
        0.0, spec.mar_density_max)
    severity = np.clip(
        severity_mean * (1.0 + spec.mar_spatial_variation * severity_field),
        0.0, spec.mar_severity_max)
    return density, severity, {
        "mar_density_target_mean": density_mean,
        "mar_severity_target_mean": severity_mean,
    }


def _new_car_base(size_m: tuple[float, float], resolution_m: float, seed: int,
                  spec: NewCarMildSpec) -> tuple[SurfaceState, np.ndarray, dict]:
    parameter_rng = np.random.default_rng(seed + 4_000_037)
    target_ra = float(parameter_rng.uniform(
        spec.base_micro_ra_min_um, spec.base_micro_ra_max_um))
    state = make_flat_patch(
        size_m, resolution_m, seed=seed + 2_000_003,
        target_ra_um=target_ra, with_scratches=False)
    state.seed = seed
    return state, state.initial_micro_height_um.copy(), {
        "base_micro_ra_target_um": target_ra,
        "base_state_seed": seed + 2_000_003,
    }


def make_new_car_mild_patch(patch_size_m: tuple[float, float] = (0.20, 0.20),
                            resolution_m: float = 0.002, seed: int = 0,
                            spec: NewCarMildSpec = NEW_CAR_MILD_SPEC) -> SurfaceProfileSample:
    state, base_micro, metadata = _new_car_base(patch_size_m, resolution_m, seed, spec)
    scratch_rng = np.random.default_rng(seed + 3_000_017)
    scratch, scratch_metadata = _make_individual_scratch_map(
        patch_size_m, resolution_m, scratch_rng, spec)
    state.micro_height_um -= scratch
    state.initial_micro_height_um -= scratch
    state.initial_scratch_depth_um[:] = scratch
    state.residual_scratch_depth_um[:] = scratch
    state.defect_mask[:] = scratch > (0.5 * spec.light_scratch_depth_min_um)
    state.healthy_mask[:] = ~state.defect_mask
    mar_density, mar_severity, mar_metadata = _make_mar_maps(
        state.shape, resolution_m, seed + 5_000_041, spec)
    sample = SurfaceProfileSample(
        state=state, profile_id=NEW_CAR_MILD, base_micro_height_um=base_micro,
        mar_density=mar_density, mar_severity_initial=mar_severity,
        metadata={
            "profile_id": NEW_CAR_MILD, "design_status": spec.design_status,
            "profile_seed": seed, "scratch_seed": seed + 3_000_017,
            "mar_seed": seed + 5_000_041, "parameter_tags": dict(spec.tags),
            **metadata, **scratch_metadata, **mar_metadata,
        },
    )
    sample.validate()
    return sample


def make_new_car_mild_centered_patch(
        map_size_m: tuple[float, float], roi_size_m: tuple[float, float],
        resolution_m: float, seed: int,
        spec: NewCarMildSpec = NEW_CAR_MILD_SPEC) -> SurfaceProfileSample:
    """Continuous full map with ROI-local individual scratches and full-map mar optics."""
    state, base_micro, metadata = _new_car_base(map_size_m, resolution_m, seed, spec)
    full = np.rint(np.asarray(map_size_m) / resolution_m).astype(int)
    roi = np.rint(np.asarray(roi_size_m) / resolution_m).astype(int)
    if np.any(roi > full) or np.any((full - roi) % 2):
        raise ValueError("ROI must fit and be centered on cell boundaries")
    start = (full - roi) // 2
    sl = (slice(int(start[0]), int(start[0] + roi[0])),
          slice(int(start[1]), int(start[1] + roi[1])))
    scratch_rng = np.random.default_rng(seed + 3_000_017)
    scratch, scratch_metadata = _make_individual_scratch_map(
        roi_size_m, resolution_m, scratch_rng, spec)
    state.micro_height_um[sl] -= scratch
    state.initial_micro_height_um[sl] -= scratch
    state.initial_scratch_depth_um[sl] = scratch
    state.residual_scratch_depth_um[sl] = scratch
    state.defect_mask[sl] = scratch > (0.5 * spec.light_scratch_depth_min_um)
    state.healthy_mask[:] = ~state.defect_mask
    mar_density, mar_severity, mar_metadata = _make_mar_maps(
        state.shape, resolution_m, seed + 5_000_041, spec)
    sample = SurfaceProfileSample(
        state=state, profile_id=NEW_CAR_MILD, base_micro_height_um=base_micro,
        mar_density=mar_density, mar_severity_initial=mar_severity,
        metadata={
            "profile_id": NEW_CAR_MILD, "design_status": spec.design_status,
            "profile_seed": seed, "scratch_seed": seed + 3_000_017,
            "mar_seed": seed + 5_000_041, "parameter_tags": dict(spec.tags),
            "scratch_region": "centered_roi", **metadata,
            **scratch_metadata, **mar_metadata,
        },
    )
    sample.validate()
    return sample


def make_surface_profile(profile_id: str, patch_size_m: tuple[float, float] = (0.20, 0.20),
                         resolution_m: float = 0.002, seed: int = 0,
                         **legacy_kwargs) -> SurfaceProfileSample:
    if profile_id == LEGACY_STRESS:
        state = make_flat_patch(
            patch_size_m=patch_size_m, resolution_m=resolution_m,
            seed=seed, **legacy_kwargs)
        sample = SurfaceProfileSample(
            state=state, profile_id=LEGACY_STRESS,
            base_micro_height_um=(state.initial_micro_height_um
                                  + state.initial_scratch_depth_um),
            mar_density=np.zeros(state.shape, dtype=float),
            mar_severity_initial=np.zeros(state.shape, dtype=float),
            metadata={
                "profile_id": LEGACY_STRESS,
                "design_status": "ESTABLISHED_GENERATOR_UNCHANGED",
                "profile_seed": seed,
                "generator": "learning.polytwin.surface_state.make_flat_patch",
                "scratch_depth_range_um": [C.SCRATCH_DEPTH_MIN_UM,
                                             C.SCRATCH_DEPTH_MAX_UM],
                "scratch_count_range": [C.SCRATCH_COUNT_MIN, C.SCRATCH_COUNT_MAX],
            },
        )
        sample.validate()
        return sample
    if profile_id == NEW_CAR_MILD:
        if legacy_kwargs:
            raise TypeError("legacy generator arguments are not valid for new_car_mild")
        return make_new_car_mild_patch(patch_size_m, resolution_m, seed)
    raise ValueError(f"unknown profile_id={profile_id!r}; expected one of {PROFILE_IDS}")
