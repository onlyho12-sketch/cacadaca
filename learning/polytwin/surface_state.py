"""SurfaceState 와 초기 표면 생성 — 02 문서 2·3장.

`nominal_surface_xyz_m` (거시형상, m) 와 `micro_height_um` (미세형상, μm) 를 분리한다.
Ra/Rz/Scratch 는 전부 micro_height_um 에서 계산한다. 단위를 섞지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config as C


@dataclass
class SurfaceState:
    """한 작업 patch 의 상태맵 묶음 (02 문서 2장)."""
    resolution_m: float
    nominal_surface_xyz_m: np.ndarray      # (N, M, 3)
    normal_xyz: np.ndarray                 # (N, M, 3)
    micro_height_um: np.ndarray            # (N, M)
    initial_micro_height_um: np.ndarray    # (N, M)  before/after 비교용
    initial_scratch_depth_um: np.ndarray
    residual_scratch_depth_um: np.ndarray
    cumulative_removal_um: np.ndarray
    clearcoat_remaining_um: np.ndarray
    initial_clearcoat_um: np.ndarray
    dwell_time_s: np.ndarray
    pass_count: np.ndarray                 # int
    peak_contact_pressure_proxy: np.ndarray
    heat_risk_proxy: np.ndarray
    temperature_c: np.ndarray                 # synthetic surface temperature [deg C]
    peak_temperature_c: np.ndarray            # maximum synthetic temperature [deg C]
    friction_heat_flux_w_m2: np.ndarray       # latest local friction heat flux [W/m2]
    temperature_removal_factor: np.ndarray    # dimensionless piecewise material factor
    thermal_damage_proxy: np.ndarray          # dimensionless cumulative thermal exposure
    healthy_mask: np.ndarray               # bool
    defect_mask: np.ndarray                # bool
    last_active_time_s: np.ndarray         # pass debounce 용 (내부 상태)
    seed: int

    @property
    def shape(self) -> tuple:
        return self.micro_height_um.shape

    @property
    def cell_area_m2(self) -> float:
        return self.resolution_m ** 2

    def copy(self) -> "SurfaceState":
        import copy as _copy
        return _copy.deepcopy(self)


def _correlated_field(shape, correlation_m: float, resolution_m: float,
                      rng: np.random.Generator) -> np.ndarray:
    """저주파 상관 난수장. 독립난수가 아니라 공간적으로 부드럽게 변한다 (02 문서 3.1)."""
    sigma_cells = max(correlation_m / resolution_m, 0.5)
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma_cells, mode="reflect")
    std = field.std()
    return field / std if std > 1e-12 else field


def _distance_to_segment(xx, yy, p0, p1) -> np.ndarray:
    """격자 각 점에서 선분 p0→p1 까지의 최단거리 [m]."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length_sq = float(d @ d)
    if length_sq < 1e-12:
        return np.hypot(xx - p0[0], yy - p0[1])
    t = ((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / length_sq
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(xx - (p0[0] + t * d[0]), yy - (p0[1] + t * d[1]))


def make_flat_patch(patch_size_m=(0.20, 0.20),
                    resolution_m: float = 0.001,
                    seed: int = 0,
                    n_scratches: int | None = None,
                    target_ra_um: float = C.RA_TARGET_UM,
                    with_scratches: bool = True) -> SurfaceState:
    """평면 patch 초기 표면 생성 (02 문서 3장).

    Ra 는 `mean(|z - mean z|)` 정의로 target_ra_um 에 맞춘다.
    ⚠ 한계: 격자 간격이 mm 급이므로 여기서의 Ra 는 실제 조도계(μm 급 샘플링)의 Ra 와
      같은 물리량이 아니다. 모델 내부 일관성 지표로만 쓴다 (SYNTHETIC).
    """
    rng = np.random.default_rng(seed)
    nx = int(round(patch_size_m[0] / resolution_m))
    ny = int(round(patch_size_m[1] / resolution_m))
    shape = (nx, ny)

    x = (np.arange(nx) + 0.5) * resolution_m
    y = (np.arange(ny) + 0.5) * resolution_m
    xx, yy = np.meshgrid(x, y, indexing="ij")

    nominal = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    normal = np.zeros((nx, ny, 3))
    normal[..., 2] = 1.0

    # ── 정상 roughness: band-limited noise 를 target Ra 로 스케일 (3.2) ──
    micro = _correlated_field(shape, C.RA_NOISE_CORRELATION_M, resolution_m, rng)
    micro -= micro.mean()
    mad = np.abs(micro).mean()
    micro *= target_ra_um / max(mad, 1e-12)

    # ── Clearcoat: 저주파 상관장 위에 40~50 μm (3.1) ──
    cc_field = _correlated_field(shape, C.CLEARCOAT_FIELD_CORRELATION_M, resolution_m, rng)
    cc_unit = 0.5 * (1.0 + np.clip(cc_field / 3.0, -1.0, 1.0))      # 0~1 로 압축
    clearcoat = C.CLEARCOAT_MIN_UM + cc_unit * (C.CLEARCOAT_MAX_UM - C.CLEARCOAT_MIN_UM)

    # ── Scratch: 직선 groove (3.3) ──
    scratch_depth = np.zeros(shape)
    if with_scratches:
        if n_scratches is None:
            n_scratches = int(rng.integers(C.SCRATCH_COUNT_MIN, C.SCRATCH_COUNT_MAX + 1))
        for _ in range(n_scratches):
            depth_um = float(rng.uniform(C.SCRATCH_DEPTH_MIN_UM, C.SCRATCH_DEPTH_MAX_UM))
            length_m = float(rng.uniform(C.SCRATCH_LENGTH_MIN_M, C.SCRATCH_LENGTH_MAX_M))
            angle = float(rng.uniform(0.0, np.pi))
            cx = float(rng.uniform(0.0, patch_size_m[0]))
            cy = float(rng.uniform(0.0, patch_size_m[1]))
            half = 0.5 * length_m * np.array([np.cos(angle), np.sin(angle)])
            dist = _distance_to_segment(xx, yy, (cx - half[0], cy - half[1]),
                                        (cx + half[0], cy + half[1]))
            groove = depth_um * np.exp(-(dist / C.SCRATCH_WIDTH_M) ** 2)
            scratch_depth = np.maximum(scratch_depth, groove)

    micro = micro - scratch_depth       # groove 는 표면을 파고든다

    defect = scratch_depth > (0.5 * C.SCRATCH_DEPTH_MIN_UM)

    return SurfaceState(
        resolution_m=resolution_m,
        nominal_surface_xyz_m=nominal,
        normal_xyz=normal,
        micro_height_um=micro.copy(),
        initial_micro_height_um=micro.copy(),
        initial_scratch_depth_um=scratch_depth,
        residual_scratch_depth_um=scratch_depth.copy(),
        cumulative_removal_um=np.zeros(shape),
        clearcoat_remaining_um=clearcoat.copy(),
        initial_clearcoat_um=clearcoat.copy(),
        dwell_time_s=np.zeros(shape),
        pass_count=np.zeros(shape, dtype=np.int32),
        peak_contact_pressure_proxy=np.zeros(shape),
        heat_risk_proxy=np.zeros(shape),
        temperature_c=np.full(shape, C.INITIAL_TEMPERATURE_C),
        peak_temperature_c=np.full(shape, C.INITIAL_TEMPERATURE_C),
        friction_heat_flux_w_m2=np.zeros(shape),
        temperature_removal_factor=np.ones(shape),
        thermal_damage_proxy=np.zeros(shape),
        healthy_mask=~defect,
        defect_mask=defect,
        last_active_time_s=np.full(shape, -1e9),
        seed=seed,
    )
