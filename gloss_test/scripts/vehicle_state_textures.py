"""Convert a validated 5x5 vehicle/RL state into RTX material textures."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_vehicle_state(npz_path, phase, safety_limit_um=35.0):
    """Load one before/after state and derive clearcoat optical integrity.

    Clearcoat thickness is not mapped linearly to gloss.  A sound clearcoat
    remains an optically present top layer while it is above the design safety
    threshold.  Only sub-threshold cells reduce the visualization-only
    clearcoat integrity map.
    """
    if phase not in {"before", "after"}:
        raise ValueError("phase must be 'before' or 'after'")
    if safety_limit_um <= 0.0:
        raise ValueError("safety_limit_um must be positive")

    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"vehicle state NPZ not found: {path}")
    required = {
        f"roughness_{phase}", f"scratch_{phase}",
        f"clearcoat_{phase}_um", "clearcoat_before_um",
    }
    with np.load(path) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"vehicle state NPZ missing arrays: {missing}")
        maps = {name: np.asarray(archive[name], dtype=np.float32) for name in required}

    shapes = {value.shape for value in maps.values()}
    if len(shapes) != 1:
        raise ValueError(f"vehicle state arrays have different shapes: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 2:
        raise ValueError(f"vehicle state grid must be square 2D, got {shape}")
    for name, value in maps.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"vehicle state array contains non-finite values: {name}")

    roughness = maps[f"roughness_{phase}"]
    scratch = maps[f"scratch_{phase}"]
    thickness = maps[f"clearcoat_{phase}_um"]
    if np.any((roughness < 0.0) | (roughness > 1.0)):
        raise ValueError("roughness state must be in [0, 1]")
    if np.any((scratch < 0.0) | (scratch > 1.0)):
        raise ValueError("scratch state must be in [0, 1]")
    if np.any(thickness < 0.0):
        raise ValueError("clearcoat thickness cannot be negative")

    integrity = np.clip(thickness / float(safety_limit_um), 0.0, 1.0)
    return {
        "grid_size": shape[0],
        "roughness": roughness,
        "scratch": scratch,
        "clearcoat_thickness_um": thickness,
        "clearcoat_before_um": maps["clearcoat_before_um"],
        "clearcoat_integrity": integrity.astype(np.float32),
        "clearcoat_safety_failure": thickness < float(safety_limit_um),
    }


def interpolate_grid(values, resolution=1024):
    """Smoothly interpolate cell-centred values into a UV texture field."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("values must be a square 2D grid")
    if resolution < values.shape[0]:
        raise ValueError("resolution must be at least the grid size")
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    uu, vv_image = np.meshgrid(axis, axis)
    vv = 1.0 - vv_image
    grid_axis = np.linspace(
        0.1, 0.9, values.shape[0], dtype=np.float32
    )
    numerator = np.zeros_like(uu)
    denominator = np.zeros_like(uu)
    sigma = 0.105 * 5.0 / values.shape[0]
    for row, center_v in enumerate(grid_axis):
        for column, center_u in enumerate(grid_axis):
            weight = np.exp(
                -((uu - center_u) ** 2 + (vv - center_v) ** 2)
                / (2.0 * sigma * sigma)
            )
            numerator += weight * values[row, column]
            denominator += weight
    return (numerator / np.maximum(denominator, 1.0e-8)).astype(np.float32)


def save_scalar_texture(path, values, resolution=1024):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.clip(interpolate_grid(values, resolution), 0.0, 1.0)
    Image.fromarray(np.round(field * 255.0).astype(np.uint8), "L").save(path)
    return path, field


def save_clearcoat_thickness_texture(path, thickness_um, resolution=1024):
    """Save a normalized diagnostic map while returning physical micrometres."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field_um = interpolate_grid(thickness_um, resolution)
    low = float(field_um.min())
    high = float(field_um.max())
    normalized = (
        np.zeros_like(field_um) if high <= low
        else (field_um - low) / (high - low)
    )
    Image.fromarray(np.round(normalized * 255.0).astype(np.uint8), "L").save(path)
    return path, field_um


def save_scratch_normal_texture(path, scratch_values, seed=20260827,
                                resolution=1024, strength=1.2):
    """Create deterministic scratches weighted by the supplied cell map."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch_field = np.clip(interpolate_grid(scratch_values, resolution), 0.0, 1.0)
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    uu, vv = np.meshgrid(axis, axis)
    rng = np.random.default_rng(seed + 71)
    height = np.zeros_like(uu)
    for _ in range(42):
        angle = rng.uniform(-np.pi, np.pi)
        offset = rng.uniform(-0.65, 0.65)
        curve = rng.uniform(-0.18, 0.18)
        centered_u = uu - 0.5
        centered_v = vv - 0.5
        transverse = -np.sin(angle) * centered_u + np.cos(angle) * centered_v
        longitudinal = np.cos(angle) * centered_u + np.sin(angle) * centered_v
        distance = transverse - offset - curve * longitudinal * longitudinal
        width = rng.uniform(0.0012, 0.0035)
        height -= rng.uniform(0.25, 0.75) * np.exp(-0.5 * (distance / width) ** 2)
    height *= scratch_field
    gradient_v, gradient_u = np.gradient(height)
    nx = -strength * gradient_u
    ny = strength * gradient_v
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    encoded = np.clip(
        np.stack((nx / length, ny / length, nz / length), axis=-1) * 0.5 + 0.5,
        0.0, 1.0,
    )
    Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), "RGB").save(path)
    return path, scratch_field
