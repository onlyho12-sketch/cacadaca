"""Deterministic whole-panel roughness and scratch maps for polishing tests."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def create_severity_grid(grid_size=5, seed=20260827, pristine_cell=(3, 3)):
    """Return varied defect severity with exactly one intentionally pristine cell."""
    rng = np.random.default_rng(seed)
    severity = rng.uniform(0.25, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    row, column = pristine_cell
    severity[row - 1, column - 1] = 0.0
    return severity


def save_severity_grid(output_dir, severity, seed, pristine_cell):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "severity_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["grid_row", "grid_column", "initial_severity"]
        )
        writer.writeheader()
        for row in range(severity.shape[0]):
            for column in range(severity.shape[1]):
                writer.writerow({
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "initial_severity": float(severity[row, column]),
                })
    payload = {
        "seed": int(seed),
        "grid": list(severity.shape),
        "pristine_cell": list(pristine_cell),
        "meaning": "0=pristine, 1=maximum simulated surface defect",
        "is_gu": False,
        "initial_severity": severity.tolist(),
    }
    (output_dir / "severity_grid.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return csv_path


def create_continuous_severity_map(
    path,
    severity,
    residual_factor=1.0,
    resolution=1024,
    pristine_cell=(3, 3),
):
    """Interpolate the 5x5 severities into a smooth, full-panel texture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    uu, vv_image = np.meshgrid(axis, axis)
    vv = 1.0 - vv_image
    grid_axis = np.linspace(0.1, 0.9, severity.shape[0], dtype=np.float32)
    numerator = np.zeros_like(uu)
    denominator = np.zeros_like(uu)
    sigma = 0.105
    for row, center_v in enumerate(grid_axis):
        for column, center_u in enumerate(grid_axis):
            weight = np.exp(
                -((uu - center_u) ** 2 + (vv - center_v) ** 2) / (2.0 * sigma * sigma)
            )
            numerator += weight * severity[row, column]
            denominator += weight
    field = numerator / np.maximum(denominator, 1.0e-6)

    # Preserve one naturally glossy island, with a soft transition to its rough neighbours.
    pristine_row, pristine_column = pristine_cell
    pristine_u = grid_axis[pristine_column - 1]
    pristine_v = grid_axis[pristine_row - 1]
    distance_squared = (uu - pristine_u) ** 2 + (vv - pristine_v) ** 2
    glossy_island = 1.0 - np.exp(-distance_squared / (2.0 * 0.055 * 0.055))
    field = np.clip(field * glossy_island * residual_factor, 0.0, 1.0).astype(np.float32)
    Image.fromarray(np.round(field * 255.0).astype(np.uint8), "L").save(path)
    return path, field


def create_distributed_scratch_normal_map(
    path, severity_field, seed=20260827, strength=1.2
):
    """Create scratches across the panel, weighted by local roughness severity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = severity_field.shape[0]
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    uu, vv = np.meshgrid(axis, axis)
    rng = np.random.default_rng(seed + 17)
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
    height *= severity_field
    gradient_v, gradient_u = np.gradient(height)
    nx = -strength * gradient_u
    ny = strength * gradient_v
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack((nx / length, ny / length, nz / length), axis=-1)
    encoded = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), "RGB").save(path)
    return path
