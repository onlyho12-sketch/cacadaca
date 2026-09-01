"""CPU tests for the Gate F10-B local vehicle geometry adapter."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.rl import gate_f10_vehicle_geometry as vg  # noqa: E402


def _patch(kx=0.0, ky=0.0, extent=0.06, n=25):
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(x, y)
    zz = 0.5 * kx * xx * xx + 0.5 * ky * yy * yy
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def test_plane_normal_curvature_and_confidence():
    cfg = vg.GeometryConfig()
    fit = vg.fit_local_surface(np.zeros(3), _patch(), np.array([0.0, 0.0, -1.0]), cfg)
    assert np.array([fit["normal_x"], fit["normal_y"], fit["normal_z"]]) == pytest.approx([0, 0, 1])
    assert fit["k1_1_m"] == pytest.approx(0.0, abs=1e-8)
    assert fit["k2_1_m"] == pytest.approx(0.0, abs=1e-8)
    assert fit["surface_class"] == "near_flat"
    assert fit["fit_confidence"] > 0.9


def test_signed_paraboloid_curvatures_at_origin():
    cfg = vg.GeometryConfig()
    fit = vg.fit_local_surface(np.zeros(3), _patch(kx=-2.0, ky=-5.0),
                               np.array([0.0, 0.0, -1.0]), cfg)
    assert fit["k1_1_m"] == pytest.approx(-2.0, rel=0.03)
    assert fit["k2_1_m"] == pytest.approx(-5.0, rel=0.03)
    assert fit["surface_class"] == "convex"


def test_saddle_classification():
    cfg = vg.GeometryConfig()
    fit = vg.fit_local_surface(np.zeros(3), _patch(kx=3.0, ky=-2.0),
                               np.array([0.0, 0.0, -1.0]), cfg)
    assert fit["surface_class"] == "saddle"
    assert fit["k1_1_m"] * fit["k2_1_m"] < 0.0


def test_robust_fit_resists_sparse_height_outliers():
    cfg = vg.GeometryConfig()
    points = _patch(kx=-2.0, ky=-4.0)
    points[::40, 2] += 0.04
    fit = vg.fit_local_surface(np.zeros(3), points, np.array([0.0, 0.0, -1.0]), cfg)
    assert fit["k1_1_m"] == pytest.approx(-2.0, abs=0.35)
    assert fit["k2_1_m"] == pytest.approx(-4.0, abs=0.35)


def test_boundary_evidence_distinguishes_edge():
    full = _patch()[:, :2]
    half = full[full[:, 0] >= 0.0]
    center_distance, center_coverage, _ = vg._boundary_evidence(full, 0.055)
    edge_distance, edge_coverage, edge_hole = vg._boundary_evidence(half, 0.055)
    assert center_distance > edge_distance
    assert center_coverage > edge_coverage
    assert edge_hole > 0.0


def test_rejects_too_few_neighbors():
    with pytest.raises(ValueError, match="at least"):
        vg.fit_local_surface(np.zeros(3), np.zeros((4, 3)), np.zeros(3), vg.GeometryConfig())
