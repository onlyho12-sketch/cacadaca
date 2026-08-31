"""Pure Gate C1 geometry/coverage regression checks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from learning.rl.gate_c_path_coverage_screen import (  # noqa: E402
    DIRECTIONS,
    EDGE_MODES,
    STEP_OVER_RATIOS,
    integrate_exposure,
)


def main() -> None:
    results = []
    for ratio in STEP_OVER_RATIOS:
        for edge in EDGE_MODES:
            for direction in DIRECTIONS:
                result = integrate_exposure(ratio, edge, direction)
                scalars = result["scalars"]
                results.append(scalars)
                assert result["normalized_exposure"].shape == (100, 100)
                assert result["tile_exposure_norm"].shape == (5, 5)
                assert np.isfinite(result["normalized_exposure"]).all()
                assert abs(result["normalized_exposure"].mean() - 1.0) < 1e-12
                assert scalars["footprint_inside_quality_map"]
                assert scalars["coverage_any_fraction"] == 1.0
                assert 0.0 <= scalars["coverage_effective_fraction"] <= 1.0
    assert len(results) == 24
    # Cross-direction exposure must be transpose-symmetric for balanced square ROI paths.
    cross = integrate_exposure(0.25, "balanced", "cross_xy")["normalized_exposure"]
    assert np.allclose(cross, cross.T, atol=1e-12)
    # Midpoint quadrature leaves no more than half a sample step beyond the
    # exact zero-margin endpoint while keeping every sampled footprint inside.
    extended = integrate_exposure(0.25, "balanced_extend5", "cross_xy")["scalars"]
    assert 0.0 <= extended["minimum_quality_map_footprint_margin_m"] <= 0.00101
    print("Gate C1 analytic coverage: 24 candidates + 8 geometry checks PASS")


if __name__ == "__main__":
    main()
