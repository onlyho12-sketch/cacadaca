"""Large deterministic CPU smoke for the F11 selective pipeline contract."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.polytwin.surface_state import make_flat_patch
from learning.rl.gate_f11_selective_pipeline import (
    CellGeometry, begin_pipeline, finish_pipeline, snapshot_before_rework, write_evidence)


def make_state():
    state = make_flat_patch((0.32, 0.32), 0.002, seed=1100,
                            target_ra_um=0.0, with_scratches=False)
    for values in (state.micro_height_um, state.initial_micro_height_um,
                   state.initial_scratch_depth_um, state.residual_scratch_depth_um,
                   state.cumulative_removal_um, state.thermal_damage_proxy):
        values[:] = 0.0
    state.defect_mask[:] = False
    state.healthy_mask[:] = True
    state.clearcoat_remaining_um[:] = 40.0
    state.initial_clearcoat_um[:] = 40.0
    state.temperature_c[:] = 25.0
    state.peak_temperature_c[:] = 25.0
    checker = np.indices((80, 80)).sum(axis=0) % 2
    state.micro_height_um[40:120, 40:120] = checker * 2.0 - 1.0
    state.initial_scratch_depth_um[40:120, 40:120] = 1.0
    state.defect_mask[40:120, 40:120] = True
    state.healthy_mask[:] = ~state.defect_mask
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    state = make_state()
    geometry = CellGeometry.from_surface_state(state)
    geometry.surface_class[:8, :8] = "concave"
    geometry.k1_1_m[:8, :8] = 3.0
    geometry.k2_1_m[:8, :8] = 2.0
    geometry.curvature_radius_m[:8, :8] = 1.0 / 3.0
    geometry.geometry_risk[:8, :8] = 1.0
    pipe = begin_pipeline(state, geometry=geometry, surface_id="synthetic_vehicle_patch")
    snapshot_before_rework(pipe, state)
    after = state.copy()
    actual = pipe.planned_footprint_mask.copy()
    target = pipe.ledger.status == "REWORK"
    after.micro_height_um[target & actual] = 0.0
    after.cumulative_removal_um[actual] += 0.02
    after.clearcoat_remaining_um[actual] -= 0.02
    after.dwell_time_s[actual] += 1.0
    after.pass_count[actual] += 1
    summary = finish_pipeline(pipe, after)
    checks = {
        "cell_count": int(np.prod(state.shape)) == 25600,
        "automatic_rework_allowed": pipe.planner_summary["automatic_rework_allowed"],
        "one_or_more_regions": pipe.planner_summary["regions"] > 0,
        "revalidation_exactly_actual_changed": summary[
            "revalidation_exactly_actual_changed"],
        "unchanged_pass_locked_revalidated_zero": summary[
            "unchanged_pass_locked_revalidated"] == 0,
        "touched_locked_cells_recorded": summary["rework_affected_cells"] > 0,
        "concave_cells_excluded": int(pipe.concave_excluded_mask.sum()) == 64,
    }
    summary["cpu_smoke_checks"] = checks
    summary["cpu_smoke_pass"] = all(checks.values())
    write_evidence(args.out_dir, pipe, summary)
    with open(os.path.join(args.out_dir, "cpu_smoke.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["cpu_smoke_pass"]:
        raise RuntimeError("F11 CPU smoke failed")


if __name__ == "__main__":
    main()
