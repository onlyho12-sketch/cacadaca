"""CPU tests for F10-D PASS_LOCKED and selective revalidation invariants."""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning.polytwin.surface_state import make_flat_patch  # noqa: E402
from learning.rl import gate_f10_quality_state as qs  # noqa: E402


def _blank():
    state = make_flat_patch((0.02, 0.02), 0.002, seed=1,
                            target_ra_um=0.0, with_scratches=False)
    for values in (state.micro_height_um, state.initial_micro_height_um,
                   state.initial_scratch_depth_um, state.residual_scratch_depth_um,
                   state.cumulative_removal_um, state.thermal_damage_proxy):
        values[:] = 0.0
    state.defect_mask[:] = False
    state.clearcoat_remaining_um[:] = 40.0
    state.peak_temperature_c[:] = 25.0
    return state


def test_blank_cells_become_pass_locked_once():
    ledger = qs.inspect_after_coarse(_blank())
    assert np.all(ledger.status == qs.PASS_LOCKED)
    assert np.all(ledger.validation_count == 1)
    assert np.all(ledger.initially_pass_locked)


def test_initial_state_precedence():
    state = _blank()
    reached = np.ones(state.shape, bool)
    unsafe = np.zeros(state.shape, bool)
    reached[0, 0] = False
    unsafe[0, 1] = True
    state.clearcoat_remaining_um[0, 2] = 29.0
    ledger = qs.inspect_after_coarse(state, reached_mask=reached,
                                     unsafe_geometry_mask=unsafe)
    assert ledger.status[0, 0] == qs.NOT_REACHED
    assert ledger.status[0, 1] == qs.UNSAFE_GEOMETRY
    assert ledger.status[0, 2] == qs.CLEARCOAT_GUARD


def test_failed_all4_cell_is_rework():
    state = _blank()
    state.initial_scratch_depth_um[5, 5] = 1.0
    state.defect_mask[5, 5] = True
    state.micro_height_um[5, 5] = -1.0
    ledger = qs.inspect_after_coarse(state)
    assert np.any(ledger.status == qs.REWORK)


def test_touched_lock_becomes_rework_affected_then_selectively_validated():
    ledger = qs.inspect_after_coarse(_blank())
    changed = np.zeros(ledger.shape, bool)
    changed[3, 3] = True
    qs.mark_surface_changes(ledger, changed)
    assert ledger.status[3, 3] == qs.REWORK_AFFECTED
    assert ledger.rework_affected_ever[3, 3]
    new_quality = {key: value.copy() for key, value in ledger.quality.items()}
    selected = qs.apply_selective_revalidation(ledger, new_quality, changed)
    assert selected.sum() == 1
    assert ledger.status[3, 3] == qs.REWORK_PASS
    assert ledger.validation_count[3, 3] == 2


def test_unchanged_pass_locked_cells_are_never_revalidated():
    ledger = qs.inspect_after_coarse(_blank())
    changed = np.zeros(ledger.shape, bool)
    changed[2, 2] = True
    qs.mark_surface_changes(ledger, changed)
    new_quality = {key: value.copy() for key, value in ledger.quality.items()}
    qs.apply_selective_revalidation(ledger, new_quality, np.ones(ledger.shape, bool))
    unchanged = ~changed
    assert np.all(ledger.validation_count[unchanged] == 1)
    assert np.all(ledger.status[unchanged] == qs.PASS_LOCKED)


def test_changed_but_failed_cell_becomes_rework_fail():
    ledger = qs.inspect_after_coarse(_blank())
    changed = np.zeros(ledger.shape, bool)
    changed[4, 4] = True
    qs.mark_surface_changes(ledger, changed)
    quality = {key: value.copy() for key, value in ledger.quality.items()}
    quality["gu_pass"][4, 4] = False
    quality["all4_pass"][4, 4] = False
    qs.apply_selective_revalidation(ledger, quality, changed)
    assert ledger.status[4, 4] == qs.REWORK_FAIL
