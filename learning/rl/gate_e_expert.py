"""Isaac-free Gate E teacher rules for the pre-BC paired screen.

Only the pass-2 decision differs between teachers.  Pass 1 always comes from
the frozen 14-D champion.  All new thresholds are PT-DESIGN and must be screened
before any dataset collection or behavior cloning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from learning.rl.env.gate_d_observation import SPATIAL120, observation_dim


CHAMPION_THEN_FINISH = "champion_then_finish"
SPATIAL_GUARDED = "spatial_guarded"
TEACHER_MODES = (CHAMPION_THEN_FINISH, SPATIAL_GUARDED)


@dataclass(frozen=True)
class GateETeacherDesign:
    """PT-DESIGN thresholds, not measured production settings."""

    finish_force_action: float = -1.0
    finish_feed_mm_s: float = 8.0
    base_feed_mm_s: float = 12.7
    feed_ratio_limit: float = 0.5
    protect_force_action: float = -1.0
    protect_feed_action: float = 0.5
    clearcoat_margin_guard_um: float = 2.0
    scratch_guard_um: float = 0.30
    under_fraction_guard: float = 0.20
    over_fraction_guard: float = 0.40

    @property
    def finish_feed_action(self) -> float:
        return ((self.finish_feed_mm_s / self.base_feed_mm_s) - 1.0) / self.feed_ratio_limit

    def metadata(self) -> dict:
        values = asdict(self)
        values["finish_feed_action"] = self.finish_feed_action
        values["design_status"] = "PT-DESIGN_SYNTHETIC_TEACHER_SCREEN"
        return values


def teacher_actions(
    observation120: torch.Tensor,
    champion_actions: torch.Tensor,
    mode: str,
    design: GateETeacherDesign = GateETeacherDesign(),
) -> tuple[torch.Tensor, list[str]]:
    """Return actions and auditable rule labels for each environment."""
    if mode not in TEACHER_MODES:
        raise ValueError(f"teacher mode must be one of {TEACHER_MODES}")
    if observation120.ndim != 2 or observation120.shape[1] != observation_dim(SPATIAL120):
        raise ValueError(f"expected (N, 120) observation, got {tuple(observation120.shape)}")
    if champion_actions.shape != (observation120.shape[0], 2):
        raise ValueError(
            f"champion actions must have shape {(observation120.shape[0], 2)}")
    if not bool(torch.isfinite(observation120).all()):
        raise ValueError("observation contains NaN/Inf")
    if not bool(torch.isfinite(champion_actions).all()):
        raise ValueError("champion action contains NaN/Inf")

    actions = champion_actions.clamp(-1.0, 1.0).clone()
    pass_number_norm = observation120[:, 14]
    later = pass_number_norm > 1e-7
    finish = torch.tensor(
        [design.finish_force_action, design.finish_feed_action],
        device=actions.device,
        dtype=actions.dtype,
    )
    protect = torch.tensor(
        [design.protect_force_action, design.protect_feed_action],
        device=actions.device,
        dtype=actions.dtype,
    )
    labels = ["pass1_champion" for _ in range(len(actions))]

    if mode == CHAMPION_THEN_FINISH:
        actions[later] = finish
        for index in later.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
            labels[index] = "later_finish"
        return actions.clamp(-1.0, 1.0), labels

    clearcoat_margin_um = observation120[:, 18] * 20.0
    scratch_max_um = observation120[:, 20:45].amax(dim=1) * 2.0
    under_fraction = observation120[:, 70:95].amax(dim=1)
    over_fraction = observation120[:, 95:120].amax(dim=1)
    guard = later & (
        (clearcoat_margin_um <= design.clearcoat_margin_guard_um)
        | (over_fraction >= design.over_fraction_guard)
    )
    defect = later & ~guard & (
        (scratch_max_um >= design.scratch_guard_um)
        | (under_fraction >= design.under_fraction_guard)
    )
    neutral = later & ~guard & ~defect
    actions[guard] = protect
    actions[neutral] = finish
    for index in guard.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
        labels[index] = "later_protect"
    for index in defect.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
        labels[index] = "later_defect_champion"
    for index in neutral.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
        labels[index] = "later_finish"
    return actions.clamp(-1.0, 1.0), labels

