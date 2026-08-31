"""Isaac-free tests for the isolated Gate E teacher screen."""
from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from learning.rl.gate_e_expert import (  # noqa: E402
    CHAMPION_THEN_FINISH,
    SPATIAL_GUARDED,
    GateETeacherDesign,
    teacher_actions,
)


def main() -> None:
    design = GateETeacherDesign()
    obs = torch.zeros((4, 120), dtype=torch.float32)
    champion = torch.tensor([
        [1.0, -1.0], [0.4, -0.2], [0.2, -0.1], [-0.3, 0.5],
    ])
    obs[1:, 14] = 0.5
    obs[1, 18] = 0.05  # 1 um clearcoat margin -> protect
    obs[2, 18] = 0.5
    obs[2, 20 + 7] = 0.4  # 0.8 um cached scratch -> champion
    obs[3, 18] = 0.5  # neutral -> finish

    baseline, baseline_labels = teacher_actions(obs, champion, CHAMPION_THEN_FINISH)
    assert baseline_labels == [
        "pass1_champion", "later_finish", "later_finish", "later_finish"]
    torch.testing.assert_close(baseline[0], champion[0])
    torch.testing.assert_close(
        baseline[1:, 0], torch.full((3,), design.finish_force_action))
    torch.testing.assert_close(
        baseline[1:, 1], torch.full((3,), design.finish_feed_action))

    spatial, labels = teacher_actions(obs, champion, SPATIAL_GUARDED)
    assert labels == [
        "pass1_champion", "later_protect", "later_defect_champion", "later_finish"]
    torch.testing.assert_close(spatial[0], champion[0])
    torch.testing.assert_close(
        spatial[1], torch.tensor([design.protect_force_action, design.protect_feed_action]))
    torch.testing.assert_close(spatial[2], champion[2])
    torch.testing.assert_close(
        spatial[3], torch.tensor([design.finish_force_action, design.finish_feed_action]))

    bad = obs.clone(); bad[0, 0] = torch.nan
    try:
        teacher_actions(bad, champion, SPATIAL_GUARDED)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN observation was accepted")
    print("Gate E expert: pass/finish/protect/defect/finite tests PASS")


if __name__ == "__main__":
    main()
