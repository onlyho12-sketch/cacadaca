"""F10-G3 isolated 120 Hz supervisor adapter over the frozen v2 evaluator."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

# The frozen v2 parser owns the common CLI.  Accept a v3 label externally,
# translate it for import, then restore the truthful label before evaluation.
sys.argv = ["curvature_safety_v2" if item == "curvature_safety_v3" else item
            for item in sys.argv]

import learning.rl.gate_f10_curvature_physx_v2_eval as base
import numpy as np
import torch
from learning.rl.gate_f10_substep_supervisor import (
    SubstepSafetyState,
    supervise_substep,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class GateF8EvalEnvV3(base.GateF8EvalEnv):
    last_instance = None

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._f10g3_state = None
        self._f10g3_nominal_force_cmd = None
        self._f10g3_intervention_substeps = None
        self._f10g3_fault_substeps = None
        self._f10g3_max_predicted_force_n = None
        self._f10g3_raw_force_max_by_env_n = None
        self._f10g3_substeps = None
        self._f10g3_total_interventions = 0
        self._f10g3_total_faults = 0
        self._f10g3_force_overload_sequences = 0
        self._f10g3_raw_force_max_n = 0.0
        super().__init__(cfg, render_mode, **kwargs)
        count = self.num_envs
        self._f10g3_state = SubstepSafetyState.zeros(count)
        self._f10g3_intervention_substeps = np.zeros(count, dtype=np.int64)
        self._f10g3_fault_substeps = np.zeros(count, dtype=np.int64)
        self._f10g3_max_predicted_force_n = np.zeros(count, dtype=np.float64)
        self._f10g3_raw_force_max_by_env_n = np.zeros(count, dtype=np.float64)
        self._f10g3_substeps = np.zeros(count, dtype=np.int64)
        GateF8EvalEnvV3.last_instance = self

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._f10g3_nominal_force_cmd = self._force_cmd.clone()

    def _apply_action(self):
        if self._f10g3_state is None or self._f10g3_nominal_force_cmd is None:
            return super()._apply_action()
        nominal = self._f10g3_nominal_force_cmd
        result = supervise_substep(
            nominal.detach().cpu().numpy(),
            self._force_sensor_n.detach().cpu().numpy(),
            self._force_sensor_filt_n.detach().cpu().numpy(),
            self._pad_gap_m.detach().cpu().numpy(),
            surface_kind=str(self.cfg.surface_kind), state=self._f10g3_state)
        intervention = np.asarray(result["intervention"], dtype=bool)
        fault = np.asarray(result["fault"], dtype=bool)
        predicted = np.asarray(result["predicted_force_n"], dtype=np.float64)
        raw = self._force_sensor_n.detach().cpu().numpy()
        self._f10g3_intervention_substeps += intervention.astype(np.int64)
        self._f10g3_fault_substeps += fault.astype(np.int64)
        self._f10g3_substeps += 1
        self._f10g3_raw_force_max_by_env_n = np.maximum(
            self._f10g3_raw_force_max_by_env_n, raw)
        self._f10g3_total_interventions += int(intervention.sum())
        self._f10g3_total_faults += int(fault.sum())
        self._f10g3_raw_force_max_n = max(
            self._f10g3_raw_force_max_n, float(np.max(raw)))
        self._f10g3_max_predicted_force_n = np.maximum(
            self._f10g3_max_predicted_force_n, predicted)
        self._force_cmd = torch.as_tensor(
            result["force_command_n"], device=self.device, dtype=nominal.dtype)
        retract = torch.as_tensor(
            result["minimum_retract_velocity_m_s"],
            device=self.device, dtype=self.contact.z_vel.dtype)
        active = torch.as_tensor(intervention, device=self.device)
        self.contact.z_vel = torch.where(
            active, torch.maximum(self.contact.z_vel, retract), self.contact.z_vel)
        super()._apply_action()

    def _capture_final(self, env_id):
        super()._capture_final(env_id)
        if self._f10g3_intervention_substeps is not None:
            row = self.gate_f8_latched_safety[env_id]
            self._f10g3_force_overload_sequences += int(
                row["force_hard_violated"])
            row.update({
                "substep_supervisor_interventions": int(
                    self._f10g3_intervention_substeps[env_id]),
                "substep_supervisor_faults": int(self._f10g3_fault_substeps[env_id]),
                "substep_supervisor_max_predicted_force_n": float(
                    self._f10g3_max_predicted_force_n[env_id]),
                "substep_supervisor_raw_force_max_n": float(
                    self._f10g3_raw_force_max_by_env_n[env_id]),
                "substep_supervisor_fraction": float(
                    self._f10g3_intervention_substeps[env_id]
                    / max(1, self._f10g3_substeps[env_id])),
            })

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._f10g3_state is None:
            return
        indices = env_ids.detach().cpu().numpy().astype(int)
        self._f10g3_state.previous_raw_force_n[indices] = 0.0
        self._f10g3_state.retract_latched[indices] = False
        self._f10g3_state.safe_streak[indices] = 0
        self._f10g3_intervention_substeps[indices] = 0
        self._f10g3_fault_substeps[indices] = 0
        self._f10g3_max_predicted_force_n[indices] = 0.0
        self._f10g3_raw_force_max_by_env_n[indices] = 0.0
        self._f10g3_substeps[indices] = 0


def correct_outputs(out_dir):
    path = os.path.join(out_dir, "metadata.json")
    common = {
        "gate": "F10G3_SUBSTEP_SUPERVISOR_PHYSX",
        "shield_mode": "curvature_safety_v3",
        "safety_interface": "F10_CURVATURE_SAFETY_V2_PLUS_SUBSTEP_SUPERVISOR_V1",
        "substep_supervisor_source": os.path.abspath(
            "learning/rl/gate_f10_substep_supervisor.py"),
        "substep_supervisor_sha256": sha256(
            "learning/rl/gate_f10_substep_supervisor.py"),
    }
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            metadata = json.load(fh)
        metadata.update(common)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
    smoke_path = os.path.join(out_dir, "smoke.json")
    if os.path.isfile(smoke_path):
        with open(smoke_path, encoding="utf-8") as fh:
            smoke = json.load(fh)
        env = GateF8EvalEnvV3.last_instance
        smoke.update(common)
        smoke.update({
            "force_overload_sequences": env._f10g3_force_overload_sequences,
            "substep_supervisor_total_interventions": env._f10g3_total_interventions,
            "substep_supervisor_total_faults": env._f10g3_total_faults,
            "raw_force_max_n": env._f10g3_raw_force_max_n,
        })
        smoke["smoke_pass"] = bool(
            smoke["smoke_pass"] and smoke["force_overload_sequences"] == 0
            and smoke["substep_supervisor_total_faults"] == 0)
        with open(smoke_path, "w", encoding="utf-8") as fh:
            json.dump(smoke, fh, indent=2, sort_keys=True)


def main():
    base.args.shield_mode = "curvature_safety_v3"
    base.GateF8EvalEnv = GateF8EvalEnvV3
    base.main()
    correct_outputs(os.path.abspath(base.args.out_dir))


if __name__ == "__main__":
    try:
        main()
    finally:
        base.app.close()
