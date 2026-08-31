"""Gate F4 one-environment curved PhysX contact diagnostic (no training)."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import sys
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-id", required=True)
parser.add_argument("--surface-kind", choices=("flat", "cylinder", "sphere", "freeform"),
                    required=True)
parser.add_argument("--radius-m", type=float, default=0.60)
parser.add_argument("--freeform-seed", type=int, default=0)
parser.add_argument("--direction-mode", choices=("same_xx", "cross_xy"), default="cross_xy")
parser.add_argument("--pad-alignment", choices=("vertical", "normal"), default="vertical")
parser.add_argument("--gate-label", default="F4")
parser.add_argument("--surface-seed", type=int, default=31000)
parser.add_argument("--physics-seed", type=int, default=20260901)
parser.add_argument("--max-control-steps", type=int, default=5000)
parser.add_argument("--progress-interval", type=int, default=1000)
parser.add_argument("--partial-trace-interval", type=int, default=0)
parser.add_argument("--out-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from learning.polytwin.factory_surface_profiles import FACTORY_PREPOLISH  # noqa: E402
from learning.rl.env.gate_f_curved_polish_env import GateFCurvedPolishEnv  # noqa: E402
from learning.rl.env.gate_f_curved_polish_env_cfg import GateFCurvedPolishEnvCfg  # noqa: E402


TILE_KEYS = (
    "removal_mean_um",
    "removal_std_um",
    "total_ra_um",
    "total_rz_um",
    "fine_ra_um",
    "fine_rz_um",
    "waviness_std_um",
    "scratch_mean_um",
    "scratch_max_um",
    "clearcoat_min_um",
    "under_fraction",
    "over_fraction",
    "coverage_fraction",
)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tile_rows(candidate_id: str, stage: str, maps: dict) -> list[dict]:
    arrays = {key: np.asarray(maps[key], dtype=np.float64) for key in TILE_KEYS}
    if any(values.shape != (5, 5) for values in arrays.values()):
        raise ValueError("established diagnostic tile maps must all be 5x5")
    rows = []
    for tile_x, tile_y in np.ndindex((5, 5)):
        rows.append({
            "candidate_id": candidate_id,
            "stage": stage,
            "tile_x": tile_x,
            "tile_y": tile_y,
            **{key: float(values[tile_x, tile_y]) for key, values in arrays.items()},
        })
    return rows


class GateF4DiagnosticEnv(GateFCurvedPolishEnv):
    """Capture final maps and latched safety immediately before framework reset."""

    def __init__(self, *env_args, **env_kwargs):
        self.gate_f4_final_diagnostic: dict[int, dict] = {}
        self.gate_f4_latched_safety: dict[int, dict] = {}
        super().__init__(*env_args, **env_kwargs)

    def _repolish_decide(self, env_id: int) -> bool:
        force_hard = bool(self._force_hard_violated[env_id])
        thermal_hard = bool(self._thermal_hard_violated[env_id])
        unstable_hard = bool(self._unstable_hard_violated[env_id])
        terminate = super()._repolish_decide(env_id)
        if terminate:
            self.gate_f4_final_diagnostic[env_id] = self.factory_diagnostic(env_id)
            self.gate_f4_latched_safety[env_id] = {
                "force_hard_violated": force_hard,
                "thermal_hard_violated": thermal_hard,
                "unstable_hard_violated": unstable_hard,
            }
        return terminate


def _finite_trace(trace_rows: list[dict]) -> bool:
    for row in trace_rows:
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not np.isfinite(value):
                    return False
    return True


def _trace_summary(trace_rows: list[dict], sensor_valid_min_n: float) -> dict:
    raw = np.asarray([row["force_sensor_raw_n"] for row in trace_rows], dtype=np.float64)
    filtered = np.asarray(
        [row["force_sensor_filtered_n"] for row in trace_rows], dtype=np.float64)
    command = np.asarray([row["force_cmd_n"] for row in trace_rows], dtype=np.float64)
    matrix = np.asarray([row["sensor_matrix_n"] for row in trace_rows], dtype=np.float64)
    gap = np.asarray([row["pad_gap_m"] for row in trace_rows], dtype=np.float64)
    clearance = np.asarray([row["command_clearance_m"] for row in trace_rows], dtype=np.float64)
    contact = filtered > float(sensor_valid_min_n)
    first_contact = int(np.flatnonzero(contact)[0]) if np.any(contact) else -1
    steady_start = min(len(trace_rows), first_contact + 40) if first_contact >= 0 else len(trace_rows)
    steady = np.arange(len(trace_rows)) >= steady_start
    steady_contact = steady & contact
    force_sample = filtered[steady_contact]
    error_sample = np.abs(filtered[steady_contact] - command[steady_contact])
    gap_error = np.abs(gap[steady] - clearance[steady])
    alignment_error = np.asarray(
        [row["normal_alignment_error_deg"] for row in trace_rows], dtype=np.float64)
    target_quat = np.asarray([
        [row[f"target_quat_{component}"] for component in "xyzw"]
        for row in trace_rows
    ], dtype=np.float64)
    target_quat_norm_error = np.abs(np.linalg.norm(target_quat, axis=-1) - 1.0)
    target_dot = np.sum(target_quat[1:] * target_quat[:-1], axis=-1)
    target_step_angle = np.degrees(
        2.0 * np.arccos(np.clip(np.abs(target_dot), 0.0, 1.0)))
    target_tilt = np.degrees(
        2.0 * np.arccos(np.clip(np.abs(target_quat[:, 0]), 0.0, 1.0)))
    return {
        "trace_row_count": len(trace_rows),
        "trace_all_finite": _finite_trace(trace_rows),
        "first_contact_control_step": first_contact,
        "steady_start_control_step": int(steady_start),
        "contact_control_steps": int(contact.sum()),
        "post_first_contact_rate": (
            float(contact[first_contact:].mean()) if first_contact >= 0 else 0.0
        ),
        "steady_contact_control_steps": int(steady_contact.sum()),
        "force_sensor_raw_max_n": float(raw.max()),
        "force_sensor_filtered_max_n": float(filtered.max()),
        "force_sensor_filtered_steady_mean_n": (
            float(force_sample.mean()) if force_sample.size else 0.0
        ),
        "force_sensor_filtered_steady_std_n": (
            float(force_sample.std()) if force_sample.size else 0.0
        ),
        "force_tracking_abs_error_steady_mean_n": (
            float(error_sample.mean()) if error_sample.size else None
        ),
        "force_tracking_abs_error_steady_p95_n": (
            float(np.percentile(error_sample, 95)) if error_sample.size else None
        ),
        "sensor_matrix_net_abs_delta_mean_n": float(np.mean(np.abs(matrix - raw))),
        "gap_tracking_abs_error_steady_mean_m": (
            float(gap_error.mean()) if gap_error.size else None
        ),
        "gap_tracking_abs_error_steady_p95_m": (
            float(np.percentile(gap_error, 95)) if gap_error.size else None
        ),
        "normal_alignment_error_steady_mean_deg": float(alignment_error[steady].mean()),
        "normal_alignment_error_steady_p95_deg": float(
            np.percentile(alignment_error[steady], 95)),
        "normal_alignment_error_max_deg": float(alignment_error.max()),
        "target_quaternion_norm_max_error": float(target_quat_norm_error.max()),
        "target_quaternion_step_angle_max_deg": (
            float(target_step_angle.max()) if target_step_angle.size else 0.0),
        "target_tilt_max_deg": float(target_tilt.max()),
        "sensor_fault_control_steps": int(sum(row["sensor_fault"] for row in trace_rows)),
        "raw_hard_force_control_steps": int(sum(
            row["force_sensor_raw_n"] > row["force_hard_limit_n"] for row in trace_rows
        )),
    }


def main() -> None:
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    cfg = GateFCurvedPolishEnvCfg()
    cfg.scene.num_envs = 1
    cfg.factory_profile_ids = (FACTORY_PREPOLISH,)
    cfg.factory_envs_per_profile = 1
    cfg.factory_path_mode = "gate_c"
    cfg.factory_gate_c_step_over_ratio = 0.40
    cfg.factory_gate_c_edge_mode = "balanced_extend5"
    cfg.factory_gate_c_direction_mode = args.direction_mode
    cfg.surface_kind = args.surface_kind
    cfg.curvature_radius_m = float(args.radius_m)
    cfg.freeform_seed = int(args.freeform_seed)
    cfg.align_pad_to_surface_normal = args.pad_alignment == "normal"
    cfg.surface_seed_base = int(args.surface_seed)
    cfg.seed = int(args.physics_seed)
    cfg.robot_feed_speed_mm_s = 12.7
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = 1
    cfg.episode_length_s = 840.0

    env = GateF4DiagnosticEnv(cfg, render_mode=None)
    env._repolish_mode = True
    try:
        env.reset()
        initial = env.factory_diagnostic(0)
        initial_tiles = _tile_rows(args.candidate_id, "initial", initial["tile_maps"])
        action = torch.zeros((1, 2), device=env.device)
        trace_rows: list[dict] = []
        log = None
        steps = 0
        while log is None and steps < int(args.max_control_steps):
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            trace_rows.append({
                "candidate_id": args.candidate_id,
                "control_step": steps,
                "sim_time_s": float(env._sim_time[0]),
                "force_cmd_n": float(env._force_cmd[0]),
                "force_sensor_raw_n": float(env._force_sensor_n[0]),
                "force_sensor_filtered_n": float(env._force_sensor_filt_n[0]),
                "force_model_n": float(env._force_model_n[0]),
                "force_used_n": float(env._force_used_n[0]),
                "sensor_matrix_n": float(env._sensor_matrix_n[0]),
                "sensor_fault": bool(env._sensor_fault[0]),
                "force_hard_limit_n": float(env.cfg.force_hard_limit_n),
                "pad_u_m": float(env._pad_uv_actual[0, 0]),
                "pad_v_m": float(env._pad_uv_actual[0, 1]),
                "pad_gap_m": float(env._pad_gap_m[0]),
                "command_clearance_m": float(env.contact.command_clearance[0]),
                "surface_normal_x": float(env._pad_surface_normal_w[0, 0]),
                "surface_normal_y": float(env._pad_surface_normal_w[0, 1]),
                "surface_normal_z": float(env._pad_surface_normal_w[0, 2]),
                "link6_axis_x": float(env._link6_axis_w[0, 0]),
                "link6_axis_y": float(env._link6_axis_w[0, 1]),
                "link6_axis_z": float(env._link6_axis_w[0, 2]),
                "pad_outward_axis_x": float(env._pad_outward_axis_w[0, 0]),
                "pad_outward_axis_y": float(env._pad_outward_axis_w[0, 1]),
                "pad_outward_axis_z": float(env._pad_outward_axis_w[0, 2]),
                "normal_alignment_error_deg": float(
                    env._normal_alignment_error_deg[0]),
                "target_quat_x": float(env._target_link6_quat_w[0, 0]),
                "target_quat_y": float(env._target_link6_quat_w[0, 1]),
                "target_quat_z": float(env._target_link6_quat_w[0, 2]),
                "target_quat_w": float(env._target_link6_quat_w[0, 3]),
                "actual_link6_quat_x": float(env._link6_quat_w[0, 0]),
                "actual_link6_quat_y": float(env._link6_quat_w[0, 1]),
                "actual_link6_quat_z": float(env._link6_quat_w[0, 2]),
                "actual_link6_quat_w": float(env._link6_quat_w[0, 3]),
                "actual_pad_quat_x": float(env._pad_quat_w[0, 0]),
                "actual_pad_quat_y": float(env._pad_quat_w[0, 1]),
                "actual_pad_quat_z": float(env._pad_quat_w[0, 2]),
                "actual_pad_quat_w": float(env._pad_quat_w[0, 3]),
            })
            if bool(terminated[0] | truncated[0]):
                log = env._repolish_log.pop(0, None)
            if int(args.partial_trace_interval) > 0 and steps % int(args.partial_trace_interval) == 0:
                _write_csv(os.path.join(out_dir, "partial_force_trace.csv"), trace_rows)
            if int(args.progress_interval) > 0 and steps % int(args.progress_interval) == 0:
                print(f"[Gate F4] {args.candidate_id} step={steps}", flush=True)
        if log is None:
            raise RuntimeError(
                f"{args.candidate_id} did not complete in {args.max_control_steps} control steps"
            )
        final = env.gate_f4_final_diagnostic.get(0)
        latched = env.gate_f4_latched_safety.get(0)
        if final is None or latched is None:
            _write_csv(os.path.join(out_dir, "incomplete_force_trace.csv"), trace_rows)
            failure = {
                "control_steps": steps,
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
                "repolish_log": log,
                "final_diagnostic_captured": final is not None,
                "latched_safety_captured": latched is not None,
            }
            with open(os.path.join(out_dir, "incomplete.json"), "w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, sort_keys=True)
            raise RuntimeError(
                "final diagnostic or latched safety was not captured: "
                + json.dumps(failure, sort_keys=True)
            )
        final_tiles = _tile_rows(args.candidate_id, "final", final["tile_maps"])
        pass_history = log.get("pass_history", [])
        if len(pass_history) != 1:
            raise RuntimeError(f"expected exactly one pass history row, got {len(pass_history)}")
        pass_row = pass_history[0]
        trace_summary = _trace_summary(trace_rows, float(cfg.sensor_valid_min_n))
        final_scalars = final["scalars"]
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gate": args.gate_label,
            "candidate_id": args.candidate_id,
            "surface_kind": args.surface_kind,
            "curvature_radius_m": float(args.radius_m),
            "freeform_seed": int(args.freeform_seed),
            "direction_mode": args.direction_mode,
            "surface_seed": int(args.surface_seed),
            "physics_seed": int(args.physics_seed),
            "vertical_pad": args.pad_alignment == "vertical",
            "normal_aligned_pad": args.pad_alignment == "normal",
            "pad_alignment": args.pad_alignment,
            "vertical_tracking_compensation_m": float(
                cfg.vertical_tracking_compensation_m),
            "zero_action": True,
            "training_performed": False,
            "complete": True,
            "control_steps": steps,
            "outcome": log["outcome"],
            "quality_ok": bool(log["quality_ok"]),
            "safety_ok": bool(log["safety_ok"]),
            **latched,
            **trace_summary,
            "fallback_steps": int(pass_row["fallback_steps"]),
            "no_contact_removal_errors": int(env._no_contact_removal_errors),
            "force_used_mean_n": float(pass_row["force_used_mean_n"]),
            "force_used_max_n": float(pass_row["force_used_max_n"]),
            "gu_before": float(pass_row["gu_before"]),
            "gu_after": float(pass_row["gu_after"]),
            "ra_before_um": float(pass_row["ra_before_um"]),
            "ra_after_um": float(pass_row["ra_after_um"]),
            "rz_before_um": float(pass_row["rz_before_um"]),
            "rz_after_um": float(pass_row["rz_after_um"]),
            "scratch_before_um": float(pass_row["scratch_before_um"]),
            "scratch_after_um": float(pass_row["scratch_after_um"]),
            "clearcoat_after_um": float(pass_row["clearcoat_after_um"]),
            "temperature_peak_c": float(pass_row["temperature_peak_c"]),
            "roi_removal_mean_um": float(final_scalars["roi_removal_mean_um"]),
            "roi_removal_max_min_um": float(final_scalars["roi_removal_max_min_um"]),
            "roi_removal_cv": float(final_scalars["roi_removal_cv"]),
            "roi_center_removal_mean_um": float(final_scalars["roi_center_removal_mean_um"]),
            "roi_edge_removal_mean_um": float(final_scalars["roi_edge_removal_mean_um"]),
            "roi_center_edge_delta_um": float(final_scalars["roi_center_edge_delta_um"]),
            "roi_coverage_fraction": float(final_scalars["roi_coverage_fraction"]),
            "mesh_vertex_count": int(getattr(env, "_gate_f_mesh_vertex_count", 0)),
            "mesh_triangle_count": int(getattr(env, "_gate_f_mesh_triangle_count", 0)),
        }
        numeric = [
            value for value in summary.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        summary["all_finite"] = bool(np.isfinite(numeric).all())

        _write_csv(os.path.join(out_dir, "force_trace.csv"), trace_rows)
        _write_csv(os.path.join(out_dir, "tiles.csv"), initial_tiles + final_tiles)
        _write_csv(os.path.join(out_dir, "pass.csv"), [{
            "candidate_id": args.candidate_id,
            **{key: value for key, value in pass_row.items()
               if not isinstance(value, (dict, list))},
        }])
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        metadata = {
            "created_utc": summary["created_utc"],
            "gate": args.gate_label,
            "environment": "GateF4DiagnosticEnv",
            "candidate_id": args.candidate_id,
            "surface_kind": args.surface_kind,
            "curvature_radius_m": float(args.radius_m),
            "freeform_seed": int(args.freeform_seed),
            "path": dict(env.factory_path_metadata),
            "profile": FACTORY_PREPOLISH,
            "surface_seed": int(args.surface_seed),
            "physics_seed": int(args.physics_seed),
            "physical_contact": True,
            "pad_alignment": args.pad_alignment,
            "vertical_tracking_compensation_m": float(
                cfg.vertical_tracking_compensation_m),
            "action": [0.0, 0.0],
            "feed_speed_mm_s": 12.7,
            "force_hard_limit_n": float(cfg.force_hard_limit_n),
            "sensor_valid_min_n": float(cfg.sensor_valid_min_n),
            "physx_executed": True,
            "training_performed": False,
            "metric_status": "PT-DESIGN_SYNTHETIC_CURVED_PHYSX_DIAGNOSTIC",
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
