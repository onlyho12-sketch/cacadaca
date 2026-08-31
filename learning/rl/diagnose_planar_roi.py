"""Gate B physical-contact diagnostic on a large plate and centered ROI.

This is intentionally separate from the established training/evaluation entry
points.  It runs the frozen Robot champion without changing its observations or
weights, and writes only new files under a caller-provided output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_sequences", type=int, default=1)
parser.add_argument("--max_passes", type=int, default=6)
parser.add_argument("--cooldown_s", type=float, default=20.0)
parser.add_argument("--feed_speed_mm_s", type=float, default=12.7)
parser.add_argument("--max_control_steps", type=int, default=180000)
parser.add_argument("--out_dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.rl.env.planar_roi_diagnostics import diagnose_roi  # noqa: E402
from learning.rl.env.planar_roi_polish_env import PlanarRoiPolishEnv  # noqa: E402
from learning.rl.env.planar_roi_polish_env_cfg import PlanarRoiPolishEnvCfg  # noqa: E402


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_policy(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"},
    ).to(device)
    actor.load_state_dict(state)
    actor.eval()

    def policy(obs):
        td = TensorDict({"policy": obs["policy"][:, :obs_dim]}, batch_size=[len(obs["policy"])])
        return actor(td).clamp(-1.0, 1.0)

    return policy, obs_dim


def _flatten_initial(env: PlanarRoiPolishEnv, env_id: int) -> tuple[dict, list[dict]]:
    diag = diagnose_roi(
        env._surfaces[env_id], env.roi_geometry,
        waviness_sigma_m=env.cfg.diagnostic_waviness_sigma_m,
        under_over_fraction=env.cfg.diagnostic_under_over_fraction,
        tiles=tuple(env.cfg.diagnostic_tiles),
    )
    row = {"env": env_id, "sequence": 0, **diag["scalars"]}
    tiles = []
    for tile_x, tile_y in np.ndindex(tuple(env.cfg.diagnostic_tiles)):
        tile = {"env": env_id, "sequence": 0, "stage": "initial",
                "pass": 0, "tile_x": tile_x, "tile_y": tile_y}
        for name, values in diag["tile_maps"].items():
            tile[name] = float(values[tile_x, tile_y])
        tiles.append(tile)
    return row, tiles


def _pass_tile_rows(env_id: int, sequence: int, pass_row: dict) -> list[dict]:
    maps = {name: np.asarray(values, dtype=float) for name, values in
            json.loads(pass_row["roi_tile_maps_json"]).items()}
    out = []
    for tile_x, tile_y in np.ndindex(next(iter(maps.values())).shape):
        tile = {"env": env_id, "sequence": sequence, "stage": "after_pass",
                "pass": int(pass_row["pass"]), "tile_x": tile_x, "tile_y": tile_y}
        for name, values in maps.items():
            tile[name] = float(values[tile_x, tile_y])
        out.append(tile)
    return out


def _pass_delta_rows(pass_rows: list[dict]) -> list[dict]:
    by_sequence: dict[tuple[int, int], list[dict]] = {}
    for row in pass_rows:
        by_sequence.setdefault((int(row["env"]), int(row["sequence"])), []).append(row)
    metrics = [
        "gu_after", "ra_after_um", "rz_after_um", "scratch_after_um",
        "clearcoat_after_um", "roi_total_ra_um", "roi_total_rz_um",
        "roi_fine_ra_um", "roi_fine_rz_um",
        "roi_ra_low_frequency_contribution_um",
        "roi_rz_low_frequency_contribution_um", "roi_removal_mean_um",
        "roi_removal_std_um", "roi_removal_cv", "roi_removal_max_min_um",
        "roi_removal_p95_p05_um", "roi_removal_waviness_ra_um",
        "roi_removal_waviness_rz_um", "roi_removal_waviness_std_um",
        "roi_removal_fine_std_um", "roi_center_edge_delta_um",
        "roi_center_edge_ratio", "roi_under_fraction", "roi_over_fraction",
        "roi_coverage_fraction",
    ]
    rows = []
    for (env_id, sequence), values in sorted(by_sequence.items()):
        values.sort(key=lambda r: int(r["pass"]))
        for before, after in zip(values, values[1:]):
            row = {"env": env_id, "sequence": sequence,
                   "from_pass": int(before["pass"]), "to_pass": int(after["pass"])}
            for metric in metrics:
                row[f"{metric}_before"] = float(before[metric])
                row[f"{metric}_after"] = float(after[metric])
                row[f"{metric}_delta"] = float(after[metric]) - float(before[metric])
            rows.append(row)
    return rows


def main() -> None:
    if args.num_sequences != 1:
        raise ValueError("Gate B freezes one sequence per env; use --num_sequences 1")
    checkpoint = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    os.makedirs(out_dir)

    cfg = PlanarRoiPolishEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_feed_speed_mm_s = args.feed_speed_mm_s
    cfg.enable_pad_physical_contact = True
    cfg.repolish_max_passes = args.max_passes
    cfg.repolish_cooldown_s = args.cooldown_s
    # The 200 mm ROI makes one state-machine pass about 4 m.  This horizon
    # accommodates six passes even at the policy's minimum allowed feed.
    cfg.episode_length_s = args.max_passes * (700.0 + args.cooldown_s) + 120.0

    env = PlanarRoiPolishEnv(cfg, render_mode=None)
    env._repolish_mode = True
    policy, obs_dim = _load_policy(checkpoint, env.device)
    obs, _ = env.reset()

    initial_rows, tile_rows = [], []
    for env_id in range(args.num_envs):
        initial, tiles = _flatten_initial(env, env_id)
        initial_rows.append(initial)
        tile_rows.extend(tiles)

    rows, pass_rows = [], []
    seq_done = np.zeros(args.num_envs, dtype=int)
    target = args.num_envs
    step = 0
    while len(rows) < target and step < args.max_control_steps:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, terminated, truncated, _ = env.step(actions)
        step += 1
        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            if seq_done[env_id] >= 1:
                continue
            log = env._repolish_log.pop(env_id, None)
            if log is None:
                continue
            before, final = log["before"], log["final"]
            rows.append({
                "checkpoint": os.path.basename(checkpoint), "env": env_id,
                "sequence": 0, "outcome": log["outcome"], "passes": log["passes"],
                "gu_before": before["gu"], "gu_final": final["gu"],
                "ra_before_um": before["ra"], "ra_final_um": final["ra"],
                "rz_before_um": before["rz"], "rz_final_um": final["rz"],
                "scratch_before_um": before["scratch"],
                "scratch_final_um": final["scratch"],
                "clearcoat_min_um": final["cc_min"],
                "temperature_peak_c": final["temperature_peak_c"],
                "quality_ok": log["quality_ok"], "safety_ok": log["safety_ok"],
            })
            for item in log.get("pass_history", []):
                row = {"checkpoint": os.path.basename(checkpoint), "env": env_id,
                       "sequence": 0, **item}
                pass_rows.append(row)
                tile_rows.extend(_pass_tile_rows(env_id, 0, item))
            seq_done[env_id] += 1
        if step % 2000 == 0:
            print(f"[Gate B] step={step} sequences={len(rows)}/{target}", flush=True)

    _write_csv(os.path.join(out_dir, "initial_diagnostics.csv"), initial_rows)
    _write_csv(os.path.join(out_dir, "planar_roi_sequences.csv"), rows)
    _write_csv(os.path.join(out_dir, "planar_roi_passes.csv"), pass_rows)
    _write_csv(os.path.join(out_dir, "planar_roi_tiles.csv"), tile_rows)
    _write_csv(os.path.join(out_dir, "pass_deltas.csv"), _pass_delta_rows(pass_rows))

    geometry = env.roi_geometry.validate()
    geometry.update({
        "map_size_x_m": env.roi_geometry.map_size_m[0],
        "map_size_y_m": env.roi_geometry.map_size_m[1],
        "roi_size_x_m": env.roi_geometry.roi_size_m[0],
        "roi_size_y_m": env.roi_geometry.roi_size_m[1],
        "resolution_m": env.roi_geometry.resolution_m,
        "pad_radius_m": env.roi_geometry.pad_radius_m,
        "step_over_ratio": env.recipe.step_over_spacing_ratio,
        "step_over_m": env.recipe.step_over_spacing_ratio * 2.0 * env.roi_geometry.pad_radius_m,
        "raster_lines": len(env._lines), "state_machine_path_length_m": env._path_len,
    })
    _write_csv(os.path.join(out_dir, "geometry.csv"), [geometry])
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint, "checkpoint_sha256": _sha256(checkpoint),
        "obs_dim": obs_dim, "num_envs": args.num_envs, "num_sequences": 1,
        "max_passes": args.max_passes, "cooldown_s": args.cooldown_s,
        "feed_speed_mm_s": args.feed_speed_mm_s, "physical_contact": True,
        "steps_executed": step, "sequences_collected": len(rows),
        "complete": len(rows) == target, "geometry": geometry,
        "metric_note": ("Fine/low-frequency split is a PT-DESIGN Gaussian diagnostic; "
                        "it is not an ISO profilometer cutoff."),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"[Gate B] wrote {out_dir}")
    print(f"[Gate B] collected={len(rows)}/{target} steps={step}")
    for outcome, count in sorted(
        {name: sum(r["outcome"] == name for r in rows)
         for name in {r["outcome"] for r in rows}}.items()
    ):
        print(f"[Gate B] {outcome}: {count}/{len(rows)}")
    env.close()
    if len(rows) != target:
        raise RuntimeError(f"incomplete collection: {len(rows)}/{target}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
