"""D단계 — 재폴리싱 상태기계 평가 (인수인계서 19장).

한 pass만 도는 학습/평가와 달리, 여기서는 정책이 실제로 목표에 도달할 때까지
(또는 실패 조건에 걸릴 때까지) 같은 표면을 반복 폴리싱한다:

    1 pass 수행 → 품질·안전 평가
      → 통과: 종료(success)
      → 미달 + 안전: 냉각 후 같은 표면에서 다음 pass
      → 안전 위반 / 최대 pass 초과 / 개선 없음: 실패 종료

이 순환은 RobotPolishEnv._get_dones()(repolish_mode=True일 때)가 내부에서 수행하고,
이 스크립트는 env.step() 을 반복 호출하며 env._repolish_log 에 쌓이는 시퀀스별
결과를 읽어 집계·CSV로 저장하는 바깥 루프다.

    /home/rokey/isaacsim-6.0.1/python.sh learning/rl/repolish_eval.py --headless \
        --checkpoint learning/rl/robot/logs/2026-08-29_13-14-15/model_700.pt \
        --num_envs 8 --num_sequences 3 --max_passes 6 --cooldown_s 20

주의: 재폴리싱은 반드시 물리 접촉 모드(PhysX 센서힘)로만 수행한다 — force_model_n
기반으로는 "실제로 목표에 도달했다"는 성공 판정을 신뢰할 수 없다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--policy_mode",
                    choices=("checkpoint", "zero", "finish_rule", "checkpoint_finish"),
                    default="checkpoint",
                    help="checkpoint 정책, residual=0 기준, 후속 약한 마무리 규칙, "
                         "또는 1차 checkpoint+2차 마무리 규칙")
parser.add_argument("--finish_force_action", type=float, default=-0.5,
                    help="finish_rule의 2차 force residual")
parser.add_argument("--finish_late_force_action", type=float, default=-1.0,
                    help="finish_rule의 3차 이후 force residual")
parser.add_argument("--finish_feed_mm_s", type=float, default=8.0,
                    help="finish_rule의 2차 이후 목표 feed")
parser.add_argument("--finish_min_gu", type=float, default=68.0,
                    help="finish_rule에서 3차 진입을 허용할 2차 종료 GU proxy 하한")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--feed_speed_mm_s", type=float, default=None,
                    help="로봇 환경 기준 이송속도 재정의 (기본: Tesla polishing 12.7 mm/s)")
parser.add_argument("--num_sequences", type=int, default=3,
                    help="env 당 반복할 (새 표면 → 재폴리싱 완료까지) 시퀀스 수")
parser.add_argument("--max_passes", type=int, default=6)
parser.add_argument("--cooldown_s", type=float, default=20.0)
parser.add_argument("--max_control_steps", type=int, default=200000,
                    help="안전 상한 — 이 스텝 안에 목표 시퀀스 수를 못 채우면 중단")
parser.add_argument("--out", type=str,
                    default=os.path.join(_REPO_ROOT, "learning", "rl", "robot", "results",
                                         "repolish_eval.csv"))
parser.add_argument("--pass_out", type=str, default=None,
                    help="pass별 진단 CSV (기본: --out 파일명 뒤에 _passes 추가)")
parser.add_argument("--tile_out", type=str, default=None,
                    help="타일별 현재 GU·낙관적 상한·처분 CSV (기본: --out 뒤에 _tiles 추가)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from learning.rl.env.robot_polish_env import RobotPolishEnv  # noqa: E402
from learning.rl.env.robot_polish_env_cfg import RobotPolishEnvCfg  # noqa: E402


def load_policy(checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    actor = MLPModel(
        dummy, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[128, 128], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                          "std_type": "scalar"}).to(device)
    actor.load_state_dict(state)
    actor.eval()

    def policy(obs):
        x = obs["policy"][:, :obs_dim]
        td = TensorDict({"policy": x}, batch_size=[len(x)])
        return actor(td).clamp(-1.0, 1.0)

    print(f"[repolish] loaded {checkpoint} (obs_dim={obs_dim})")
    return policy


def main():
    env_cfg = RobotPolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.feed_speed_mm_s is not None:
        env_cfg.robot_feed_speed_mm_s = args.feed_speed_mm_s
    env_cfg.enable_pad_physical_contact = True   # repolish는 반드시 실측 PhysX 힘 기준
    env_cfg.repolish_max_passes = args.max_passes
    env_cfg.repolish_cooldown_s = args.cooldown_s
    if args.policy_mode == "finish_rule":
        env_cfg.repolish_finish_gate_after_pass = 2
        env_cfg.repolish_finish_min_gu = args.finish_min_gu
    # 한 시퀀스(최대 pass 수)를 다 담을 수 있도록 넉넉히: 공칭 1 pass 완주 시간(BO recipe
    # 기준 raster 총 길이/feed, 대략 200~250s) + pass당 냉각시간을 max_passes 배 확보.
    nominal_pass_s = 260.0
    env_cfg.episode_length_s = args.max_passes * (nominal_pass_s + args.cooldown_s) + 60.0

    env = RobotPolishEnv(env_cfg, render_mode=None)
    env._repolish_mode = True
    print(f"[repolish] envs={args.num_envs} num_sequences={args.num_sequences} "
          f"max_passes={args.max_passes} cooldown={args.cooldown_s}s "
          f"episode_length_s={env_cfg.episode_length_s:.0f}")

    if args.policy_mode == "zero":
        def policy(obs):
            return torch.zeros((len(obs["policy"]), 2), device=env.device)
        policy_name = "zero_action"
        print("[repolish] policy=zero_action (force/feed residual 모두 0)")
    elif args.policy_mode == "finish_rule":
        base_feed = env.recipe.feed_speed_mm_s
        finish_feed_action = ((args.finish_feed_mm_s / base_feed) - 1.0) / env.cfg.feed_ratio_limit
        finish_feed_action = float(np.clip(finish_feed_action, -1.0, 1.0))

        def policy(obs):
            actions = torch.zeros((len(obs["policy"]), 2), device=env.device)
            later = env._pass_count > 0
            late = env._pass_count > 1
            actions[later, 0] = float(np.clip(args.finish_force_action, -1.0, 1.0))
            actions[late, 0] = float(np.clip(args.finish_late_force_action, -1.0, 1.0))
            actions[later, 1] = finish_feed_action
            return actions

        policy_name = "finish_rule"
        print(f"[repolish] policy=finish_rule (pass1 residual=0; pass2 force_action="
              f"{args.finish_force_action:.3f}; pass3+ force_action="
              f"{args.finish_late_force_action:.3f}; feed={args.finish_feed_mm_s:.3f} mm/s, "
              f"feed_action={finish_feed_action:.3f}; finish_min_gu={args.finish_min_gu:.2f})")
    elif args.policy_mode == "checkpoint_finish":
        checkpoint_policy = load_policy(args.checkpoint, env.device)
        base_feed = env.recipe.feed_speed_mm_s
        finish_feed_action = ((args.finish_feed_mm_s / base_feed) - 1.0) / env.cfg.feed_ratio_limit
        finish_feed_action = float(np.clip(finish_feed_action, -1.0, 1.0))

        def policy(obs):
            # Pass 1 keeps the current champion exactly as-is.  From pass 2 onward,
            # replace only the two residual actions with the controlled finish pair.
            actions = checkpoint_policy(obs).clone()
            later = env._pass_count > 0
            actions[later, 0] = float(np.clip(args.finish_force_action, -1.0, 1.0))
            actions[later, 1] = finish_feed_action
            return actions

        policy_name = (
            f"{os.path.basename(args.checkpoint)}+finish"
            f"(force={float(np.clip(args.finish_force_action, -1.0, 1.0)):.3f},"
            f"feed={args.finish_feed_mm_s:.3f})"
        )
        print(f"[repolish] policy=checkpoint_finish (pass1={os.path.basename(args.checkpoint)}; "
              f"pass2+ force_action={args.finish_force_action:.3f}; "
              f"feed={args.finish_feed_mm_s:.3f} mm/s, feed_action={finish_feed_action:.3f})")
    else:
        policy = load_policy(args.checkpoint, env.device)
        policy_name = os.path.basename(args.checkpoint)

    obs, _ = env.reset()
    seq_done = np.zeros(args.num_envs, dtype=int)
    rows = []
    pass_rows = []
    tile_rows = []
    step = 0
    target = args.num_envs * args.num_sequences
    while len(rows) < target and step < args.max_control_steps:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, terminated, truncated, _ = env.step(actions)
        step += 1
        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        for i in done_ids:
            if seq_done[i] >= args.num_sequences:
                continue
            log = env._repolish_log.pop(i, None)
            if log is None:
                continue
            b, f = log["before"], log["final"]
            history = log.get("pass_history", [])
            best_pass = 0
            best_gu = float(b["gu"])
            for p in history:
                if float(p["gu_after"]) > best_gu:
                    best_gu = float(p["gu_after"])
                    best_pass = int(p["pass"])
            outcome = log["outcome"]
            if outcome == "success":
                disposition = "success"
            elif outcome in {"fail_clearcoat", "fail_clearcoat_budget"}:
                disposition = "recoat_or_refinish_review"
            elif outcome in {"fail_overheat", "fail_force_overload",
                             "fail_unstable_contact", "fail_timeout"}:
                disposition = "human_review_required_safety_stop"
            else:
                disposition = "best_safe_result_current_process_limit"
            rows.append({
                "checkpoint": policy_name, "env": i,
                "sequence": seq_done[i], "outcome": outcome, "disposition": disposition,
                "passes": log["passes"], "best_observed_pass": best_pass,
                "best_observed_gu": round(best_gu, 2),
                "gu_before": round(b["gu"], 2), "gu_final": round(f["gu"], 2),
                "ra_final_um": round(f["ra"], 4), "rz_final_um": round(f["rz"], 3),
                "scratch_before_um": round(b["scratch"], 3), "scratch_final_um": round(f["scratch"], 3),
                "clearcoat_min_um": round(f["cc_min"], 2),
                "temperature_peak_c": round(f["temperature_peak_c"], 2),
                "thermal_damage_peak": round(f["thermal_damage_peak"], 6),
                "quality_ok": log["quality_ok"], "safety_ok": log["safety_ok"],
            })
            for p in history:
                pass_rows.append({
                    "checkpoint": policy_name, "env": i,
                    "sequence": seq_done[i], **p,
                })
                gu_map = np.asarray(json.loads(p["tile_gu_map_json"]), dtype=float)
                ceiling_map = np.asarray(
                    json.loads(p["tile_optimistic_ceiling_gu_map_json"]), dtype=float)
                gu_min_baseline_map = np.asarray(
                    json.loads(p["tile_gu_min_baseline_map_json"]), dtype=float)
                gu_cellwise_worst_map = np.asarray(
                    json.loads(p["tile_gu_cellwise_worst_map_json"]), dtype=float)
                ceiling_min_baseline_map = np.asarray(json.loads(
                    p["tile_optimistic_ceiling_min_baseline_map_json"]), dtype=float)
                ceiling_cellwise_worst_map = np.asarray(json.loads(
                    p["tile_optimistic_ceiling_cellwise_worst_map_json"]), dtype=float)
                q_cc_current_map = np.asarray(
                    json.loads(p["tile_q_clearcoat_current_map_json"]), dtype=float)
                q_cc_min_baseline_map = np.asarray(
                    json.loads(p["tile_q_clearcoat_min_baseline_map_json"]), dtype=float)
                q_cc_cellwise_worst_map = np.asarray(
                    json.loads(p["tile_q_clearcoat_cellwise_worst_map_json"]), dtype=float)
                limiter_map = np.asarray(json.loads(p["tile_limiter_map_json"]), dtype=object)
                cc_map = np.asarray(json.loads(p["tile_cc_min_map_json"]), dtype=float)
                cc_initial_min_map = np.asarray(
                    json.loads(p["tile_cc_initial_min_map_json"]), dtype=float)
                cc_initial_mean_map = np.asarray(
                    json.loads(p["tile_cc_initial_mean_map_json"]), dtype=float)
                for tile_x, tile_y in np.ndindex(gu_map.shape):
                    gu = float(gu_map[tile_x, tile_y])
                    ceiling = float(ceiling_map[tile_x, tile_y])
                    cc_min = float(cc_map[tile_x, tile_y])
                    if gu >= env.cfg.repolish_target_gu:
                        tile_disposition = "target_met"
                    elif ceiling < env.cfg.repolish_target_gu:
                        tile_disposition = "current_clearcoat_or_thermal_limit"
                    elif cc_min <= (env.cfg.clearcoat_safety_limit_um
                                    + env.cfg.repolish_cc_safety_margin_um):
                        tile_disposition = "clearcoat_budget_stop"
                    else:
                        tile_disposition = "plausible_rework_candidate"
                    tile_rows.append({
                        "checkpoint": policy_name,
                        "env": i,
                        "sequence": seq_done[i],
                        "pass": int(p["pass"]),
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "gu_current": round(gu, 3),
                        "gu_optimistic_ceiling": round(ceiling, 3),
                        "gu_if_min_baseline": round(
                            float(gu_min_baseline_map[tile_x, tile_y]), 3),
                        "gu_if_cellwise_worst": round(
                            float(gu_cellwise_worst_map[tile_x, tile_y]), 3),
                        "gu_ceiling_if_min_baseline": round(
                            float(ceiling_min_baseline_map[tile_x, tile_y]), 3),
                        "gu_ceiling_if_cellwise_worst": round(
                            float(ceiling_cellwise_worst_map[tile_x, tile_y]), 3),
                        "ceiling_margin_to_target": round(
                            ceiling - env.cfg.repolish_target_gu, 3),
                        "q_clearcoat_current": round(
                            float(q_cc_current_map[tile_x, tile_y]), 6),
                        "q_clearcoat_min_baseline": round(
                            float(q_cc_min_baseline_map[tile_x, tile_y]), 6),
                        "q_clearcoat_cellwise_worst": round(
                            float(q_cc_cellwise_worst_map[tile_x, tile_y]), 6),
                        "clearcoat_initial_min_um": round(
                            float(cc_initial_min_map[tile_x, tile_y]), 3),
                        "clearcoat_initial_mean_um": round(
                            float(cc_initial_mean_map[tile_x, tile_y]), 3),
                        "clearcoat_min_um": round(cc_min, 3),
                        "limiting_term_current": str(limiter_map[tile_x, tile_y]),
                        "disposition": tile_disposition,
                    })
            seq_done[i] += 1
        if step % 2000 == 0:
            print(f"[repolish] step={step} sequences={len(rows)}/{target} "
                  f"(min per-env={int(seq_done.min())})", flush=True)

    if step >= args.max_control_steps and len(rows) < target:
        print(f"[repolish] ⚠ 안전 상한({args.max_control_steps} step) 도달 — "
              f"{len(rows)}/{target} 시퀀스만 수집됨")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n에피소드(시퀀스)별 CSV → {args.out}")
    if pass_rows:
        pass_out = args.pass_out
        if pass_out is None:
            stem, ext = os.path.splitext(args.out)
            pass_out = stem + "_passes" + (ext or ".csv")
        os.makedirs(os.path.dirname(pass_out) or ".", exist_ok=True)
        with open(pass_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pass_rows[0].keys()))
            w.writeheader(); w.writerows(pass_rows)
        print(f"pass별 진단 CSV → {pass_out}")
    if tile_rows:
        tile_out = args.tile_out
        if tile_out is None:
            stem, ext = os.path.splitext(args.out)
            tile_out = stem + "_tiles" + (ext or ".csv")
        os.makedirs(os.path.dirname(tile_out) or ".", exist_ok=True)
        with open(tile_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tile_rows[0].keys()))
            w.writeheader(); w.writerows(tile_rows)
        print(f"타일별 한계 진단 CSV → {tile_out}")

    n = len(rows)
    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    succ = [r for r in rows if r["outcome"] == "success"]
    print(f"\n{'=' * 60}\n재폴리싱 결과 — n={n} 시퀀스 (env={args.num_envs} × seq={args.num_sequences})")
    for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:3d}/{n}  ({v / max(n,1):.0%})")
    if succ:
        passes = np.array([r["passes"] for r in succ])
        gu = np.array([r["gu_final"] for r in succ])
        print(f"\n성공({len(succ)}건) 평균 pass 수: {passes.mean():.2f} (min {passes.min()}, max {passes.max()})")
        print(f"성공 시 최종 GU proxy 평균: {gu.mean():.2f}")
    print(f"전체 성공률: {len(succ)}/{n} = {len(succ) / max(n,1):.0%}")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
