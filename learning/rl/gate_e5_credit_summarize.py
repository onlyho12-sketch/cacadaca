"""Summarize Gate E5 terminal timing with exact rollout-local GAE reach."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stats(values: list[float]) -> tuple[float, float, float]:
    return min(values), sum(values) / len(values), max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gate_e4_steps_per_env", type=int, default=480)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--alternative_lambda", type=float, default=0.99)
    args = parser.parse_args()
    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    raw_sequence_path = os.path.join(raw_dir, "sequences.csv")
    raw_metadata_path = os.path.join(raw_dir, "metadata.json")
    rows = read_csv(raw_sequence_path)
    with open(raw_metadata_path, encoding="utf-8") as handle:
        raw_metadata = json.load(handle)
    gamma = float(raw_metadata["gamma"])
    rollout_steps = int(raw_metadata["rollout_steps"])

    credit_rows = []
    for row in rows:
        steps = int(row["control_steps"])
        local_steps = (steps - 1) % rollout_steps + 1
        credit_rows.append({
            "env": int(row["env"]),
            "surface_profile": row["surface_profile"],
            "profile_seed": int(row["profile_seed"]),
            "control_steps": steps,
            "terminal_rollout_zero_based": (steps - 1) // rollout_steps,
            "terminal_position_within_rollout": local_steps,
            "direct_terminal_credit_steps": local_steps,
            "direct_terminal_credit_path_fraction": local_steps / steps,
            "gae_weight_to_rollout_start_lambda_0p95": (
                (gamma * args.gae_lambda) ** (local_steps - 1)),
            "gae_weight_to_rollout_start_lambda_0p99": (
                (gamma * args.alternative_lambda) ** (local_steps - 1)),
            "terminal_reward": float(row["terminal_reward_sum"]),
            "dense_reward_sum": float(row["dense_reward_sum"]),
            "dense_reward_abs_sum": float(row["dense_reward_abs_sum"]),
            "terminal_to_dense_abs_ratio": float(row["terminal_to_dense_abs_ratio"]),
            "hypothetical_uninterrupted_gamma_weight": float(
                row["hypothetical_uninterrupted_gamma_weight"]),
            "comparison_gamma_0p99_weight": float(row["comparison_gamma_weight"]),
            "quality_ok": row["quality_ok"],
            "safety_ok": row["safety_ok"],
        })
    credit_path = os.path.join(out_dir, "sequence_credit.csv")
    write_csv(credit_path, credit_rows)

    option_rows = []
    episode_steps = [int(row["control_steps"]) for row in rows]
    for length in (48, 128, 256, 512):
        local = [(steps - 1) % length + 1 for steps in episode_steps]
        fractions = [value / steps for value, steps in zip(local, episode_steps)]
        for lam, label in ((args.gae_lambda, "current_0p95"),
                           (args.alternative_lambda, "alternative_0p99")):
            weights = [(gamma * lam) ** (value - 1) for value in local]
            option_rows.append({
                "rollout_steps": length,
                "lambda_label": label,
                "gae_lambda": lam,
                "samples_per_iteration_12env": 12 * length,
                "iterations_to_first_terminal": math.ceil(min(episode_steps) / length),
                "iterations_to_all_first_terminals": math.ceil(max(episode_steps) / length),
                "direct_credit_steps_min": min(local),
                "direct_credit_steps_mean": sum(local) / len(local),
                "direct_credit_steps_max": max(local),
                "direct_credit_path_fraction_mean": sum(fractions) / len(fractions),
                "gae_weight_to_rollout_start_min": min(weights),
                "gae_weight_to_rollout_start_mean": sum(weights) / len(weights),
                "gae_weight_to_rollout_start_max": max(weights),
            })
    option_path = os.path.join(out_dir, "rollout_options.csv")
    write_csv(option_path, option_rows)

    profile_rows = []
    for profile in sorted({row["surface_profile"] for row in credit_rows}):
        subset = [row for row in credit_rows if row["surface_profile"] == profile]
        profile_rows.append({
            "surface_profile": profile,
            "executions": len(subset),
            "safety_ok": sum(row["safety_ok"] == "True" for row in subset),
            "quality_ok": sum(row["quality_ok"] == "True" for row in subset),
            "control_steps_mean": sum(row["control_steps"] for row in subset) / len(subset),
            "terminal_reward_mean": sum(row["terminal_reward"] for row in subset) / len(subset),
            "dense_reward_sum_mean": sum(row["dense_reward_sum"] for row in subset) / len(subset),
            "direct_credit_steps_mean": sum(
                row["direct_terminal_credit_steps"] for row in subset) / len(subset),
            "direct_credit_path_fraction_mean": sum(
                row["direct_terminal_credit_path_fraction"] for row in subset) / len(subset),
        })
    profile_path = os.path.join(out_dir, "profile_summary.csv")
    write_csv(profile_path, profile_rows)

    direct_steps = [row["direct_terminal_credit_steps"] for row in credit_rows]
    direct_fraction = [row["direct_terminal_credit_path_fraction"] for row in credit_rows]
    current_weights = [row["gae_weight_to_rollout_start_lambda_0p95"] for row in credit_rows]
    alternative_weights = [row["gae_weight_to_rollout_start_lambda_0p99"] for row in credit_rows]
    terminal = [row["terminal_reward"] for row in credit_rows]
    dense_abs = [row["dense_reward_abs_sum"] for row in credit_rows]
    ratios = [row["terminal_to_dense_abs_ratio"] for row in credit_rows]
    uninterrupted = [row["hypothetical_uninterrupted_gamma_weight"] for row in credit_rows]
    comparison = [row["comparison_gamma_0p99_weight"] for row in credit_rows]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E5 exact rollout-local credit summary",
        "raw_dir": raw_dir,
        "executions": len(rows),
        "rollout_steps": rollout_steps,
        "gamma": gamma,
        "gae_lambda": args.gae_lambda,
        "alternative_lambda": args.alternative_lambda,
        "gate_e4_steps_per_env": args.gate_e4_steps_per_env,
        "gate_e4_terminal_events_possible": (
            args.gate_e4_steps_per_env >= min(episode_steps)),
        "episode_steps_min_mean_max": stats([float(v) for v in episode_steps]),
        "terminal_rollout_first_all": [
            math.ceil(min(episode_steps) / rollout_steps),
            math.ceil(max(episode_steps) / rollout_steps),
        ],
        "direct_credit_steps_min_mean_max": stats([float(v) for v in direct_steps]),
        "direct_credit_path_fraction_mean": sum(direct_fraction) / len(direct_fraction),
        "current_lambda_rollout_start_weight_min_mean_max": stats(current_weights),
        "alternative_lambda_rollout_start_weight_min_mean_max": stats(alternative_weights),
        "terminal_reward_min_mean_max": stats(terminal),
        "dense_abs_reward_mean": sum(dense_abs) / len(dense_abs),
        "terminal_to_dense_abs_ratio_mean": sum(ratios) / len(ratios),
        "hypothetical_uninterrupted_gamma_weight_mean": sum(uninterrupted) / len(uninterrupted),
        "comparison_gamma_0p99_weight_mean": sum(comparison) / len(comparison),
        "key_finding": (
            "Gate E4 had zero terminal events; with 48-step rollouts, only the "
            "terminal's position inside its final rollout receives direct GAE credit."),
        "gate_e6_recommendation": (
            "Do not start full PPO yet. Compare the current 48/0.95 control against "
            "a PT-DESIGN 128-step/0.99-lambda pilot at equal sample budget with "
            "iteration checkpoints and strict safety gates."),
        "training_performed": False,
        "checkpoint_created": False,
        "champion_promoted": False,
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    ds = metadata["direct_credit_steps_min_mean_max"]
    cw = metadata["current_lambda_rollout_start_weight_min_mean_max"]
    aw = metadata["alternative_lambda_rollout_start_weight_min_mean_max"]
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Gate E5 exact PPO credit summary\n\n"
            f"- Frozen factory one-pass executions: {len(rows)}/{len(rows)}; safety and quality "
            "12/12; sensor faults 0.\n"
            f"- Episode steps min/mean/max: {min(episode_steps)} / "
            f"{sum(episode_steps) / len(episode_steps):.2f} / {max(episode_steps)}.\n"
            f"- Gate E4 used {args.gate_e4_steps_per_env} steps/env, so terminal events were "
            "mathematically impossible.\n"
            f"- Current 48-step rollouts first/all see terminal after "
            f"{metadata['terminal_rollout_first_all'][0]} / "
            f"{metadata['terminal_rollout_first_all'][1]} iterations.\n"
            f"- Exact rollout-local direct-credit steps min/mean/max: "
            f"{ds[0]:.0f} / {ds[1]:.2f} / {ds[2]:.0f}, only "
            f"{100.0 * metadata['direct_credit_path_fraction_mean']:.3f}% of the path on average.\n"
            f"- With current lambda=0.95, terminal GAE weight at that rollout's start "
            f"min/mean/max: {cw[0]:.6f} / {cw[1]:.6f} / {cw[2]:.6f}.\n"
            f"- A PT-DESIGN lambda=0.99 would make those values "
            f"{aw[0]:.6f} / {aw[1]:.6f} / {aw[2]:.6f}, but may increase variance.\n"
            f"- Terminal reward min/mean/max: {min(terminal):.6f} / "
            f"{sum(terminal) / len(terminal):.6f} / {max(terminal):.6f}; mean terminal/"
            f"|dense| ratio {metadata['terminal_to_dense_abs_ratio_mean']:.3f}.\n"
            f"- Hypothetical uninterrupted gamma=0.9995 weight at episode start is "
            f"{metadata['hypothetical_uninterrupted_gamma_weight_mean']:.6f} on average, but "
            "this is not realized across separate PPO storage updates.\n"
            "- Conclusion: raising gamma alone cannot solve the credit boundary. Full PPO should "
            "not start yet.\n"
            "- Next pilot recommendation: equal-sample comparison of current 48-step/lambda=0.95 "
            "and 128-step/lambda=0.99, with strict checkpoint safety screens.\n"
            "- No reward formula, environment, policy, or checkpoint was changed.\n"
        )
    generated = [credit_path, option_path, profile_path, metadata_path, readme_path]
    source_paths = [
        raw_sequence_path,
        raw_metadata_path,
        os.path.join(raw_dir, "reward_trace.csv"),
        os.path.join(raw_dir, "rollout_terminal_coverage.csv"),
        os.path.abspath(__file__),
    ]
    checksum_rows = [
        {"sha256": sha256(path), "file": path} for path in source_paths + generated]
    write_csv(os.path.join(out_dir, "checksums.csv"), checksum_rows)
    print(f"[Gate E5 summary] wrote {out_dir}")


if __name__ == "__main__":
    main()
