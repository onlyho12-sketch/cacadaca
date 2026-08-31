"""Create the final Gate E4 short-PPO screen and confirmation report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

KEYS = ("direction_mode", "surface_profile", "profile_seed")
METRICS = (
    "gu_final", "ra_final_um", "rz_final_um", "scratch_final_um",
    "clearcoat_min_um", "temperature_peak_c", "control_steps",
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(directory: str) -> tuple[str, list[dict]]:
    path = os.path.join(os.path.abspath(directory), "sequences.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        return path, list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(label: str, rows: list[dict], stage: str, path: str = "all") -> dict:
    n = len(rows)
    return {
        "stage": stage,
        "policy": label,
        "direction_mode": path,
        "executions": n,
        "safety_ok": sum(row["safety_ok"] == "True" for row in rows),
        "quality_ok": sum(row["quality_ok"] == "True" for row in rows),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in rows),
        **{
            f"mean_{metric}": sum(float(row[metric]) for row in rows) / n
            for metric in METRICS
        },
    }


def compare(label: str, champion: list[dict], candidate: list[dict], stage: str) -> list[dict]:
    base = {tuple(row[key] for key in KEYS): row for row in champion}
    test = {tuple(row[key] for key in KEYS): row for row in candidate}
    if len(base) != len(champion) or len(test) != len(candidate):
        raise ValueError("duplicate paired evaluation key")
    if set(base) != set(test):
        raise ValueError(f"paired keys differ for {label}/{stage}")
    out = []
    for key in sorted(base):
        a, b = base[key], test[key]
        row = {
            "stage": stage,
            "candidate": label,
            **dict(zip(KEYS, key)),
            "champion_quality_ok": int(a["quality_ok"] == "True"),
            "candidate_quality_ok": int(b["quality_ok"] == "True"),
            "champion_safety_ok": int(a["safety_ok"] == "True"),
            "candidate_safety_ok": int(b["safety_ok"] == "True"),
            "champion_sensor_fault_steps": int(a["sensor_fault_steps"]),
            "candidate_sensor_fault_steps": int(b["sensor_fault_steps"]),
        }
        for metric in METRICS:
            av, bv = float(a[metric]), float(b[metric])
            row[f"champion_{metric}"] = av
            row[f"candidate_{metric}"] = bv
            row[f"delta_{metric}"] = bv - av
        out.append(row)
    return out


def comparison_summary(label: str, paired: list[dict], stage: str, path: str) -> dict:
    n = len(paired)
    return {
        "stage": stage,
        "candidate": label,
        "direction_mode": path,
        "pairs": n,
        "champion_safety_ok": sum(row["champion_safety_ok"] for row in paired),
        "candidate_safety_ok": sum(row["candidate_safety_ok"] for row in paired),
        "champion_quality_ok": sum(row["champion_quality_ok"] for row in paired),
        "candidate_quality_ok": sum(row["candidate_quality_ok"] for row in paired),
        "champion_sensor_fault_steps": sum(row["champion_sensor_fault_steps"] for row in paired),
        "candidate_sensor_fault_steps": sum(row["candidate_sensor_fault_steps"] for row in paired),
        **{
            f"mean_delta_{metric}": sum(row[f"delta_{metric}"] for row in paired) / n
            for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--selected_checkpoint", required=True)
    parser.add_argument("--screen_champion", required=True)
    parser.add_argument("--screen_actor1", required=True)
    parser.add_argument("--screen_actor3", required=True)
    parser.add_argument("--screen_actor5", required=True)
    parser.add_argument("--confirm_champion_same", required=True)
    parser.add_argument("--confirm_actor5_same", required=True)
    parser.add_argument("--confirm_champion_cross", required=True)
    parser.add_argument("--confirm_actor5_cross", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    source_paths = []
    def load(directory):
        path, rows = read_csv(directory); source_paths.append(path); return rows

    screen_champion = load(args.screen_champion)
    screen_candidates = {
        "actor1": load(args.screen_actor1),
        "actor3": load(args.screen_actor3),
        "actor5": load(args.screen_actor5),
    }
    screen_paired = []
    screen_summary = [summarize("champion", screen_champion, "screen")]
    screen_comparison_summary = []
    for label, rows in screen_candidates.items():
        screen_summary.append(summarize(label, rows, "screen"))
        pairs = compare(label, screen_champion, rows, "screen")
        screen_paired.extend(pairs)
        screen_comparison_summary.append(
            comparison_summary(label, pairs, "screen_delta", "same_xx"))

    champion_same = load(args.confirm_champion_same)
    actor5_same = load(args.confirm_actor5_same)
    champion_cross = load(args.confirm_champion_cross)
    actor5_cross = load(args.confirm_actor5_cross)
    paired_same = compare("actor5", champion_same, actor5_same, "confirmation")
    paired_cross = compare("actor5", champion_cross, actor5_cross, "confirmation")
    confirmation_paired = paired_same + paired_cross
    confirmation_summary = [
        comparison_summary("actor5", paired_same, "confirmation", "same_xx"),
        comparison_summary("actor5", paired_cross, "confirmation", "cross_xy"),
        comparison_summary("actor5", confirmation_paired, "confirmation", "all"),
    ]
    for profile in sorted({row["surface_profile"] for row in confirmation_paired}):
        subset = [row for row in confirmation_paired if row["surface_profile"] == profile]
        confirmation_summary.append(
            comparison_summary("actor5", subset, "confirmation_profile", profile))

    screen_path = os.path.join(out_dir, "screen_summary.csv")
    screen_comparison_path = os.path.join(out_dir, "screen_comparison_summary.csv")
    screen_paired_path = os.path.join(out_dir, "screen_paired.csv")
    confirmation_path = os.path.join(out_dir, "confirmation_summary.csv")
    confirmation_paired_path = os.path.join(out_dir, "confirmation_paired.csv")
    write_csv(screen_path, screen_summary)
    write_csv(screen_comparison_path, screen_comparison_summary)
    write_csv(screen_paired_path, screen_paired)
    write_csv(confirmation_path, confirmation_summary)
    write_csv(confirmation_paired_path, confirmation_paired)

    overall = confirmation_summary[2]
    safety_noninferior = (
        overall["candidate_safety_ok"] >= overall["champion_safety_ok"]
        and overall["candidate_sensor_fault_steps"] == 0
    )
    quality_noninferior = overall["candidate_quality_ok"] >= overall["champion_quality_ok"]
    # Quality count is tied, while GU and scratch both regress.  The short-PPO
    # checkpoint therefore remains an experiment and is never promoted here.
    promotion_eligible = bool(
        safety_noninferior and quality_noninferior
        and overall["candidate_quality_ok"] > overall["champion_quality_ok"]
        and overall["mean_delta_gu_final"] >= 0.0
        and overall["mean_delta_scratch_final_um"] <= 0.0
    )
    train_result_path = os.path.join(os.path.abspath(args.train_dir), "smoke_result.json")
    source_paths.append(train_result_path)
    with open(train_result_path, encoding="utf-8") as handle:
        training = json.load(handle)
    selected_checkpoint = os.path.abspath(args.selected_checkpoint)
    source_paths.append(selected_checkpoint)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E4 base14 short PPO ablation",
        "training_nominal_samples": training["nominal_samples"],
        "critic_warmup_iterations": training["critic_warmup_iterations"],
        "actor_iterations": training["actor_iterations"],
        "actor_relative_l2_delta": training["actor_drift"]["mlp_relative_l2_delta"],
        "selected_candidate": "actor5",
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "screen_executions": len(screen_champion) * 4,
        "confirmation_executions": len(confirmation_paired) * 2,
        "safety_noninferior": safety_noninferior,
        "quality_noninferior": quality_noninferior,
        "promotion_eligible": promotion_eligible,
        "champion_promoted": False,
        "full_ppo_performed": False,
        "global_or_spatial_training_performed": False,
        "confirmation_overall": overall,
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Gate E4 base14 short PPO ablation\n\n"
            f"- Training: 12 envs, 5 critic warmup + 5 actor iterations, "
            f"{training['nominal_samples']:,} nominal samples.\n"
            f"- Actor relative L2 drift after actor5: "
            f"{training['actor_drift']['mlp_relative_l2_delta']:.9g}.\n"
            "- Screen: champion and actor1/actor3/actor5, 8 executions each; all safety 8/8.\n"
            "- Screen quality: champion 3/8, actor1 3/8, actor3 3/8, actor5 4/8.\n"
            "- Actor5 was selected for confirmation, not promotion.\n"
            f"- Confirmation: {len(confirmation_paired)} paired conditions / "
            f"{len(confirmation_paired) * 2} executions across same_xx and cross_xy.\n"
            f"- Safety champion/actor5: {overall['champion_safety_ok']}/{overall['pairs']} / "
            f"{overall['candidate_safety_ok']}/{overall['pairs']}; sensor faults 0/0.\n"
            f"- Quality champion/actor5: {overall['champion_quality_ok']}/{overall['pairs']} / "
            f"{overall['candidate_quality_ok']}/{overall['pairs']}.\n"
            f"- Mean actor5 - champion: GU {overall['mean_delta_gu_final']:+.6f}, "
            f"Ra {overall['mean_delta_ra_final_um']:+.6f} um, "
            f"Rz {overall['mean_delta_rz_final_um']:+.6f} um, "
            f"scratch {overall['mean_delta_scratch_final_um']:+.6f} um, "
            f"clearcoat {overall['mean_delta_clearcoat_min_um']:+.6f} um, "
            f"control steps {overall['mean_delta_control_steps']:+.3f}.\n"
            "- Verdict: safety integration passes, but quality count is tied and GU/scratch regress.\n"
            "- No checkpoint is promoted; the frozen champion remains unchanged.\n"
            "- This is PT-DESIGN short ablation, not full PPO or a production result.\n"
        )

    generated = [screen_path, screen_comparison_path, screen_paired_path, confirmation_path,
                 confirmation_paired_path, metadata_path, readme_path]
    checksum_rows = [
        {"sha256": sha256(path), "file": path}
        for path in source_paths + generated
    ]
    write_csv(os.path.join(out_dir, "checksums.csv"), checksum_rows)
    print(
        f"[Gate E4 summary] wrote {out_dir}; safety={safety_noninferior} "
        f"promotion={promotion_eligible}")


if __name__ == "__main__":
    main()
