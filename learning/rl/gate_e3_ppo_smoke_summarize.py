"""Summarize paired one-pass champion vs Gate E3 PPO-smoke evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone


PAIR_KEYS = ("direction_mode", "surface_profile", "profile_seed")
METRICS = (
    "gu_final", "ra_final_um", "rz_final_um", "scratch_final_um",
    "clearcoat_min_um", "temperature_peak_c",
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_rows(champion: list[dict], smoke: list[dict]) -> list[dict]:
    def indexed(rows):
        out = {}
        for row in rows:
            key = tuple(row[name] for name in PAIR_KEYS)
            if key in out:
                raise ValueError(f"duplicate pair key {key}")
            out[key] = row
        return out

    left, right = indexed(champion), indexed(smoke)
    if set(left) != set(right):
        raise ValueError("champion and smoke evaluation keys differ")
    paired = []
    for key in sorted(left):
        a, b = left[key], right[key]
        row = dict(zip(PAIR_KEYS, key))
        row.update({
            "champion_outcome": a["outcome"],
            "smoke_outcome": b["outcome"],
            "champion_quality_ok": int(a["quality_ok"] == "True"),
            "smoke_quality_ok": int(b["quality_ok"] == "True"),
            "champion_safety_ok": int(a["safety_ok"] == "True"),
            "smoke_safety_ok": int(b["safety_ok"] == "True"),
            "champion_sensor_fault_steps": int(a["sensor_fault_steps"]),
            "smoke_sensor_fault_steps": int(b["sensor_fault_steps"]),
        })
        for metric in METRICS:
            row[f"champion_{metric}"] = float(a[metric])
            row[f"smoke_{metric}"] = float(b[metric])
            row[f"delta_{metric}"] = float(b[metric]) - float(a[metric])
        paired.append(row)
    return paired


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion_dir", required=True)
    parser.add_argument("--smoke_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    champion_csv = os.path.join(os.path.abspath(args.champion_dir), "sequences.csv")
    smoke_csv = os.path.join(os.path.abspath(args.smoke_dir), "sequences.csv")
    train_result = os.path.join(os.path.abspath(args.train_dir), "smoke_result.json")
    champion = read_rows(champion_csv)
    smoke = read_rows(smoke_csv)
    paired = pair_rows(champion, smoke)
    if not paired:
        raise RuntimeError("paired evaluation is empty")
    paired_path = os.path.join(out_dir, "paired_vs_champion.csv")
    write_csv(paired_path, paired)

    def count(name: str, value: int) -> int:
        return sum(int(row[name]) == value for row in paired)

    with open(train_result, encoding="utf-8") as handle:
        training = json.load(handle)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E3 base14 PPO integration smoke",
        "pairs": len(paired),
        "champion_quality_ok": count("champion_quality_ok", 1),
        "smoke_quality_ok": count("smoke_quality_ok", 1),
        "champion_safety_ok": count("champion_safety_ok", 1),
        "smoke_safety_ok": count("smoke_safety_ok", 1),
        "champion_sensor_fault_steps": sum(row["champion_sensor_fault_steps"] for row in paired),
        "smoke_sensor_fault_steps": sum(row["smoke_sensor_fault_steps"] for row in paired),
        "mean_delta": {
            metric: sum(row[f"delta_{metric}"] for row in paired) / len(paired)
            for metric in METRICS
        },
        "training_integration_smoke_pass": training["integration_smoke_pass"],
        "training_nominal_samples": training["nominal_samples"],
        "actor_relative_l2_delta": training["actor_drift"]["mlp_relative_l2_delta"],
        "safety_noninferior": (
            count("smoke_safety_ok", 1) >= count("champion_safety_ok", 1)
            and sum(row["smoke_sensor_fault_steps"] for row in paired) == 0
        ),
        "quality_noninferior_count": count("smoke_quality_ok", 1) >= count("champion_quality_ok", 1),
        "champion_promoted": False,
        "full_ppo_performed": False,
    }
    summary["gate_pass"] = bool(
        summary["training_integration_smoke_pass"] and summary["safety_noninferior"])
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    readme_path = os.path.join(out_dir, "README.md")
    delta = summary["mean_delta"]
    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Gate E3 base14 PPO integration smoke\n\n"
            f"- PPO integration gate: **{'PASS' if summary['gate_pass'] else 'FAIL'}**\n"
            f"- Scope: {training['num_envs']} envs x 48 steps x "
            f"{training['total_iterations']} iterations = "
            f"{training['nominal_samples']:,} nominal samples\n"
            f"- Critic warmup / actor update: "
            f"{training['critic_warmup_iterations']} / {training['actor_iterations']} iterations\n"
            f"- Actor relative L2 drift: {summary['actor_relative_l2_delta']:.9g}\n"
            f"- Paired unseen one-pass executions: {summary['pairs']}\n"
            f"- Safety champion/smoke: {summary['champion_safety_ok']}/{summary['pairs']} / "
            f"{summary['smoke_safety_ok']}/{summary['pairs']}\n"
            f"- Quality champion/smoke: {summary['champion_quality_ok']}/{summary['pairs']} / "
            f"{summary['smoke_quality_ok']}/{summary['pairs']}\n"
            f"- Mean smoke - champion: GU {delta['gu_final']:+.6f}, "
            f"Ra {delta['ra_final_um']:+.6f} um, "
            f"scratch {delta['scratch_final_um']:+.6f} um\n"
            "- This is a PT-DESIGN integration smoke, not a performance result or full PPO run.\n"
            "- No checkpoint was promoted; the frozen champion remains unchanged.\n"
        )

    checksum_rows = []
    for path in (champion_csv, smoke_csv, train_result, paired_path, metadata_path, readme_path):
        checksum_rows.append({"sha256": sha256(path), "file": path})
    write_csv(os.path.join(out_dir, "checksums.csv"), checksum_rows)
    print(f"[Gate E3 summary] wrote {out_dir}; gate_pass={summary['gate_pass']}")


if __name__ == "__main__":
    main()
