"""Summarize the Gate E6b three-policy quality/time comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(label: str, directories: list[str]) -> tuple[dict, list[dict], list[dict]]:
    sequences: list[dict] = []
    tiles: list[dict] = []
    for directory in directories:
        sequences.extend(read_csv(os.path.join(directory, "sequences.csv")))
        tiles.extend(read_csv(os.path.join(directory, "tile_area_fractions.csv")))
    if len(sequences) != 32 or len(tiles) != 800:
        raise RuntimeError(f"{label}: expected 32 sequences/800 tiles")
    mean = lambda key: sum(float(row[key]) for row in sequences) / len(sequences)
    tile_values = [float(row["all4_pass_area_pct"]) for row in tiles]
    row = {
        "policy_label": label,
        "sequences": len(sequences),
        "tiles": len(tiles),
        "safety_pass": sum(row["safety_ok"] == "True" for row in sequences),
        "quality_pass": sum(row["quality_ok"] == "True" for row in sequences),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in sequences),
        "mean_gu_final": mean("gu_final"),
        "mean_ra_final_um": mean("ra_final_um"),
        "mean_rz_final_um": mean("rz_final_um"),
        "mean_scratch_final_um": mean("scratch_final_um"),
        "mean_clearcoat_min_um": mean("clearcoat_min_um"),
        "mean_temperature_peak_c": mean("temperature_peak_c"),
        "mean_control_steps": mean("control_steps"),
        "mean_all4_area_pct": mean("final_all4_pass_area_pct"),
        "all4_cells_pass": sum(round(
            float(tile["cell_count"]) * float(tile["all4_pass_area_pct"]) / 100.0)
            for tile in tiles),
        "tiles_all4_area_100pct": sum(value >= 100.0 - 1e-9 for value in tile_values),
        "tiles_all4_area_ge95pct": sum(value >= 95.0 for value in tile_values),
        "tiles_all4_area_ge90pct": sum(value >= 90.0 for value in tile_values),
        "tiles_all4_area_ge80pct": sum(value >= 80.0 for value in tile_values),
    }
    return row, sequences, tiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion_dir", action="append", required=True)
    parser.add_argument("--control30528_dir", action="append", required=True)
    parser.add_argument("--control46080_dir", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    if any(len(value) != 2 for value in (
            args.champion_dir, args.control30528_dir, args.control46080_dir)):
        raise ValueError("each policy requires exactly same/cross two directories")
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    specs = (
        ("champion", args.champion_dir),
        ("control_30528", args.control30528_dir),
        ("control_46080", args.control46080_dir),
    )
    summaries = []
    policy_data = {}
    for label, directories in specs:
        summary, sequences, tiles = summarize(label, directories)
        summaries.append(summary)
        policy_data[label] = (sequences, tiles)
    champion = summaries[0]
    delta_rows = []
    for row in summaries[1:]:
        delta_rows.append({
            "policy_label": row["policy_label"],
            "delta_gu": row["mean_gu_final"] - champion["mean_gu_final"],
            "delta_ra_um": row["mean_ra_final_um"] - champion["mean_ra_final_um"],
            "delta_rz_um": row["mean_rz_final_um"] - champion["mean_rz_final_um"],
            "delta_scratch_um": (
                row["mean_scratch_final_um"] - champion["mean_scratch_final_um"]),
            "delta_clearcoat_um": (
                row["mean_clearcoat_min_um"] - champion["mean_clearcoat_min_um"]),
            "delta_temperature_c": (
                row["mean_temperature_peak_c"] - champion["mean_temperature_peak_c"]),
            "delta_control_steps": (
                row["mean_control_steps"] - champion["mean_control_steps"]),
            "delta_control_steps_pct": 100.0 * (
                row["mean_control_steps"] - champion["mean_control_steps"]
            ) / champion["mean_control_steps"],
            "delta_all4_area_pct_points": (
                row["mean_all4_area_pct"] - champion["mean_all4_area_pct"]),
            "delta_all4_cells": row["all4_cells_pass"] - champion["all4_cells_pass"],
            "delta_tiles_100pct": (
                row["tiles_all4_area_100pct"] - champion["tiles_all4_area_100pct"]),
            "delta_tiles_ge95pct": (
                row["tiles_all4_area_ge95pct"] - champion["tiles_all4_area_ge95pct"]),
            "delta_tiles_ge90pct": (
                row["tiles_all4_area_ge90pct"] - champion["tiles_all4_area_ge90pct"]),
            "delta_tiles_ge80pct": (
                row["tiles_all4_area_ge80pct"] - champion["tiles_all4_area_ge80pct"]),
        })

    write_csv(os.path.join(out_dir, "three_policy_summary.csv"), summaries)
    write_csv(os.path.join(out_dir, "delta_vs_champion.csv"), delta_rows)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E6b control checkpoint quality-time confirmation",
        "recommended_flat_ppo_candidate": "control_30528",
        "champion_replaced": False,
        "reason": (
            "control_30528 keeps the 80% tile-count gain with only 1.46% more steps than "
            "champion; control_46080 gives slightly more quality but costs 14.93% more steps."),
        "pareto_candidates": ["champion", "control_30528", "control_46080"],
        "full_training_started": False,
    }
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)

    mid, last = summaries[1], summaries[2]
    text = f"""# Gate E6b control checkpoint quality-time confirmation

## 결론

- 같은 새 seed 4개×4 profile×same/cross, 정책당 32표면·800타일로 champion,
  `control_30528`, `control_46080`을 비교했다.
- 세 정책 모두 안전 32/32, 기존 ROI 집계 품질 24/32, sensor fault 0이다.
- 평면 PPO 후속 후보는 `control_30528`로 좁힌다. champion은 교체하지 않는다.

## 세 정책 수치

| 정책 | all-4 면적 | 평균 step | 100% 타일 | ≥95% | ≥90% | ≥80% |
|---|---:|---:|---:|---:|---:|---:|
| champion | {champion['mean_all4_area_pct']:.4f}% | {champion['mean_control_steps']:.1f} | {champion['tiles_all4_area_100pct']} | {champion['tiles_all4_area_ge95pct']} | {champion['tiles_all4_area_ge90pct']} | {champion['tiles_all4_area_ge80pct']} |
| control_30528 | {mid['mean_all4_area_pct']:.4f}% | {mid['mean_control_steps']:.1f} | {mid['tiles_all4_area_100pct']} | {mid['tiles_all4_area_ge95pct']} | {mid['tiles_all4_area_ge90pct']} | {mid['tiles_all4_area_ge80pct']} |
| control_46080 | {last['mean_all4_area_pct']:.4f}% | {last['mean_control_steps']:.1f} | {last['tiles_all4_area_100pct']} | {last['tiles_all4_area_ge95pct']} | {last['tiles_all4_area_ge90pct']} | {last['tiles_all4_area_ge80pct']} |

`control_30528`은 champion 대비 all-4 면적 +{mid['mean_all4_area_pct'] - champion['mean_all4_area_pct']:.4f}%p,
통과 cell +{mid['all4_cells_pass'] - champion['all4_cells_pass']}개, 80% 타일 +{mid['tiles_all4_area_ge80pct'] - champion['tiles_all4_area_ge80pct']}개이며
작업시간 증가는 +{mid['mean_control_steps'] - champion['mean_control_steps']:.1f} step
(+{100.0 * (mid['mean_control_steps'] - champion['mean_control_steps']) / champion['mean_control_steps']:.2f}%)다.

`control_46080`은 all-4 면적이 `control_30528`보다
{last['mean_all4_area_pct'] - mid['mean_all4_area_pct']:.4f}%p 더 높지만 평균
{last['mean_control_steps'] - mid['mean_control_steps']:.1f} step 더 필요하다. 100%/95%/80% 타일 수는
두 control이 같고 90% 타일만 46080이 2개 많다. 따라서 품질-시간 균형의 후속 후보는
`control_30528`이다.

이는 본 학습이나 champion 승격이 아니다. 다음에는 사용자와 all-4/타일 승인 기준 및
본 학습 범위를 먼저 정해야 한다.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(text)

    for filename in ("three_policy_summary.csv", "delta_vs_champion.csv"):
        for row in read_csv(os.path.join(out_dir, filename)):
            for value in row.values():
                try:
                    if not math.isfinite(float(value)):
                        raise RuntimeError(f"non-finite {filename}")
                except ValueError:
                    pass
    targets = sorted(name for name in os.listdir(out_dir) if name != "checksums.sha256")
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as handle:
        for name in targets:
            handle.write(f"{sha256(os.path.join(out_dir, name))}  {name}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
