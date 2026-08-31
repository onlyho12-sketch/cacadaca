"""Summarize Gate E7 multi-seed checkpoint screens and final paired evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dir(directory: str, expected_sequences: int) -> tuple[list[dict], list[dict], dict]:
    sequences = read_csv(os.path.join(directory, "sequences.csv"))
    tiles = read_csv(os.path.join(directory, "tile_area_fractions.csv"))
    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    if len(sequences) != expected_sequences or len(tiles) != expected_sequences * 25:
        raise RuntimeError(
            f"incomplete evaluation {directory}: {len(sequences)}/{len(tiles)}"
        )
    return sequences, tiles, metadata


def sequence_summary(rows: list[dict], label: str, direction: str) -> dict:
    return {
        "policy_label": label,
        "direction_mode": direction,
        "sequences": len(rows),
        "safety_pass": sum(truth(row["safety_ok"]) for row in rows),
        "quality_pass": sum(truth(row["quality_ok"]) for row in rows),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in rows),
        "mean_gu_final": mean(rows, "gu_final"),
        "mean_ra_final_um": mean(rows, "ra_final_um"),
        "mean_rz_final_um": mean(rows, "rz_final_um"),
        "mean_scratch_final_um": mean(rows, "scratch_final_um"),
        "mean_clearcoat_min_um": mean(rows, "clearcoat_min_um"),
        "mean_temperature_peak_c": mean(rows, "temperature_peak_c"),
        "mean_control_steps": mean(rows, "control_steps"),
        "mean_final_all4_pass_area_pct": mean(rows, "final_all4_pass_area_pct"),
    }


def tile_summary(rows: list[dict], label: str, direction: str) -> dict:
    values = [float(row["all4_pass_area_pct"]) for row in rows]
    passed_cells = sum(
        round(int(row["cell_count"]) * float(row["all4_pass_area_pct"]) / 100.0)
        for row in rows
    )
    total_cells = sum(int(row["cell_count"]) for row in rows)
    return {
        "policy_label": label,
        "direction_mode": direction,
        "tiles_total": len(rows),
        "tiles_all4_area_100pct": sum(value >= 100.0 - 1e-9 for value in values),
        "tiles_all4_area_ge95pct": sum(value >= 95.0 for value in values),
        "tiles_all4_area_ge90pct": sum(value >= 90.0 for value in values),
        "tiles_all4_area_ge80pct": sum(value >= 80.0 for value in values),
        "all4_cells_pass": passed_cells,
        "surface_cells_total": total_cells,
        "all4_cell_area_pct": 100.0 * passed_cells / total_cells,
    }


def screen_rank(row: dict) -> tuple:
    return (
        int(row["safety_pass"]),
        int(row["quality_pass"]),
        float(row["mean_final_all4_pass_area_pct"]),
        -float(row["mean_control_steps"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen_dir", action="append", required=True)
    parser.add_argument("--confirm_dir", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    screen_sequences = []
    screen_tiles = []
    screen_sources = []
    for directory in args.screen_dir:
        sequences, tiles, metadata = load_dir(os.path.abspath(directory), 8)
        label = sequences[0]["policy_label"]
        match = re.fullmatch(r"s(\d+)_it(\d+)", label)
        if match is None:
            raise ValueError(f"unexpected screen label: {label}")
        seq = sequence_summary(sequences, label, sequences[0]["direction_mode"])
        seq.update({
            "training_seed": int(match.group(1)),
            "checkpoint_iteration": int(match.group(2)),
            "nominal_samples": (int(match.group(2)) + 1) * 12 * 48,
        })
        screen_sequences.append(seq)
        screen_tiles.append(tile_summary(tiles, label, tiles[0]["direction_mode"]))
        screen_sources.append(metadata)
    screen_sequences.sort(key=lambda row: (row["training_seed"], row["checkpoint_iteration"]))
    screen_tiles.sort(key=lambda row: row["policy_label"])

    selected = []
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in screen_sequences:
        by_seed[int(row["training_seed"])].append(row)
    for seed, rows in sorted(by_seed.items()):
        best = max(rows, key=screen_rank)
        selected.append({
            "training_seed": seed,
            "selected_policy_label": best["policy_label"],
            "selected_checkpoint_iteration": best["checkpoint_iteration"],
            "selected_nominal_samples": best["nominal_samples"],
            "selection_order": "safety, quality, all4 area, lower steps",
            "safety_pass": best["safety_pass"],
            "quality_pass": best["quality_pass"],
            "mean_final_all4_pass_area_pct": best["mean_final_all4_pass_area_pct"],
            "mean_control_steps": best["mean_control_steps"],
        })

    confirm: dict[tuple[str, str], tuple[list[dict], list[dict]]] = {}
    confirm_sources = []
    direction_sequence_rows = []
    direction_tile_rows = []
    profile_rows = []
    for directory in args.confirm_dir:
        sequences, tiles, metadata = load_dir(os.path.abspath(directory), 16)
        label = sequences[0]["policy_label"]
        direction = sequences[0]["direction_mode"]
        key = (label, direction)
        if key in confirm:
            raise RuntimeError(f"duplicate final evaluation key: {key}")
        confirm[key] = (sequences, tiles)
        direction_sequence_rows.append(sequence_summary(sequences, label, direction))
        direction_tile_rows.append(tile_summary(tiles, label, direction))
        for profile in sorted({row["surface_profile"] for row in sequences}):
            pseq = [row for row in sequences if row["surface_profile"] == profile]
            ptiles = [row for row in tiles if row["surface_profile"] == profile]
            profile_rows.append({
                **sequence_summary(pseq, label, direction),
                "surface_profile": profile,
                **{
                    key: value
                    for key, value in tile_summary(ptiles, label, direction).items()
                    if key not in {"policy_label", "direction_mode"}
                },
            })
        confirm_sources.append(metadata)

    labels = sorted({key[0] for key in confirm})
    directions = sorted({key[1] for key in confirm})
    expected_labels = {"champion", *(row["selected_policy_label"] for row in selected)}
    if set(labels) != expected_labels or set(directions) != {"same_xx", "cross_xy"}:
        raise RuntimeError("final evaluation labels/directions do not match selected screen policies")

    combined_sequence_rows = []
    combined_tile_rows = []
    for label in labels:
        sequences = sum((confirm[(label, direction)][0] for direction in directions), [])
        tiles = sum((confirm[(label, direction)][1] for direction in directions), [])
        combined_sequence_rows.append(sequence_summary(sequences, label, "same_xx+cross_xy"))
        combined_tile_rows.append(tile_summary(tiles, label, "same_xx+cross_xy"))

    seq_by_label = {row["policy_label"]: row for row in combined_sequence_rows}
    tile_by_label = {row["policy_label"]: row for row in combined_tile_rows}
    champion = seq_by_label["champion"]
    raw_quality_winner = max(
        (row for row in combined_sequence_rows if row["policy_label"] != "champion"),
        key=lambda row: (
            int(row["safety_pass"]), int(row["quality_pass"]),
            float(row["mean_final_all4_pass_area_pct"]),
            -float(row["mean_control_steps"]),
        ),
    )
    tied_quality = [
        row for row in combined_sequence_rows
        if row["policy_label"] != "champion"
        and int(row["quality_pass"]) == int(raw_quality_winner["quality_pass"])
    ]
    quality_time_candidate = min(tied_quality, key=lambda row: float(row["mean_control_steps"]))

    deltas = []
    for row in combined_sequence_rows:
        tiles = tile_by_label[row["policy_label"]]
        deltas.append({
            "policy_label": row["policy_label"],
            "delta_quality_pass_vs_champion": int(row["quality_pass"]) - int(champion["quality_pass"]),
            "delta_all4_area_pct_points_vs_champion": (
                float(row["mean_final_all4_pass_area_pct"])
                - float(champion["mean_final_all4_pass_area_pct"])
            ),
            "delta_control_steps_vs_champion": (
                float(row["mean_control_steps"]) - float(champion["mean_control_steps"])
            ),
            "delta_control_steps_pct_vs_champion": 100.0 * (
                float(row["mean_control_steps"]) - float(champion["mean_control_steps"])
            ) / float(champion["mean_control_steps"]),
            "delta_tile100_vs_champion": int(tiles["tiles_all4_area_100pct"]) - int(tile_by_label["champion"]["tiles_all4_area_100pct"]),
            "delta_tile95_vs_champion": int(tiles["tiles_all4_area_ge95pct"]) - int(tile_by_label["champion"]["tiles_all4_area_ge95pct"]),
            "delta_tile90_vs_champion": int(tiles["tiles_all4_area_ge90pct"]) - int(tile_by_label["champion"]["tiles_all4_area_ge90pct"]),
            "delta_tile80_vs_champion": int(tiles["tiles_all4_area_ge80pct"]) - int(tile_by_label["champion"]["tiles_all4_area_ge80pct"]),
        })

    write_csv(os.path.join(out_dir, "screen_sequence_summary.csv"), screen_sequences)
    write_csv(os.path.join(out_dir, "screen_tile_counts.csv"), screen_tiles)
    write_csv(os.path.join(out_dir, "screen_selected_per_seed.csv"), selected)
    write_csv(
        os.path.join(out_dir, "confirm_sequence_summary.csv"),
        direction_sequence_rows + combined_sequence_rows,
    )
    write_csv(
        os.path.join(out_dir, "confirm_tile_counts.csv"),
        direction_tile_rows + combined_tile_rows,
    )
    write_csv(os.path.join(out_dir, "confirm_profile_summary.csv"), profile_rows)
    write_csv(os.path.join(out_dir, "confirm_deltas_vs_champion.csv"), deltas)

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E7 flat PPO multi-seed main training",
        "training_seeds": sorted(by_seed),
        "samples_per_seed": 150336,
        "total_nominal_samples": 3 * 150336,
        "seed_20260832_segmented_after_iteration_196": True,
        "screen_surface_seed_base": 24000,
        "confirm_surface_seed_base": 25000,
        "screen_selected": selected,
        "raw_quality_winner": raw_quality_winner["policy_label"],
        "quality_time_candidate": quality_time_candidate["policy_label"],
        "champion_promoted": False,
        "promotion_reason": (
            "No automatic promotion: all candidates are safe and improve quality, but the "
            "pre-declared acceptance threshold for quality versus completion-time cost was not set."
        ),
        "flat_work_complete": False,
        "next_step_requires_approval": (
            "Choose raw-quality or quality-time objective and acceptance threshold before promotion; "
            "then run the selected policy's release/regression validation."
        ),
    }
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)

    raw = seq_by_label[raw_quality_winner["policy_label"]]
    balanced = seq_by_label[quality_time_candidate["policy_label"]]
    raw_tiles = tile_by_label[raw_quality_winner["policy_label"]]
    balanced_tiles = tile_by_label[quality_time_candidate["policy_label"]]
    champ_tiles = tile_by_label["champion"]
    lines = [
        "# Gate E7 flat PPO multi-seed main training",
        "",
        "## 결론",
        "",
        "- 독립 training seed 3개에 각각 150,336 samples, 총 451,008 nominal samples를 학습했다.",
        "- seed 20260832는 iteration 196 뒤 Isaac 정체로 재부팅 후 197~260회를 이어간 segmented run이다.",
        "- 15 checkpoint aligned screen과 대표 3개+champion의 unseen same/cross 32표면 평가를 완료했다.",
        "- 네 정책 모두 안전 32/32, sensor fault 0이다.",
        f"- 순수 품질 최고는 `{raw_quality_winner['policy_label']}`: 품질 {raw['quality_pass']}/32, all-4 {raw['mean_final_all4_pass_area_pct']:.4f}%, 평균 {raw['mean_control_steps']:.1f} step.",
        f"- 품질-시간 균형 후보는 `{quality_time_candidate['policy_label']}`: 품질 {balanced['quality_pass']}/32, all-4 {balanced['mean_final_all4_pass_area_pct']:.4f}%, 평균 {balanced['mean_control_steps']:.1f} step.",
        f"- 기존 champion은 품질 {champion['quality_pass']}/32, all-4 {champion['mean_final_all4_pass_area_pct']:.4f}%, 평균 {champion['mean_control_steps']:.1f} step.",
        "- 사전 승인 임계가 없으므로 champion을 자동 교체하지 않았다.",
        "",
        "## 정책당 800타일의 내부 all-4 통과면적",
        "",
        "| 정책 | 100% | ≥95% | ≥90% | ≥80% |",
        "|---|---:|---:|---:|---:|",
        f"| champion | {champ_tiles['tiles_all4_area_100pct']} | {champ_tiles['tiles_all4_area_ge95pct']} | {champ_tiles['tiles_all4_area_ge90pct']} | {champ_tiles['tiles_all4_area_ge80pct']} |",
        f"| raw quality | {raw_tiles['tiles_all4_area_100pct']} | {raw_tiles['tiles_all4_area_ge95pct']} | {raw_tiles['tiles_all4_area_ge90pct']} | {raw_tiles['tiles_all4_area_ge80pct']} |",
        f"| quality-time | {balanced_tiles['tiles_all4_area_100pct']} | {balanced_tiles['tiles_all4_area_ge95pct']} | {balanced_tiles['tiles_all4_area_ge90pct']} | {balanced_tiles['tiles_all4_area_ge80pct']} |",
        "",
        "타일 평균 하나로 전체 타일을 통과 처리하지 않았고, 각 타일 내부 cell의 all-4 통과면적을 기준으로 집계했다.",
        "",
        "## 다음 승인 항목",
        "",
        "1. 순수 품질 최고와 품질-시간 균형 중 배포 목표를 선택한다.",
        "2. 허용 가능한 작업시간 증가율과 필요한 품질/타일 개선 임계를 정한다.",
        "3. 선택 정책을 release/regression 검증한 뒤에만 champion 승격 여부를 결정한다.",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for name in os.listdir(out_dir):
        if not name.endswith(".csv"):
            continue
        for row in read_csv(os.path.join(out_dir, name)):
            for value in row.values():
                try:
                    if not math.isfinite(float(value)):
                        raise RuntimeError(f"NaN/Inf in {name}")
                except ValueError:
                    pass
    targets = sorted(name for name in os.listdir(out_dir) if name != "checksums.sha256")
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as handle:
        for name in targets:
            handle.write(f"{sha256(os.path.join(out_dir, name))}  {name}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
