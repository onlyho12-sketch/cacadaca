"""Summarize Gate E6 equal-sample screens and paired area confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone


AREA_KEYS = (
    "gu_pass", "ra_pass", "rz_pass", "scratch_pass", "all4_pass",
    "clearcoat_pass", "temperature_pass",
)
SEQUENCE_METRICS = (
    "gu_final", "ra_final_um", "rz_final_um", "scratch_final_um",
    "clearcoat_min_um", "temperature_peak_c", "control_steps",
    "final_gu_pass_area_pct", "final_ra_pass_area_pct",
    "final_rz_pass_area_pct", "final_scratch_pass_area_pct",
    "final_all4_pass_area_pct",
)


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output rows for {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _sequence_summary(rows: list[dict], stage: str) -> dict:
    first = rows[0]
    out = {
        "stage": stage,
        "policy_label": first["policy_label"],
        "direction_mode": first["direction_mode"],
        "sequences": len(rows),
        "safety_pass": sum(_truth(row["safety_ok"]) for row in rows),
        "quality_pass": sum(_truth(row["quality_ok"]) for row in rows),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in rows),
    }
    for key in SEQUENCE_METRICS:
        out[f"mean_{key}"] = _mean(rows, key)
    for key in AREA_KEYS:
        out[f"mean_tile_{key}_area_pct_min"] = _mean(
            rows, f"final_tile_{key}_area_pct_min")
        out[f"mean_tile_{key}_area_pct_p10"] = _mean(
            rows, f"final_tile_{key}_area_pct_p10")
    return out


def _tile_summary(rows: list[dict], stage: str) -> dict:
    first = rows[0]
    values = [float(row["all4_pass_area_pct"]) for row in rows]
    exact_cells = sum(
        int(round(float(row["cell_count"]) * float(row["all4_pass_area_pct"]) / 100.0))
        for row in rows)
    total_cells = sum(int(row["cell_count"]) for row in rows)
    return {
        "stage": stage,
        "policy_label": first["policy_label"],
        "direction_mode": first["direction_mode"],
        "tiles_total": len(rows),
        "tiles_all4_area_100pct": sum(value >= 100.0 - 1e-9 for value in values),
        "tiles_all4_area_ge95pct": sum(value >= 95.0 for value in values),
        "tiles_all4_area_ge90pct": sum(value >= 90.0 for value in values),
        "all4_cells_pass": exact_cells,
        "surface_cells_total": total_cells,
        "all4_cell_area_pct": 100.0 * exact_cells / total_cells,
    }


def _load_dir(directory: str) -> tuple[list[dict], list[dict], dict]:
    sequences = _read(os.path.join(directory, "sequences.csv"))
    tiles = _read(os.path.join(directory, "tile_area_fractions.csv"))
    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = int(metadata["expected"])
    if len(sequences) != expected or len(tiles) != expected * 25:
        raise RuntimeError(f"incomplete Gate E6 directory: {directory}")
    return sequences, tiles, metadata


def _key(row: dict) -> tuple:
    return row["direction_mode"], row["surface_profile"], int(row["profile_seed"])


def _tile_key(row: dict) -> tuple:
    return (*_key(row), int(row["tile_i"]), int(row["tile_j"]))


def _paired(champion: list[dict], trained: list[dict]) -> list[dict]:
    left = {_key(row): row for row in champion}
    right = {_key(row): row for row in trained}
    if left.keys() != right.keys():
        raise RuntimeError("confirmation sequence paired keys differ")
    rows = []
    for key in sorted(left):
        c, p = left[key], right[key]
        row = {
            "direction_mode": key[0], "surface_profile": key[1],
            "profile_seed": key[2],
            "champion_safety_ok": int(_truth(c["safety_ok"])),
            "trained_safety_ok": int(_truth(p["safety_ok"])),
            "champion_quality_ok": int(_truth(c["quality_ok"])),
            "trained_quality_ok": int(_truth(p["quality_ok"])),
        }
        for metric in SEQUENCE_METRICS:
            row[f"champion_{metric}"] = float(c[metric])
            row[f"trained_{metric}"] = float(p[metric])
            row[f"delta_{metric}"] = float(p[metric]) - float(c[metric])
        rows.append(row)
    return rows


def _paired_tiles(champion: list[dict], trained: list[dict]) -> list[dict]:
    left = {_tile_key(row): row for row in champion}
    right = {_tile_key(row): row for row in trained}
    if left.keys() != right.keys():
        raise RuntimeError("confirmation tile paired keys differ")
    rows = []
    for key in sorted(left):
        c, p = left[key], right[key]
        c_area = float(c["all4_pass_area_pct"])
        p_area = float(p["all4_pass_area_pct"])
        rows.append({
            "direction_mode": key[0], "surface_profile": key[1],
            "profile_seed": key[2], "tile_i": key[3], "tile_j": key[4],
            "champion_all4_area_pct": c_area,
            "trained_all4_area_pct": p_area,
            "delta_all4_area_pct": p_area - c_area,
            "champion_strict100": int(c_area >= 100.0 - 1e-9),
            "trained_strict100": int(p_area >= 100.0 - 1e-9),
            "champion_ge95": int(c_area >= 95.0),
            "trained_ge95": int(p_area >= 95.0),
            "champion_ge90": int(c_area >= 90.0),
            "trained_ge90": int(p_area >= 90.0),
        })
    return rows


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    screen_sequence_summary = []
    screen_tile_summary = []
    source_metadata = []
    for directory in args.screen_dir:
        sequences, tiles, metadata = _load_dir(os.path.abspath(directory))
        screen_sequence_summary.append(_sequence_summary(sequences, "screen"))
        screen_tile_summary.append(_tile_summary(tiles, "screen"))
        source_metadata.append(metadata)

    confirm_sequences: dict[tuple[str, str], list[dict]] = {}
    confirm_tiles: dict[tuple[str, str], list[dict]] = {}
    confirm_sequence_summary = []
    confirm_tile_summary = []
    profile_summary = []
    for directory in args.confirm_dir:
        sequences, tiles, metadata = _load_dir(os.path.abspath(directory))
        label = sequences[0]["policy_label"]
        direction = sequences[0]["direction_mode"]
        key = (label, direction)
        if key in confirm_sequences:
            raise RuntimeError(f"duplicate confirmation key {key}")
        confirm_sequences[key] = sequences
        confirm_tiles[key] = tiles
        confirm_sequence_summary.append(_sequence_summary(sequences, "confirm"))
        confirm_tile_summary.append(_tile_summary(tiles, "confirm"))
        for profile in sorted({row["surface_profile"] for row in sequences}):
            seq_group = [row for row in sequences if row["surface_profile"] == profile]
            tile_group = [row for row in tiles if row["surface_profile"] == profile]
            profile_summary.append({
                **_sequence_summary(seq_group, "confirm"),
                **{key: value for key, value in _tile_summary(tile_group, "confirm").items()
                   if key not in {"stage", "policy_label", "direction_mode"}},
                "surface_profile": profile,
            })

    policies = sorted({key[0] for key in confirm_sequences})
    directions = sorted({key[1] for key in confirm_sequences})
    if len(policies) != 2 or "champion" not in policies or len(directions) != 2:
        raise RuntimeError("confirmation must contain champion/trained across two directions")
    trained_label = next(label for label in policies if label != "champion")
    paired_rows: list[dict] = []
    paired_tile_rows: list[dict] = []
    for direction in directions:
        paired_rows.extend(_paired(
            confirm_sequences[("champion", direction)],
            confirm_sequences[(trained_label, direction)]))
        paired_tile_rows.extend(_paired_tiles(
            confirm_tiles[("champion", direction)],
            confirm_tiles[(trained_label, direction)]))

    combined_sequence_summary = []
    combined_tile_summary = []
    for policy in policies:
        seq = sum((confirm_sequences[(policy, direction)] for direction in directions), [])
        tiles = sum((confirm_tiles[(policy, direction)] for direction in directions), [])
        seq_summary = _sequence_summary(seq, "confirm_combined")
        seq_summary["direction_mode"] = "same_xx+cross_xy"
        tile_summary = _tile_summary(tiles, "confirm_combined")
        tile_summary["direction_mode"] = "same_xx+cross_xy"
        combined_sequence_summary.append(seq_summary)
        combined_tile_summary.append(tile_summary)

    _write(os.path.join(out_dir, "screen_sequence_summary.csv"), screen_sequence_summary)
    _write(os.path.join(out_dir, "screen_tile_counts.csv"), screen_tile_summary)
    _write(os.path.join(out_dir, "confirm_sequence_summary.csv"),
           confirm_sequence_summary + combined_sequence_summary)
    _write(os.path.join(out_dir, "confirm_tile_counts.csv"),
           confirm_tile_summary + combined_tile_summary)
    _write(os.path.join(out_dir, "confirm_profile_summary.csv"), profile_summary)
    _write(os.path.join(out_dir, "paired_sequences.csv"), paired_rows)
    _write(os.path.join(out_dir, "paired_tiles.csv"), paired_tile_rows)

    combined_seq = {row["policy_label"]: row for row in combined_sequence_summary}
    combined_tiles = {row["policy_label"]: row for row in combined_tile_summary}
    champ = combined_seq["champion"]
    trained = combined_seq[trained_label]
    champ_tiles = combined_tiles["champion"]
    trained_tiles = combined_tiles[trained_label]
    delta_area = trained["mean_final_all4_pass_area_pct"] - champ[
        "mean_final_all4_pass_area_pct"]
    delta_steps = trained["mean_control_steps"] - champ["mean_control_steps"]
    delta_steps_pct = 100.0 * delta_steps / champ["mean_control_steps"]
    tile_winner = {}
    for column in (
        "tiles_all4_area_100pct", "tiles_all4_area_ge95pct", "tiles_all4_area_ge90pct"):
        a, b = int(champ_tiles[column]), int(trained_tiles[column])
        tile_winner[column] = "tie" if a == b else trained_label if b > a else "champion"

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate E6 equal-sample PPO credit pilot",
        "selected_screen_checkpoint": trained_label,
        "selected_training_arm": "control_48_step_lambda_0.95",
        "long_rollout_candidate_outperformed_control": False,
        "confirmation_sequences_per_policy": 32,
        "confirmation_tiles_per_policy": 800,
        "champion_safety_pass": int(champ["safety_pass"]),
        "trained_safety_pass": int(trained["safety_pass"]),
        "champion_quality_pass": int(champ["quality_pass"]),
        "trained_quality_pass": int(trained["quality_pass"]),
        "delta_mean_all4_area_percentage_points": delta_area,
        "delta_mean_control_steps": delta_steps,
        "delta_mean_control_steps_pct": delta_steps_pct,
        "tile_count_winner": tile_winner,
        "champion_promoted": False,
        "promotion_reason": (
            "No promotion: pilot quality gains are small, strict/95% tile counts do not improve, "
            "and mean completion steps increase materially."),
        "flat_work_complete": False,
        "next_flat_step_requires_approval": (
            "Decide area/tile acceptance thresholds, then run approved larger-seed/full-training gate."),
        "curved_surface_scope": (
            "Not trained in Gate E6; audit as a separate next gate before any unified policy."),
    }
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)

    lines = [
        "# Gate E6 equal-sample PPO credit pilot",
        "",
        "## 결론",
        "",
        f"- 48-step/λ0.95 control과 128-step/λ0.99 candidate는 각각 46,080 samples를 사용했다.",
        f"- screen 최우수는 `{trained_label}`(48-step/λ0.95 control 최종)였다. 긴 rollout 후보가 control을 이기지 못했다.",
        f"- 새 seed same/cross 합계에서 안전 통과는 champion {champ['safety_pass']}/32, trained {trained['safety_pass']}/32이고 품질 집계 통과는 각각 {champ['quality_pass']}/32, {trained['quality_pass']}/32로 같다.",
        f"- 네 목표 동시 통과 면적은 champion {champ['mean_final_all4_pass_area_pct']:.4f}% → trained {trained['mean_final_all4_pass_area_pct']:.4f}% (Δ {delta_area:+.4f}%p)다.",
        f"- 평균 작업 step은 {champ['mean_control_steps']:.1f} → {trained['mean_control_steps']:.1f} (Δ {delta_steps:+.1f}, {delta_steps_pct:+.2f}%)로 늘었다.",
        "- 따라서 champion을 승격/교체하지 않는다. Gate E6는 설정 선택 pilot이며 평면 작업 전체 완료가 아니다.",
        "",
        "## 타일을 가장 많이 통과시킨 정책",
        "",
        f"same/cross 32표면, 정책당 800타일 기준:",
        "",
        f"- 내부 all-4 면적 100% 타일: champion {champ_tiles['tiles_all4_area_100pct']}, trained {trained_tiles['tiles_all4_area_100pct']} — {tile_winner['tiles_all4_area_100pct']}",
        f"- 내부 all-4 면적 ≥95% 타일: champion {champ_tiles['tiles_all4_area_ge95pct']}, trained {trained_tiles['tiles_all4_area_ge95pct']} — {tile_winner['tiles_all4_area_ge95pct']}",
        f"- 내부 all-4 면적 ≥90% 타일: champion {champ_tiles['tiles_all4_area_ge90pct']}, trained {trained_tiles['tiles_all4_area_ge90pct']} — {tile_winner['tiles_all4_area_ge90pct']}",
        "",
        "엄격 100%와 95% 기준에서는 최다 정책이 챔피언과 공동 1위이고, 90% 기준에서만 trained가 더 많다. 타일 평균값 하나로 타일 전체를 통과 처리하지 않았다.",
        "",
        "## 셀 내부 면적 정의",
        "",
        "- 기존 2 mm ROI 셀마다 5×5-cell(10×10 mm) PT-DESIGN 국소창으로 GU/Ra/Rz를 평가했다.",
        "- 품질 4개: local GU≥70, local Ra≤0.20 μm, local Rz≤2.0 μm, scratch가 초기보다 개선 또는 초기<0.05 μm.",
        "- clearcoat/temperature/force-contact는 품질 4개와 섞지 않고 별도 안전으로 판정했다.",
        "- 출력은 SYNTHETIC이며 실제 광택계/조도계 실측으로 표현하면 안 된다.",
        "",
        "## 다음 작업(미승인)",
        "",
        "1. all-4 면적 및 타일 100/95/90% 중 실제 승인 기준을 정한다.",
        "2. 승인 후 선택 설정(현 결과는 48-step/λ0.95)으로 더 큰 seed/본 학습 Gate를 설계한다.",
        "3. 평면 검증 후 곡면은 기존 곡면 자산을 동결한 별도 Gate에서 재현성부터 감사한다. 현재 base14에는 곡률/법선 구분 정보가 없어 바로 혼합 학습하지 않는다.",
        "",
        "## 산출물",
        "",
        "- `screen_sequence_summary.csv`, `screen_tile_counts.csv`",
        "- `confirm_sequence_summary.csv`, `confirm_tile_counts.csv`, `confirm_profile_summary.csv`",
        "- `paired_sequences.csv`, `paired_tiles.csv`, `decision.json`, `checksums.sha256`",
    ]
    readme = os.path.join(out_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for csv_path in [path for path in os.listdir(out_dir) if path.endswith(".csv")]:
        for row in _read(os.path.join(out_dir, csv_path)):
            for value in row.values():
                try:
                    if not math.isfinite(float(value)):
                        raise RuntimeError(f"NaN/Inf in {csv_path}")
                except ValueError:
                    pass
    checksum_targets = sorted(
        name for name in os.listdir(out_dir) if name != "checksums.sha256")
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as handle:
        for name in checksum_targets:
            handle.write(f"{_sha256(os.path.join(out_dir, name))}  {name}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

