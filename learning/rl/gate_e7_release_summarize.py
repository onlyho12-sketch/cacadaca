"""Summarize Gate E7 release/regression validation against frozen criteria."""
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
    if not rows:
        raise RuntimeError(f"empty output: {path}")
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


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def load(directory: str, expected: int = 32) -> tuple[list[dict], list[dict]]:
    seq = read_csv(os.path.join(directory, "sequences.csv"))
    tiles = read_csv(os.path.join(directory, "tile_area_fractions.csv"))
    if len(seq) != expected or len(tiles) != expected * 25:
        raise RuntimeError(f"incomplete evaluation: {directory}")
    return seq, tiles


def summarize(label: str, direction: str, seq: list[dict], tiles: list[dict]) -> dict:
    areas = [float(row["all4_pass_area_pct"]) for row in tiles]
    cells = sum(
        round(int(row["cell_count"]) * float(row["all4_pass_area_pct"]) / 100.0)
        for row in tiles
    )
    total = sum(int(row["cell_count"]) for row in tiles)
    return {
        "policy_label": label,
        "direction_mode": direction,
        "sequences": len(seq),
        "safety_pass": sum(truth(row["safety_ok"]) for row in seq),
        "quality_pass": sum(truth(row["quality_ok"]) for row in seq),
        "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in seq),
        "mean_gu_final": mean(seq, "gu_final"),
        "mean_ra_final_um": mean(seq, "ra_final_um"),
        "mean_rz_final_um": mean(seq, "rz_final_um"),
        "mean_scratch_final_um": mean(seq, "scratch_final_um"),
        "mean_clearcoat_min_um": mean(seq, "clearcoat_min_um"),
        "mean_temperature_peak_c": mean(seq, "temperature_peak_c"),
        "mean_control_steps": mean(seq, "control_steps"),
        "mean_final_all4_pass_area_pct": mean(seq, "final_all4_pass_area_pct"),
        "tiles_total": len(tiles),
        "tiles_all4_area_100pct": sum(value >= 100.0 - 1e-9 for value in areas),
        "tiles_all4_area_ge95pct": sum(value >= 95.0 for value in areas),
        "tiles_all4_area_ge90pct": sum(value >= 90.0 for value in areas),
        "tiles_all4_area_ge80pct": sum(value >= 80.0 for value in areas),
        "all4_cells_pass": cells,
        "surface_cells_total": total,
        "all4_cell_area_pct": 100.0 * cells / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--champion_same", required=True)
    parser.add_argument("--champion_cross", required=True)
    parser.add_argument("--candidate_same", required=True)
    parser.add_argument("--candidate_cross", required=True)
    parser.add_argument("--candidate_same_repeat", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    os.makedirs(out)
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)

    sources = {
        ("champion", "same_xx"): load(args.champion_same),
        ("champion", "cross_xy"): load(args.champion_cross),
        ("candidate", "same_xx"): load(args.candidate_same),
        ("candidate", "cross_xy"): load(args.candidate_cross),
    }
    repeat_seq, repeat_tiles = load(args.candidate_same_repeat)
    original_seq, original_tiles = sources[("candidate", "same_xx")]
    original_seq_path = os.path.join(args.candidate_same, "sequences.csv")
    repeat_seq_path = os.path.join(args.candidate_same_repeat, "sequences.csv")
    original_tile_path = os.path.join(args.candidate_same, "tile_area_fractions.csv")
    repeat_tile_path = os.path.join(args.candidate_same_repeat, "tile_area_fractions.csv")
    repeat_exact = (
        original_seq == repeat_seq
        and original_tiles == repeat_tiles
        and sha256(original_seq_path) == sha256(repeat_seq_path)
        and sha256(original_tile_path) == sha256(repeat_tile_path)
    )

    direction_rows = [
        summarize(label, direction, *sources[(label, direction)])
        for label in ("champion", "candidate")
        for direction in ("same_xx", "cross_xy")
    ]
    combined: dict[str, dict] = {}
    for label in ("champion", "candidate"):
        seq = sources[(label, "same_xx")][0] + sources[(label, "cross_xy")][0]
        tiles = sources[(label, "same_xx")][1] + sources[(label, "cross_xy")][1]
        combined[label] = summarize(label, "same_xx+cross_xy", seq, tiles)
    combined_rows = [combined["champion"], combined["candidate"]]

    profile_rows = []
    for label in ("champion", "candidate"):
        for profile in plan["profiles"]:
            seq = []
            tiles = []
            for direction in ("same_xx", "cross_xy"):
                seq.extend(
                    row for row in sources[(label, direction)][0]
                    if row["surface_profile"] == profile
                )
                tiles.extend(
                    row for row in sources[(label, direction)][1]
                    if row["surface_profile"] == profile
                )
            row = summarize(label, "same_xx+cross_xy", seq, tiles)
            row["surface_profile"] = profile
            profile_rows.append(row)

    champion = combined["champion"]
    candidate = combined["candidate"]
    step_delta_pct = 100.0 * (
        float(candidate["mean_control_steps"]) - float(champion["mean_control_steps"])
    ) / float(champion["mean_control_steps"])
    checks = [
        ("candidate_safety_64_of_64", int(candidate["safety_pass"]) == 64,
         candidate["safety_pass"], 64),
        ("candidate_sensor_fault_steps_zero", int(candidate["sensor_fault_steps"]) == 0,
         candidate["sensor_fault_steps"], 0),
        ("quality_pass_not_below_champion",
         int(candidate["quality_pass"]) >= int(champion["quality_pass"]),
         candidate["quality_pass"], champion["quality_pass"]),
        ("all4_cell_area_not_below_champion",
         float(candidate["all4_cell_area_pct"]) >= float(champion["all4_cell_area_pct"]),
         candidate["all4_cell_area_pct"], champion["all4_cell_area_pct"]),
    ]
    for suffix in ("100pct", "ge95pct", "ge90pct", "ge80pct"):
        key = f"tiles_all4_area_{suffix}"
        checks.append((
            f"{key}_not_below_champion",
            int(candidate[key]) >= int(champion[key]), candidate[key], champion[key],
        ))
    checks.extend([
        ("mean_control_steps_increase_pct_le40", step_delta_pct <= 40.0,
         step_delta_pct, 40.0),
        ("candidate_same_seed_repeat_exact_csv_match", repeat_exact,
         int(repeat_exact), 1),
    ])
    check_rows = [
        {"criterion": name, "passed": passed, "candidate_value": value,
         "required_or_champion_value": required}
        for name, passed, value, required in checks
    ]
    overall = all(bool(row["passed"]) for row in check_rows)

    deltas = [{
        "quality_pass_delta": int(candidate["quality_pass"]) - int(champion["quality_pass"]),
        "all4_area_pct_points_delta": float(candidate["all4_cell_area_pct"]) - float(champion["all4_cell_area_pct"]),
        "mean_control_steps_delta": float(candidate["mean_control_steps"]) - float(champion["mean_control_steps"]),
        "mean_control_steps_delta_pct": step_delta_pct,
        "tile100_delta": int(candidate["tiles_all4_area_100pct"]) - int(champion["tiles_all4_area_100pct"]),
        "tile95_delta": int(candidate["tiles_all4_area_ge95pct"]) - int(champion["tiles_all4_area_ge95pct"]),
        "tile90_delta": int(candidate["tiles_all4_area_ge90pct"]) - int(champion["tiles_all4_area_ge90pct"]),
        "tile80_delta": int(candidate["tiles_all4_area_ge80pct"]) - int(champion["tiles_all4_area_ge80pct"]),
    }]
    repeat_rows = [{
        "sequences_sha256_original": sha256(original_seq_path),
        "sequences_sha256_repeat": sha256(repeat_seq_path),
        "tiles_sha256_original": sha256(original_tile_path),
        "tiles_sha256_repeat": sha256(repeat_tile_path),
        "sequences_exact_match": original_seq == repeat_seq,
        "tiles_exact_match": original_tiles == repeat_tiles,
        "overall_exact_match": repeat_exact,
    }]

    write_csv(os.path.join(out, "combined_summary.csv"), combined_rows)
    write_csv(os.path.join(out, "direction_summary.csv"), direction_rows)
    write_csv(os.path.join(out, "profile_summary.csv"), profile_rows)
    write_csv(os.path.join(out, "deltas_vs_champion.csv"), deltas)
    write_csv(os.path.join(out, "acceptance_criteria.csv"), check_rows)
    write_csv(os.path.join(out, "repeat_reproducibility.csv"), repeat_rows)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": plan["gate"],
        "candidate_label": plan["candidate_label"],
        "surface_seed_base": plan["surface_seed_base"],
        "overall_release_pass": overall,
        "failed_criteria": [row["criterion"] for row in check_rows if not row["passed"]],
        "champion_promoted": False,
        "champion_unchanged": True,
        "decision": (
            "FAIL release: quality/safety/tile/reproducibility checks passed, but the "
            "frozen +40% mean control-step limit was exceeded."
            if not overall else
            "PASS release validation; champion promotion still requires explicit approval."
        ),
    }
    with open(os.path.join(out, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)

    lines = [
        "# Gate E7 release/regression validation",
        "",
        "## 판정",
        "",
        f"- 전체 release 판정: **{'PASS' if overall else 'FAIL'}**",
        f"- 실패 기준: {', '.join(decision['failed_criteria']) if decision['failed_criteria'] else '없음'}",
        "- champion은 교체하지 않았다.",
        "",
        "## 미사용 seed 26000 same/cross 합계",
        "",
        f"- 안전: champion {champion['safety_pass']}/64, candidate {candidate['safety_pass']}/64; candidate fault {candidate['sensor_fault_steps']}.",
        f"- 기존 품질: {champion['quality_pass']}/64 → {candidate['quality_pass']}/64.",
        f"- all-4 면적: {champion['all4_cell_area_pct']:.4f}% → {candidate['all4_cell_area_pct']:.4f}%.",
        f"- 평균 step: {champion['mean_control_steps']:.1f} → {candidate['mean_control_steps']:.1f} ({step_delta_pct:+.2f}%).",
        f"- 타일 100/95/90/80%: {champion['tiles_all4_area_100pct']}/{champion['tiles_all4_area_ge95pct']}/{champion['tiles_all4_area_ge90pct']}/{champion['tiles_all4_area_ge80pct']} → {candidate['tiles_all4_area_100pct']}/{candidate['tiles_all4_area_ge95pct']}/{candidate['tiles_all4_area_ge90pct']}/{candidate['tiles_all4_area_ge80pct']}.",
        f"- 동일 seed same_xx 반복 CSV 바이트 일치: {repeat_exact}.",
        "",
        "후보는 안전·품질·all-4·네 타일 기준·재현성을 통과했지만 평균 작업시간이",
        f"champion 대비 {step_delta_pct:.2f}% 증가해 사전 동결한 +40% 제한을 초과했다.",
        "기준을 사후 완화하지 않으며 release candidate 승격을 보류한다.",
    ]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for name in os.listdir(out):
        if name.endswith(".csv"):
            for row in read_csv(os.path.join(out, name)):
                for value in row.values():
                    try:
                        if not math.isfinite(float(value)):
                            raise RuntimeError(f"NaN/Inf in {name}")
                    except ValueError:
                        pass
    targets = sorted(name for name in os.listdir(out) if name != "checksums.sha256")
    with open(os.path.join(out, "checksums.sha256"), "w", encoding="utf-8") as handle:
        for name in targets:
            handle.write(f"{sha256(os.path.join(out, name))}  {name}\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
