#!/usr/bin/env python3
"""Export one complete BMW six-region initial state for RL inference."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapt_vehicle_rl_150 import (  # noqa: E402
    NORMAL_COLUMNS,
    POSITION_COLUMNS,
    _angle_deg,
    canonical_cell_id,
    read_reference_geometry,
    validate_six_region_geometry,
)
from gu_proxy import LiteratureGuProxyConfig, relative_gloss_to_gu_proxy  # noqa: E402


INPUT_COLUMNS = (
    "contract_version",
    "data_origin",
    "source_data_origin",
    "episode_id",
    "source_seed",
    "scenario_seed",
    "region_id",
    "region_label",
    "cell_id",
    "grid_row",
    "grid_column",
    *POSITION_COLUMNS,
    *NORMAL_COLUMNS,
    "roughness_before",
    "scratch_before",
    "ra_before_um",
    "rz_before_um",
    "clearcoat_before_um",
    "relative_gloss_before_not_gu",
    "gu_proxy_before",
)

SOURCE_REQUIRED_COLUMNS = (
    "data_origin",
    "seed",
    "scenario_seed",
    "region_id",
    "grid_row",
    "grid_column",
    *POSITION_COLUMNS,
    *NORMAL_COLUMNS,
    "roughness_before",
    "scratch_before",
    "ra_before_um",
    "rz_before_um",
    "clearcoat_before_um",
    "relative_gloss_before_not_gu",
    "gu_proxy_before",
)

CONTRACT_VERSION = "polytwin_vehicle_6region_150_rl_initial_v1"
DATA_ORIGIN = "synthetic_initial_state_not_rl"


def _read_rows(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: header is missing")
        missing = [name for name in SOURCE_REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: initial-state columns missing: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _finite_number(row, name, identity):
    try:
        value = float(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{identity}: {name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{identity}: {name} must be finite")
    return value


def export_initial_state(
    geometry_csv,
    state_csv,
    output_csv,
    seed,
    episode_id=None,
    position_tolerance_m=1.0e-6,
    normal_angle_tolerance_deg=1.0e-3,
    gu_proxy_tolerance=1.0e-3,
):
    geometry = read_reference_geometry(geometry_csv)
    regions = validate_six_region_geometry(geometry)
    source_rows = _read_rows(state_csv)
    seed = int(seed)
    selected = [row for row in source_rows if int(row["seed"]) == seed]
    if not selected:
        choices = sorted({int(row["seed"]) for row in source_rows})
        raise ValueError(f"seed {seed} not found; choices={choices}")
    if episode_id is None:
        episode_id = f"vehicle_seed_{seed}_initial"
    if not episode_id:
        raise ValueError("episode_id must not be empty")

    config = LiteratureGuProxyConfig()
    output_rows = []
    seen = set()
    maximum_position_error = 0.0
    maximum_normal_error = 0.0
    maximum_gu_error = 0.0
    for source in selected:
        try:
            grid_row = int(source["grid_row"])
            grid_column = int(source["grid_column"])
        except (TypeError, ValueError) as error:
            raise ValueError("grid_row/grid_column must be integers") from error
        key = (source["region_id"].strip(), grid_row, grid_column)
        if key not in geometry:
            raise ValueError(f"unknown vehicle cell: {key}")
        if key in seen:
            raise ValueError(f"duplicate vehicle cell: {key}")
        seen.add(key)
        reference = geometry[key]
        position = tuple(_finite_number(source, name, key) for name in POSITION_COLUMNS)
        normal = tuple(_finite_number(source, name, key) for name in NORMAL_COLUMNS)
        normal_length = math.sqrt(sum(value * value for value in normal))
        if abs(normal_length - 1.0) > 1.0e-3:
            raise ValueError(f"{key}: normal length {normal_length:.6f} is not 1")
        position_error = math.dist(position, reference["position"])
        normal_error = _angle_deg(normal, reference["normal"])
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_normal_error = max(maximum_normal_error, normal_error)
        if position_error > position_tolerance_m:
            raise ValueError(
                f"{key}: position mismatch {position_error:.6g} m exceeds "
                f"{position_tolerance_m:.6g} m"
            )
        if normal_error > normal_angle_tolerance_deg:
            raise ValueError(
                f"{key}: normal mismatch {normal_error:.6g} deg exceeds "
                f"{normal_angle_tolerance_deg:.6g} deg"
            )

        numeric = {
            name: _finite_number(source, name, key)
            for name in (
                "roughness_before",
                "scratch_before",
                "ra_before_um",
                "rz_before_um",
                "clearcoat_before_um",
                "relative_gloss_before_not_gu",
                "gu_proxy_before",
            )
        }
        if not 0.0 <= numeric["roughness_before"] <= 1.0:
            raise ValueError(f"{key}: roughness_before must be in [0, 1]")
        if not 0.0 <= numeric["scratch_before"] <= 1.0:
            raise ValueError(f"{key}: scratch_before must be in [0, 1]")
        if not 0.0 <= numeric["relative_gloss_before_not_gu"] <= 1.0:
            raise ValueError(f"{key}: relative gloss must be in [0, 1]")
        for name in ("ra_before_um", "rz_before_um", "clearcoat_before_um"):
            if numeric[name] < 0.0:
                raise ValueError(f"{key}: {name} must be non-negative")
        if not 0.0 <= numeric["gu_proxy_before"] <= 100.0:
            raise ValueError(f"{key}: gu_proxy_before must be in [0, 100]")
        derived_gu = float(relative_gloss_to_gu_proxy(
            numeric["relative_gloss_before_not_gu"], config
        ))
        gu_error = abs(derived_gu - numeric["gu_proxy_before"])
        maximum_gu_error = max(maximum_gu_error, gu_error)
        if gu_error > gu_proxy_tolerance:
            raise ValueError(
                f"{key}: GU proxy mismatch {gu_error:.6g} exceeds "
                f"{gu_proxy_tolerance:.6g}"
            )

        output_rows.append({
            "contract_version": CONTRACT_VERSION,
            "data_origin": DATA_ORIGIN,
            "source_data_origin": source["data_origin"].strip(),
            "episode_id": episode_id,
            "source_seed": seed,
            "scenario_seed": int(source["scenario_seed"]),
            "region_id": key[0],
            "region_label": reference["region_label"],
            "cell_id": canonical_cell_id(*key),
            "grid_row": grid_row,
            "grid_column": grid_column,
            **{name: value for name, value in zip(POSITION_COLUMNS, position)},
            **{name: value for name, value in zip(NORMAL_COLUMNS, normal)},
            **numeric,
        })

    expected = set(geometry)
    if seen != expected:
        raise ValueError(
            f"seed {seed}: expected 150 cells; missing={sorted(expected - seen)[:10]}, "
            f"extra={sorted(seen - expected)[:10]}"
        )
    output_rows.sort(key=lambda row: (
        row["region_id"], int(row["grid_row"]), int(row["grid_column"])
    ))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "status": "vehicle_6region_150_initial_state_exported",
        "contract_version": CONTRACT_VERSION,
        "data_origin": DATA_ORIGIN,
        "source_state_csv": str(Path(state_csv).resolve()),
        "reference_geometry_csv": str(Path(geometry_csv).resolve()),
        "output_csv": str(output_csv.resolve()),
        "episode_id": episode_id,
        "source_seed": seed,
        "region_count": len(regions),
        "regions": regions,
        "cell_count": len(output_rows),
        "maximum_position_match_error_m": maximum_position_error,
        "maximum_normal_match_error_deg": maximum_normal_error,
        "maximum_gu_proxy_mapping_error": maximum_gu_error,
        "contains_rl_actions": False,
        "contains_after_state": False,
        "ready_for_rl_inference_input": True,
        "important": (
            "This is seeded synthetic initial-state input, not an RL result, physical "
            "measurement, or measured Gloss Meter GU. The RL owner must return actions "
            "and after-state values through the separate output contract."
        ),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    report_path = output_csv.with_suffix(".report.txt")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path.write_text(
        "\n".join((
            "BMW 6영역 150셀 RL 초기 입력",
            f"상태: {summary['status']}",
            f"에피소드: {episode_id}",
            f"합성 시드: {seed}",
            f"영역/셀: {len(regions)}영역 / {len(output_rows)}셀",
            f"최대 위치 오차: {maximum_position_error:.9f} m",
            f"최대 법선 오차: {maximum_normal_error:.9f} deg",
            f"최대 GU proxy 교차검증 오차: {maximum_gu_error:.9f}",
            "RL 행동 포함: 아니오",
            "폴리싱 후 상태 포함: 아니오",
            "주의: 합성 초기 상태이며 실제 RL 결과나 실측 GU가 아닙니다.",
        )) + "\n",
        encoding="utf-8",
    )
    return output_csv, summary_path, report_path, output_rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--state-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--episode-id")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = export_initial_state(
        args.geometry_csv,
        args.state_csv,
        args.output_csv,
        args.seed,
        episode_id=args.episode_id,
    )
    summary = outputs[-1]
    print("[RL input] BMW 6영역 150셀 초기 입력 생성 완료")
    print(f"[RL input] episode={summary['episode_id']}")
    print(f"[RL input] 영역={summary['region_count']}, 셀={summary['cell_count']}")
    print("[RL input] 출처=합성 초기 상태(RL 결과 아님)")
    print("[RL input] 행동/폴리싱 후 상태=미포함(담당자 출력 대상)")
    for path in outputs[:3]:
        print(f"[RL input] output: {path}")


if __name__ == "__main__":
    main()
