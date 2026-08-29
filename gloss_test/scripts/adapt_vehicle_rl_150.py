#!/usr/bin/env python3
"""Create and validate the BMW six-region, 150-cell RL hand-off CSV."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_rl_output import (  # noqa: E402
    REQUIRED_COLUMNS as BASE_REQUIRED_COLUMNS,
    evaluate,
    read_and_validate,
)
from gu_proxy import LiteratureGuProxyConfig  # noqa: E402


ADAPTER_REQUIRED_COLUMNS = (
    "region_id",
    *BASE_REQUIRED_COLUMNS,
    "gu_proxy_before",
    "gu_proxy_after",
)
GEOMETRY_KEY_COLUMNS = ("region_id", "grid_row", "grid_column")
POSITION_COLUMNS = ("position_x_m", "position_y_m", "position_z_m")
NORMAL_COLUMNS = ("normal_x", "normal_y", "normal_z")


def canonical_cell_id(region_id, grid_row, grid_column):
    return f"{region_id}_r{int(grid_row):02d}_c{int(grid_column):02d}"


def _read_csv(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: header is missing")
        return list(reader.fieldnames), list(reader)


def read_reference_geometry(path):
    fieldnames, rows = _read_csv(path)
    required = (*GEOMETRY_KEY_COLUMNS, *POSITION_COLUMNS, *NORMAL_COLUMNS)
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"{path}: geometry columns missing: {missing}")
    if not rows:
        raise ValueError(f"{path}: no geometry rows")

    geometry = {}
    for line_number, row in enumerate(rows, start=2):
        try:
            grid_row = int(row["grid_row"])
            grid_column = int(row["grid_column"])
            position = tuple(float(row[name]) for name in POSITION_COLUMNS)
            normal = tuple(float(row[name]) for name in NORMAL_COLUMNS)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: invalid geometry value at row {line_number}"
            ) from error
        region_id = row["region_id"].strip()
        if not region_id:
            raise ValueError(f"{path}: empty region_id at row {line_number}")
        key = (region_id, grid_row, grid_column)
        if key in geometry:
            raise ValueError(f"{path}: duplicate geometry cell {key}")
        if not all(math.isfinite(value) for value in (*position, *normal)):
            raise ValueError(f"{path}: non-finite geometry at cell {key}")
        normal_length = math.sqrt(sum(value * value for value in normal))
        if abs(normal_length - 1.0) > 1.0e-3:
            raise ValueError(
                f"{path}: reference normal is not unit length at cell {key}"
            )
        geometry[key] = {
            "region_id": region_id,
            "region_label": row.get("region_label", ""),
            "grid_row": grid_row,
            "grid_column": grid_column,
            "position": position,
            "normal": normal,
        }
    return geometry


def validate_six_region_geometry(geometry):
    regions = sorted({key[0] for key in geometry})
    errors = []
    if len(geometry) != 150:
        errors.append(f"expected 150 reference cells, found {len(geometry)}")
    if len(regions) != 6:
        errors.append(f"expected 6 regions, found {len(regions)}: {regions}")
    for region_id in regions:
        cells = {(key[1], key[2]) for key in geometry if key[0] == region_id}
        expected = {(row, column) for row in range(1, 6) for column in range(1, 6)}
        if cells != expected:
            errors.append(
                f"{region_id}: incomplete 5x5 grid; "
                f"missing={sorted(expected - cells)}, extra={sorted(cells - expected)}"
            )
    if errors:
        raise ValueError("; ".join(errors))
    return regions


def create_template(geometry_csv, output_csv, episode_id="replace_with_rl_episode"):
    geometry = read_reference_geometry(geometry_csv)
    validate_six_region_geometry(geometry)
    rows = []
    for key in sorted(geometry):
        cell = geometry[key]
        row = {name: "" for name in ADAPTER_REQUIRED_COLUMNS}
        row.update({
            "data_origin": "replace_with_real_rl_run_id",
            "episode_id": episode_id,
            "cell_id": canonical_cell_id(*key),
            "region_id": cell["region_id"],
            "grid_row": cell["grid_row"],
            "grid_column": cell["grid_column"],
        })
        for name, value in zip(POSITION_COLUMNS, cell["position"]):
            row[name] = f"{value:.12g}"
        for name, value in zip(NORMAL_COLUMNS, cell["normal"]):
            row[name] = f"{value:.12g}"
        rows.append(row)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADAPTER_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv, rows


def _angle_deg(first, second):
    dot = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(value * value for value in first))
    second_length = math.sqrt(sum(value * value for value in second))
    cosine = max(-1.0, min(1.0, dot / first_length / second_length))
    return math.degrees(math.acos(cosine))


def validate_and_normalize(
    input_csv,
    geometry_csv,
    episode_id=None,
    position_tolerance_m=1.0e-3,
    normal_angle_tolerance_deg=1.0,
    gu_proxy_tolerance=1.0e-3,
    target_gu=70.0,
):
    geometry = read_reference_geometry(geometry_csv)
    regions = validate_six_region_geometry(geometry)
    fieldnames, raw_rows = _read_csv(input_csv)
    missing = [name for name in ADAPTER_REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"{input_csv}: missing adapter columns: {missing}")
    if not raw_rows:
        raise ValueError(f"{input_csv}: no data rows")

    episode_ids = sorted({row["episode_id"].strip() for row in raw_rows})
    if episode_id is None:
        if len(episode_ids) != 1:
            raise ValueError(
                f"input has {len(episode_ids)} episodes; select one from {episode_ids}"
            )
        episode_id = episode_ids[0]
    if not episode_id or episode_id not in episode_ids:
        raise ValueError(f"episode {episode_id!r} not found; choices={episode_ids}")

    selected_raw = [row for row in raw_rows if row["episode_id"].strip() == episode_id]
    selected_path = Path(input_csv)
    # The shared validator accepts a file, so validate the selected episode only
    # through an in-memory-equivalent temporary CSV next to no project artifact.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        selected_csv = Path(directory) / "selected_episode.csv"
        with selected_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_raw)
        validated = read_and_validate(selected_csv)

    config = LiteratureGuProxyConfig(target_gu=target_gu)
    evaluated = evaluate(validated, config)
    raw_by_identity = {
        (row["episode_id"].strip(), row["cell_id"].strip()): row
        for row in selected_raw
    }
    normalized = []
    seen_geometry_keys = set()
    maximum_position_error = 0.0
    maximum_normal_error = 0.0
    maximum_gu_error = 0.0

    for row_number, row in enumerate(evaluated, start=2):
        raw = raw_by_identity[(row["episode_id"], row["cell_id"])]
        region_id = raw["region_id"].strip()
        key = (region_id, int(row["grid_row"]), int(row["grid_column"]))
        if key not in geometry:
            raise ValueError(f"row {row_number}: unknown vehicle geometry cell {key}")
        if key in seen_geometry_keys:
            raise ValueError(f"row {row_number}: duplicate vehicle geometry cell {key}")
        seen_geometry_keys.add(key)
        expected_cell_id = canonical_cell_id(*key)
        if row["cell_id"] != expected_cell_id:
            raise ValueError(
                f"row {row_number}: cell_id {row['cell_id']!r} does not match "
                f"canonical {expected_cell_id!r}"
            )

        reference = geometry[key]
        position = tuple(float(row[name]) for name in POSITION_COLUMNS)
        normal = tuple(float(row[name]) for name in NORMAL_COLUMNS)
        position_error = math.dist(position, reference["position"])
        normal_error = _angle_deg(normal, reference["normal"])
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_normal_error = max(maximum_normal_error, normal_error)
        if position_error > position_tolerance_m:
            raise ValueError(
                f"row {row_number}: position mismatch at {key}; "
                f"error={position_error:.6g} m, limit={position_tolerance_m:.6g} m"
            )
        if normal_error > normal_angle_tolerance_deg:
            raise ValueError(
                f"row {row_number}: normal mismatch at {key}; "
                f"error={normal_error:.6g} deg, limit={normal_angle_tolerance_deg:.6g} deg"
            )

        try:
            reported_before = float(raw["gu_proxy_before"])
            reported_after = float(raw["gu_proxy_after"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"row {row_number}: gu_proxy_before/after must be numeric"
            ) from error
        if not all(math.isfinite(value) for value in (reported_before, reported_after)):
            raise ValueError(f"row {row_number}: GU proxy values must be finite")
        if not all(0.0 <= value <= 100.0 for value in (reported_before, reported_after)):
            raise ValueError(f"row {row_number}: GU proxy values must be in [0, 100]")
        derived_before = float(row["predicted_20deg_gu_proxy_before"])
        derived_after = float(row["predicted_20deg_gu_proxy_after"])
        gu_error = max(
            abs(reported_before - derived_before),
            abs(reported_after - derived_after),
        )
        maximum_gu_error = max(maximum_gu_error, gu_error)
        if gu_error > gu_proxy_tolerance:
            raise ValueError(
                f"row {row_number}: reported GU proxy disagrees with optical mapping "
                f"at {key}; error={gu_error:.6g}, limit={gu_proxy_tolerance:.6g}"
            )

        normalized_row = dict(row)
        normalized_row.update({
            "region_id": region_id,
            "region_label": reference["region_label"],
            "gu_proxy_before": reported_before,
            "gu_proxy_after": reported_after,
            "position_match_error_m": position_error,
            "normal_match_error_deg": normal_error,
            "gu_proxy_mapping_error": gu_error,
            "gu_proxy_target_pass": reported_after >= target_gu,
            "clearcoat_mass_balance_pass": True,
        })
        normalized.append(normalized_row)

    expected_keys = set(geometry)
    missing_keys = sorted(expected_keys - seen_geometry_keys)
    if missing_keys:
        raise ValueError(
            f"episode {episode_id!r}: missing {len(missing_keys)} vehicle cells; "
            f"first={missing_keys[:10]}"
        )
    extra_keys = sorted(seen_geometry_keys - expected_keys)
    if extra_keys:
        raise ValueError(f"episode {episode_id!r}: extra vehicle cells: {extra_keys[:10]}")

    normalized.sort(key=lambda row: (
        row["region_id"], int(row["grid_row"]), int(row["grid_column"])
    ))
    origins = sorted({row["data_origin"] for row in normalized})
    summary = {
        "status": "vehicle_6region_150_rl_output_validated",
        "input_contract_version": "polytwin_vehicle_6region_150_rl_v1",
        "source_csv": str(selected_path.resolve()),
        "episode_id": episode_id,
        "data_origins": origins,
        "contains_synthetic_origin": any("synthetic" in value.lower() for value in origins),
        "region_count": len(regions),
        "regions": regions,
        "cell_count": len(normalized),
        "expected_cell_count": len(geometry),
        "maximum_position_match_error_m": maximum_position_error,
        "position_tolerance_m": position_tolerance_m,
        "maximum_normal_match_error_deg": maximum_normal_error,
        "normal_angle_tolerance_deg": normal_angle_tolerance_deg,
        "maximum_gu_proxy_mapping_error": maximum_gu_error,
        "gu_proxy_tolerance": gu_proxy_tolerance,
        "gu_proxy_target": target_gu,
        "gu_proxy_target_pass_count": sum(row["gu_proxy_target_pass"] for row in normalized),
        "quality_improvement_counts": {
            "roughness": sum(row["roughness_after"] <= row["roughness_before"] for row in normalized),
            "scratch": sum(row["scratch_after"] <= row["scratch_before"] for row in normalized),
            "ra": sum(row["ra_after_um"] <= row["ra_before_um"] for row in normalized),
            "rz": sum(row["rz_after_um"] <= row["rz_before_um"] for row in normalized),
            "gu_proxy": sum(row["gu_proxy_after"] >= row["gu_proxy_before"] for row in normalized),
        },
        "minimum_clearcoat_after_um": min(row["clearcoat_after_um"] for row in normalized),
        "maximum_clearcoat_removed_um": max(row["clearcoat_removed_um"] for row in normalized),
        "passed": True,
        "important": (
            "Validation proves the RL CSV contract and BMW cell mapping only. "
            "GU values remain literature proxies, not measured Gloss Meter GU."
        ),
    }
    return normalized, summary


def write_validation_outputs(output_dir, rows, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rl_vehicle_150_cells_normalized.csv"
    json_path = output_dir / "rl_vehicle_150_validation.json"
    text_path = output_dir / "rl_vehicle_150_validation.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    improvements = summary["quality_improvement_counts"]
    lines = [
        "PolyTwin RL → BMW 6영역 150셀 검증",
        f"상태: {summary['status']}",
        f"에피소드: {summary['episode_id']}",
        f"출처: {', '.join(summary['data_origins'])}",
        f"영역/셀: {summary['region_count']}영역 / {summary['cell_count']}셀",
        f"최대 위치 오차: {summary['maximum_position_match_error_m']:.9f} m",
        f"최대 법선 오차: {summary['maximum_normal_match_error_deg']:.9f} deg",
        f"최대 GU proxy 교차검증 오차: {summary['maximum_gu_proxy_mapping_error']:.9f}",
        f"GU proxy 목표 통과: {summary['gu_proxy_target_pass_count']}/{summary['cell_count']}",
        f"Roughness 개선/유지: {improvements['roughness']}/{summary['cell_count']}",
        f"Scratch 개선/유지: {improvements['scratch']}/{summary['cell_count']}",
        f"Ra 개선/유지: {improvements['ra']}/{summary['cell_count']}",
        f"Rz 개선/유지: {improvements['rz']}/{summary['cell_count']}",
        f"GU proxy 개선/유지: {improvements['gu_proxy']}/{summary['cell_count']}",
        f"최소 잔여 Clearcoat: {summary['minimum_clearcoat_after_um']:.6f} um",
        f"최대 Clearcoat 제거량: {summary['maximum_clearcoat_removed_um']:.6f} um",
        "주의: GU proxy는 실측 GU가 아닙니다.",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, text_path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template", help="create the blank 150-cell hand-off CSV")
    template.add_argument("--geometry-csv", type=Path, required=True)
    template.add_argument("--output-csv", type=Path, required=True)
    template.add_argument("--episode-id", default="replace_with_rl_episode")

    validate = subparsers.add_parser("validate", help="validate one received RL episode")
    validate.add_argument("--geometry-csv", type=Path, required=True)
    validate.add_argument("--input-csv", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--episode-id")
    validate.add_argument("--position-tolerance-m", type=float, default=1.0e-3)
    validate.add_argument("--normal-angle-tolerance-deg", type=float, default=1.0)
    validate.add_argument("--gu-proxy-tolerance", type=float, default=1.0e-3)
    validate.add_argument("--target-gu", type=float, default=70.0)
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "template":
        path, rows = create_template(args.geometry_csv, args.output_csv, args.episode_id)
        print("[RL 150 adapter] 빈 입력 템플릿 생성 완료")
        print(f"[RL 150 adapter] 영역=6, 셀={len(rows)}")
        print(f"[RL 150 adapter] CSV: {path}")
        print("[RL 150 adapter] 공정·품질 열은 RL 담당자가 채워야 합니다.")
        return
    rows, summary = validate_and_normalize(
        args.input_csv,
        args.geometry_csv,
        episode_id=args.episode_id,
        position_tolerance_m=args.position_tolerance_m,
        normal_angle_tolerance_deg=args.normal_angle_tolerance_deg,
        gu_proxy_tolerance=args.gu_proxy_tolerance,
        target_gu=args.target_gu,
    )
    paths = write_validation_outputs(args.output_dir, rows, summary)
    print("[RL 150 adapter] 검증 통과")
    print(f"[RL 150 adapter] episode={summary['episode_id']}")
    print(f"[RL 150 adapter] 영역={summary['region_count']}, 셀={summary['cell_count']}")
    print(
        "[RL 150 adapter] 위치/법선 최대 오차="
        f"{summary['maximum_position_match_error_m']:.9f} m / "
        f"{summary['maximum_normal_match_error_deg']:.9f} deg"
    )
    print(
        "[RL 150 adapter] GU proxy 목표 통과="
        f"{summary['gu_proxy_target_pass_count']}/{summary['cell_count']}"
    )
    print(
        "[RL 150 adapter] 최소 잔여 Clearcoat="
        f"{summary['minimum_clearcoat_after_um']:.6f} um"
    )
    for path in paths:
        print(f"[RL 150 adapter] output: {path}")


if __name__ == "__main__":
    main()
