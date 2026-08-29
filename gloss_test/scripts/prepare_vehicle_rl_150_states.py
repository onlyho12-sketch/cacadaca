#!/usr/bin/env python3
"""Convert a validated 150-cell BMW RL CSV into six render-ready 5x5 states."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


GRID_SIZE = 5
MAP_COLUMNS = (
    "force_n", "rpm", "feed_mm_s", "step_over_ratio", "pass_count",
    "roughness_before", "roughness_after",
    "scratch_before", "scratch_after",
    "ra_before_um", "ra_after_um", "rz_before_um", "rz_after_um",
    "clearcoat_before_um", "clearcoat_removed_um", "clearcoat_after_um",
    "relative_gloss_before_not_gu", "relative_gloss_after_not_gu",
    "gu_proxy_before", "gu_proxy_after",
)


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: header is missing")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    required = {
        "region_id", "region_label", "episode_id", "cell_id",
        "grid_row", "grid_column", "data_origin",
        "position_x_m", "position_y_m", "position_z_m",
        "normal_x", "normal_y", "normal_z", *MAP_COLUMNS,
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"{path}: missing normalized columns: {missing}")
    if not rows:
        raise ValueError(f"{path}: no rows")
    return fieldnames, rows


def _number(row, name, identity):
    try:
        value = float(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{identity}: {name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{identity}: {name} must be finite")
    return value


def group_complete_regions(rows):
    grouped = {}
    identities = set()
    for row in rows:
        region_id = row["region_id"].strip()
        try:
            grid_row = int(row["grid_row"])
            grid_column = int(row["grid_column"])
        except ValueError as error:
            raise ValueError("grid_row/grid_column must be integers") from error
        identity = (region_id, grid_row, grid_column)
        if identity in identities:
            raise ValueError(f"duplicate normalized cell: {identity}")
        identities.add(identity)
        grouped.setdefault(region_id, []).append(row)
    if len(grouped) != 6 or len(rows) != 150:
        raise ValueError(
            f"expected six regions and 150 cells, got {len(grouped)} and {len(rows)}"
        )
    expected = {(row, column) for row in range(1, 6) for column in range(1, 6)}
    for region_id, region_rows in grouped.items():
        cells = {(int(row["grid_row"]), int(row["grid_column"])) for row in region_rows}
        if cells != expected:
            raise ValueError(
                f"{region_id}: incomplete 5x5 grid; "
                f"missing={sorted(expected - cells)}, extra={sorted(cells - expected)}"
            )
        region_rows.sort(key=lambda row: (int(row["grid_row"]), int(row["grid_column"])))
    return dict(sorted(grouped.items()))


def grid_from_rows(rows, column):
    values = np.empty((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for row in rows:
        identity = (row["region_id"], row["grid_row"], row["grid_column"])
        values[int(row["grid_row"]) - 1, int(row["grid_column"]) - 1] = _number(
            row, column, identity
        )
    return values


def prepare_states(
    normalized_csv,
    output_dir,
    target_gu=70.0,
    ra_min_um=0.070,
    ra_max_um=0.100,
    clearcoat_safety_limit_um=35.0,
    rz_max_um=None,
):
    if target_gu < 0.0:
        raise ValueError("target_gu must be non-negative")
    if not 0.0 <= ra_min_um <= ra_max_um:
        raise ValueError("Ra target must satisfy 0 <= min <= max")
    if clearcoat_safety_limit_um < 0.0:
        raise ValueError("clearcoat safety limit must be non-negative")
    if rz_max_um is not None and rz_max_um < 0.0:
        raise ValueError("Rz maximum must be non-negative")

    fieldnames, rows = _read_rows(normalized_csv)
    grouped = group_complete_regions(rows)
    episode_ids = sorted({row["episode_id"] for row in rows})
    if len(episode_ids) != 1:
        raise ValueError(f"normalized CSV must contain one episode, got {episode_ids}")
    origins = sorted({row["data_origin"] for row in rows})

    output_dir = Path(output_dir)
    state_dir = output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    region_summaries = []
    output_rows = []
    state_paths = {}
    for region_id, region_rows in grouped.items():
        maps = {column: grid_from_rows(region_rows, column) for column in MAP_COLUMNS}
        gu_pass = maps["gu_proxy_after"] >= float(target_gu)
        ra_pass = (
            (maps["ra_after_um"] >= float(ra_min_um))
            & (maps["ra_after_um"] <= float(ra_max_um))
        )
        clearcoat_pass = maps["clearcoat_after_um"] >= float(clearcoat_safety_limit_um)
        rz_pass = (
            np.ones_like(gu_pass, dtype=bool)
            if rz_max_um is None else maps["rz_after_um"] <= float(rz_max_um)
        )
        all_configured_pass = gu_pass & ra_pass & clearcoat_pass & rz_pass
        maps.update({
            "gu_proxy_target_pass": gu_pass,
            "ra_target_pass": ra_pass,
            "clearcoat_safety_pass": clearcoat_pass,
            "rz_target_pass": rz_pass,
            "all_configured_targets_pass": all_configured_pass,
        })
        path = state_dir / f"{region_id}_rl_state_maps.npz"
        np.savez_compressed(path, **maps)
        state_paths[region_id] = str(path.resolve())

        for row in region_rows:
            grid_index = (int(row["grid_row"]) - 1, int(row["grid_column"]) - 1)
            enriched = dict(row)
            enriched.update({
                "gu_proxy_target_pass": bool(gu_pass[grid_index]),
                "ra_target_pass": bool(ra_pass[grid_index]),
                "clearcoat_safety_pass": bool(clearcoat_pass[grid_index]),
                "rz_target_configured": rz_max_um is not None,
                "rz_target_pass": bool(rz_pass[grid_index]),
                "all_configured_targets_pass": bool(all_configured_pass[grid_index]),
            })
            output_rows.append(enriched)

        label = region_rows[0].get("region_label", region_id)
        region_summaries.append({
            "region_id": region_id,
            "region_label": label,
            "cell_count": len(region_rows),
            "state_npz": str(path.resolve()),
            "gu_proxy_mean_before": float(maps["gu_proxy_before"].mean()),
            "gu_proxy_mean_after": float(maps["gu_proxy_after"].mean()),
            "gu_proxy_target_pass_count": int(gu_pass.sum()),
            "ra_target_pass_count": int(ra_pass.sum()),
            "clearcoat_safety_pass_count": int(clearcoat_pass.sum()),
            "rz_target_pass_count": int(rz_pass.sum()) if rz_max_um is not None else None,
            "all_configured_targets_pass_count": int(all_configured_pass.sum()),
            "minimum_clearcoat_after_um": float(maps["clearcoat_after_um"].min()),
            "maximum_clearcoat_removed_um": float(maps["clearcoat_removed_um"].max()),
        })

    cells_path = output_dir / "vehicle_rl_150_render_cells.csv"
    with cells_path.open("w", newline="", encoding="utf-8") as handle:
        extra = (
            "gu_proxy_target_pass", "ra_target_pass", "clearcoat_safety_pass",
            "rz_target_configured", "rz_target_pass", "all_configured_targets_pass",
        )
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra])
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "status": "vehicle_6region_150_rl_states_prepared",
        "input_contract_version": "polytwin_vehicle_6region_150_rl_v1",
        "source_csv": str(Path(normalized_csv).resolve()),
        "episode_id": episode_ids[0],
        "data_origins": origins,
        "contains_synthetic_origin": any("synthetic" in origin.lower() for origin in origins),
        "region_count": len(grouped),
        "cell_count": len(rows),
        "targets": {
            "gu_proxy_min": target_gu,
            "ra_after_um_min": ra_min_um,
            "ra_after_um_max": ra_max_um,
            "clearcoat_after_um_min": clearcoat_safety_limit_um,
            "rz_after_um_max": rz_max_um,
            "rz_target_configured": rz_max_um is not None,
        },
        "state_paths": state_paths,
        "regions": region_summaries,
        "totals": {
            "gu_proxy_target_pass_count": sum(
                item["gu_proxy_target_pass_count"] for item in region_summaries
            ),
            "ra_target_pass_count": sum(
                item["ra_target_pass_count"] for item in region_summaries
            ),
            "clearcoat_safety_pass_count": sum(
                item["clearcoat_safety_pass_count"] for item in region_summaries
            ),
            "rz_target_pass_count": (
                sum(item["rz_target_pass_count"] for item in region_summaries)
                if rz_max_um is not None else None
            ),
            "all_configured_targets_pass_count": sum(
                item["all_configured_targets_pass_count"] for item in region_summaries
            ),
        },
        "render_ready": True,
        "actual_rtx_performed": False,
        "important": (
            "These maps preserve received RL values. GU proxy is not measured GU; "
            "Rz is excluded from the combined target unless --rz-max-um is supplied."
        ),
    }
    summary_path = output_dir / "vehicle_rl_150_state_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / "vehicle_rl_150_state_report.txt"
    totals = summary["totals"]
    report_lines = [
        "PolyTwin BMW 6영역 150셀 RL 렌더 상태 준비",
        f"에피소드: {summary['episode_id']}",
        f"영역/셀: {summary['region_count']}영역 / {summary['cell_count']}셀",
        f"GU proxy 목표 통과: {totals['gu_proxy_target_pass_count']}/{len(rows)}",
        f"Ra 목표 통과: {totals['ra_target_pass_count']}/{len(rows)}",
        f"Clearcoat 안전 통과: {totals['clearcoat_safety_pass_count']}/{len(rows)}",
        "Rz 목표: " + (
            f"{totals['rz_target_pass_count']}/{len(rows)}"
            if rz_max_um is not None else "미설정(통합판정에서 제외)"
        ),
        f"설정된 목표 동시 통과: {totals['all_configured_targets_pass_count']}/{len(rows)}",
        "실제 RTX 수행: 아니오",
        "주의: GU proxy는 실측 GU가 아닙니다.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return cells_path, summary_path, report_path, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-gu", type=float, default=70.0)
    parser.add_argument("--ra-min-um", type=float, default=0.070)
    parser.add_argument("--ra-max-um", type=float, default=0.100)
    parser.add_argument("--clearcoat-safety-limit-um", type=float, default=35.0)
    parser.add_argument("--rz-max-um", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = prepare_states(
        args.input_csv,
        args.output_dir,
        target_gu=args.target_gu,
        ra_min_um=args.ra_min_um,
        ra_max_um=args.ra_max_um,
        clearcoat_safety_limit_um=args.clearcoat_safety_limit_um,
        rz_max_um=args.rz_max_um,
    )
    summary = paths[-1]
    totals = summary["totals"]
    print("[RL state] 6영역 150셀 렌더 상태 준비 완료")
    print(f"[RL state] episode={summary['episode_id']}")
    print(f"[RL state] GU proxy 통과={totals['gu_proxy_target_pass_count']}/150")
    print(f"[RL state] Ra 통과={totals['ra_target_pass_count']}/150")
    print(f"[RL state] Clearcoat 통과={totals['clearcoat_safety_pass_count']}/150")
    print(f"[RL state] 동시 통과={totals['all_configured_targets_pass_count']}/150")
    for path in paths[:-1]:
        print(f"[RL state] output: {path}")


if __name__ == "__main__":
    main()
