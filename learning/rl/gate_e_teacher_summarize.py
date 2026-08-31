"""Isaac-free aggregation for four immutable Gate E1 teacher/path runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    all_rows: list[dict] = []
    input_metadata = []
    for item in args.inputs:
        run_dir = os.path.abspath(item)
        rows = _read_csv(os.path.join(run_dir, "sequences.csv"))
        all_rows.extend(rows)
        with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as handle:
            input_metadata.append(json.load(handle))

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["teacher"], row["direction_mode"], row["surface_profile"])].append(row)
    summary_rows = []
    for (teacher, direction, profile), rows in sorted(grouped.items()):
        summary_rows.append({
            "teacher": teacher,
            "direction_mode": direction,
            "surface_profile": profile,
            "n": len(rows),
            "success_n": sum(row["outcome"] == "success" for row in rows),
            "quality_ok_n": sum(_truth(row["quality_ok"]) for row in rows),
            "safety_ok_n": sum(_truth(row["safety_ok"]) for row in rows),
            "sensor_fault_steps": sum(int(row["sensor_fault_steps"]) for row in rows),
            "mean_passes": _mean(rows, "passes"),
            "mean_gu_final": _mean(rows, "gu_final"),
            "mean_ra_final_um": _mean(rows, "ra_final_um"),
            "mean_rz_final_um": _mean(rows, "rz_final_um"),
            "mean_scratch_final_um": _mean(rows, "scratch_final_um"),
            "min_clearcoat_um": min(float(row["clearcoat_min_um"]) for row in rows),
            "max_temperature_c": max(float(row["temperature_peak_c"]) for row in rows),
            "later_finish_steps": sum(int(row["later_finish_steps"]) for row in rows),
            "later_protect_steps": sum(int(row["later_protect_steps"]) for row in rows),
            "later_defect_champion_steps": sum(
                int(row["later_defect_champion_steps"]) for row in rows),
        })

    keyed: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for row in all_rows:
        key = (row["direction_mode"], row["surface_profile"], int(row["profile_seed"]))
        keyed[key][row["teacher"]] = row
    paired_rows = []
    for (direction, profile, seed), pair in sorted(keyed.items()):
        if set(pair) != {"champion_then_finish", "spatial_guarded"}:
            raise RuntimeError(f"incomplete teacher pair for {(direction, profile, seed)}")
        base = pair["champion_then_finish"]
        spatial = pair["spatial_guarded"]
        paired_rows.append({
            "direction_mode": direction,
            "surface_profile": profile,
            "profile_seed": seed,
            "baseline_outcome": base["outcome"],
            "spatial_outcome": spatial["outcome"],
            "baseline_safety_ok": base["safety_ok"],
            "spatial_safety_ok": spatial["safety_ok"],
            "delta_gu_spatial_minus_baseline": (
                float(spatial["gu_final"]) - float(base["gu_final"])),
            "delta_ra_um_spatial_minus_baseline": (
                float(spatial["ra_final_um"]) - float(base["ra_final_um"])),
            "delta_rz_um_spatial_minus_baseline": (
                float(spatial["rz_final_um"]) - float(base["rz_final_um"])),
            "delta_scratch_um_spatial_minus_baseline": (
                float(spatial["scratch_final_um"]) - float(base["scratch_final_um"])),
            "delta_clearcoat_um_spatial_minus_baseline": (
                float(spatial["clearcoat_min_um"]) - float(base["clearcoat_min_um"])),
        })

    _write_csv(os.path.join(out_dir, "all_sequences.csv"), all_rows)
    _write_csv(os.path.join(out_dir, "teacher_profile_summary.csv"), summary_rows)
    _write_csv(os.path.join(out_dir, "paired_teacher_differences.csv"), paired_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dirs": [os.path.abspath(item) for item in args.inputs],
        "input_runs": input_metadata,
        "executions": len(all_rows),
        "unique_paired_surfaces": len(paired_rows),
        "training_performed": False,
        "selection_status": "REQUIRES_RESULT_INTERPRETATION",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    checksums = []
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            checksums.append({"file": name, "sha256": _sha256(path)})
    _write_csv(os.path.join(out_dir, "checksums.csv"), checksums)
    print(
        f"Gate E1 summary: executions={len(all_rows)} "
        f"paired_surfaces={len(paired_rows)} -> {out_dir}")


if __name__ == "__main__":
    main()

