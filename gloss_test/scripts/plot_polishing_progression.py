#!/usr/bin/env python3
"""Aggregate stage summaries into a polishing recovery curve."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


output_dir = Path(sys.argv[1])
specs = [argument.split("=", 1) for argument in sys.argv[2:]]
rows = []
for stage_index, (label, result_dir_text) in enumerate(specs):
    result_dir = Path(result_dir_text)
    summary = json.loads((result_dir / "spatial_summary.json").read_text(encoding="utf-8"))
    measurements = list(csv.DictReader(
        (result_dir / "spatial_gloss_results.csv").open(encoding="utf-8")
    ))
    defect = next(
        row for row in measurements
        if int(row["grid_row"]) == 4 and int(row["grid_column"]) == 4
    )
    rows.append({
        "stage_index": stage_index,
        "stage_label": label,
        "defect_clearcoat_roughness": summary["intentional_defect"]["roughness"],
        "scratch_strength": (2.8, 1.8, 0.9, 0.35, 0.0)[stage_index],
        "relative_gloss_not_GU": float(defect["relative_to_center"]),
        "detected_minimum_row": summary["detected_minimum_cell"][0],
        "detected_minimum_column": summary["detected_minimum_cell"][1],
        "stage_passed": summary["passed"],
    })

output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "polishing_progression.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

fig, axis = plt.subplots(figsize=(7.5, 4.8))
axis.plot(
    [row["stage_index"] for row in rows],
    [row["relative_gloss_not_GU"] for row in rows],
    "o-", linewidth=2.5,
)
axis.set_xticks(
    [row["stage_index"] for row in rows],
    [row["stage_label"] for row in rows],
)
axis.set_ylim(0.0, 1.05)
axis.set_ylabel("Relative gloss (normal = 1.0, not GU)")
axis.set_xlabel("Polishing stage")
axis.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = output_dir / "polishing_gloss_recovery_not_GU.png"
fig.savefig(plot_path, dpi=180)

validation = {
    "metric": "relative_gloss_not_GU",
    "is_GU": False,
    "stage_count": len(rows),
    "monotonic_recovery": all(
        current["relative_gloss_not_GU"] >= previous["relative_gloss_not_GU"]
        for previous, current in zip(rows, rows[1:])
    ),
    "final_recovery_at_least_95pct": rows[-1]["relative_gloss_not_GU"] >= 0.95,
    "all_stages_passed": all(row["stage_passed"] for row in rows),
}
validation["passed"] = all((
    validation["monotonic_recovery"],
    validation["final_recovery_at_least_95pct"],
    validation["all_stages_passed"],
))
(output_dir / "polishing_validation.json").write_text(
    json.dumps(validation, indent=2), encoding="utf-8"
)
print(f"[Polishing] CSV saved: {csv_path}")
print(f"[Polishing] plot saved: {plot_path}")
print(f"[Polishing] validation: {validation}")
if not validation["passed"]:
    raise SystemExit(1)
