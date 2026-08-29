#!/usr/bin/env python3
"""Create a 2D relative-gloss heatmap from a spatial scan CSV."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


csv_path = Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
size = max(int(row["grid_row"]) for row in rows)
values = np.zeros((size, size), dtype=float)
for row in rows:
    values[int(row["grid_row"]) - 1, int(row["grid_column"]) - 1] = float(
        row["relative_to_center"]
    )

extent = [
    -100.0,
    100.0,
    -100.0,
    100.0,
]
color_min = min(float(values.min()), 0.99)
color_max = max(float(values.max()), 1.01)
fig, axis = plt.subplots(figsize=(6.4, 5.2))
image = axis.imshow(
    values,
    origin="lower",
    extent=extent,
    cmap="viridis",
    aspect="equal",
    vmin=color_min,
    vmax=color_max,
)
for row in rows:
    axis.text(
        float(row["tangent_offset_mm"]),
        float(row["bitangent_offset_mm"]),
        f"{float(row['relative_to_center']):.3f}",
        ha="center", va="center", color="white", fontsize=8,
    )
axis.set_xlabel("Tangent offset (mm)")
axis.set_ylabel("Bitangent offset (mm)")
axis.set_title("Spatial relative gloss (center = 1.0, not GU)")
fig.colorbar(image, ax=axis, label="Relative to center")
fig.tight_layout()
output = csv_path.parent / "plots" / "spatial_relative_gloss_heatmap.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180)
print(f"[Spatial Scan] heatmap saved: {output}")
