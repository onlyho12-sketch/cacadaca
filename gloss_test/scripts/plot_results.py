"""Generate a simple result plot outside or inside Isaac Python."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


csv_path = Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
x = [float(row["clearcoat_roughness"]) for row in rows]
y = [float(row["relative_gloss"]) for row in rows]
yerr = [float(row.get("relative_gloss_std", 0.0)) for row in rows]

plt.figure(figsize=(7, 4.5))
plt.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=2, capsize=4)
plt.xlabel("Clearcoat roughness (renderer parameter)")
plt.ylabel("Relative gloss (not GU)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
output = csv_path.parent / "plots" / "relative_gloss_vs_roughness.png"
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=160)
print(f"[Gloss Test] plot saved: {output}")
