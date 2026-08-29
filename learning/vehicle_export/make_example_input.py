"""차량 6영역 × 5×5 = 150셀 입력 CSV 예제 생성.

    ~/isaacsim/kit/python/bin/python3 learning/vehicle_export/make_example_input.py

실제 운용에서는 차량 검사 시스템이 이 스키마로 CSV 를 공급한다 (README 3장).
초기 Rq/Rz/GU 는 참고열 — 여기서는 같은 seed 로 합성한 표면에서 twin 이 잰 값을 넣어
입력 예제가 자기모순 없게 한다 (export 스크립트도 같은 절차로 재합성).
"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel      # noqa: E402
from learning.polytwin.roughness_metrics import ra_um, rz_um             # noqa: E402
from learning.vehicle_export.export_vehicle_results import (             # noqa: E402
    INPUT_COLUMNS, rq_um, synthesize_cell_patch)

# 영역: (id, 이름, 원점[m], u축, v축, 법선) — 차량 좌표계 x 전방 / y 좌 / z 상
# 셀 격자는 u,v 방향 5×5, 셀 간격은 영역 크기/5.
REGIONS = [
    ("R1", "bonnet",        (1.60, -0.60, 0.95), (1.00, 0.0, 0.10), (0.0, 1.20, 0.0), (-0.10, 0.0, 0.995)),
    ("R2", "roof",          (-0.20, -0.55, 1.45), (1.20, 0.0, 0.0),  (0.0, 1.10, 0.0), (0.0, 0.0, 1.0)),
    ("R3", "door_left",     (-0.10, 0.85, 0.55), (1.20, 0.0, 0.0),  (0.0, 0.0, 0.45), (0.0, 1.0, 0.0)),
    ("R4", "door_right",    (-0.10, -0.85, 0.55), (1.20, 0.0, 0.0), (0.0, 0.0, 0.45), (0.0, -1.0, 0.0)),
    ("R5", "fender_left",   (1.90, 0.72, 0.60), (0.60, 0.0, 0.10),  (0.0, 0.05, 0.30), (0.15, 0.97, 0.19)),
    ("R6", "fender_right",  (1.90, -0.72, 0.60), (0.60, 0.0, 0.10), (0.0, -0.05, 0.30), (0.15, -0.97, 0.19)),
]
GRID = 5


def main():
    rng = np.random.default_rng(2026)
    gloss = LiteratureGlossProxyModel()
    out = os.path.join(_HERE, "vehicle_150_cells.csv")
    rows = []
    for rid, rname, origin, u, v, normal in REGIONS:
        origin, u, v = map(np.asarray, (origin, u, v))
        n = np.asarray(normal, float); n /= np.linalg.norm(n)
        for i in range(GRID):
            for j in range(GRID):
                cell = i * GRID + j
                pos = origin + u * ((i + 0.5) / GRID) + v * ((j + 0.5) / GRID)
                seed = 7000 + 101 * int(rid[1]) + cell     # export 기본 seed 와 동일 규칙
                ra0 = float(np.clip(rng.normal(0.08, 0.012), 0.05, 0.12))
                scr0 = float(rng.uniform(0.4, 1.9))
                cc0 = float(rng.uniform(40.0, 50.0))
                st = synthesize_cell_patch(seed, ra0, scr0, cc0)
                rows.append({
                    "region_id": rid, "region_name": rname, "cell_id": cell,
                    "position_x_m": f"{pos[0]:.3f}", "position_y_m": f"{pos[1]:.3f}",
                    "position_z_m": f"{pos[2]:.3f}",
                    "normal_x": f"{n[0]:.4f}", "normal_y": f"{n[1]:.4f}", "normal_z": f"{n[2]:.4f}",
                    "init_roughness_um": f"{rq_um(st.micro_height_um):.4f}",
                    "init_scratch_um": f"{scr0:.3f}",
                    "init_ra_um": f"{ra0:.4f}",
                    "init_rz_um": f"{rz_um(st.micro_height_um):.4f}",
                    "init_clearcoat_um": f"{cc0:.2f}",
                    "init_gu_proxy": f"{gloss.evaluate(st)['summary']['gu_mean']:.2f}",
                    "surface_seed": seed,
                })
        print(f"  {rid} {rname}: {GRID*GRID} cells")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INPUT_COLUMNS)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
