import csv
import tempfile
import unittest
from pathlib import Path

from scripts.vehicle_seed_repeatability import (
    REGION_ORDER,
    generate_rows,
    parse_seeds,
    read_geometry,
    run,
)


class VehicleSeedRepeatabilityTest(unittest.TestCase):
    def make_geometry(self, path):
        rows = []
        for region_index, region_id in enumerate(REGION_ORDER):
            for grid_row in range(1, 6):
                for grid_column in range(1, 6):
                    rows.append({
                        "region_id": region_id,
                        "grid_row": grid_row,
                        "grid_column": grid_column,
                        "position_x_m": region_index,
                        "position_y_m": grid_row,
                        "position_z_m": grid_column,
                        "normal_x": 0.0,
                        "normal_y": 0.0,
                        "normal_z": 1.0,
                    })
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_seed_parser_deduplicates(self):
        self.assertEqual(parse_seeds("7, 8,7"), [7, 8])

    def test_generates_reproducible_complete_vehicle_states(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry_path = Path(directory) / "geometry.csv"
            self.make_geometry(geometry_path)
            geometry = read_geometry(geometry_path)
            first = generate_rows(geometry, [7, 8])
            second = generate_rows(geometry, [7, 8])
            self.assertEqual(first[0], second[0])
            self.assertEqual(len(first[0]), 300)
            self.assertEqual(len(first[2]), 12)
            self.assertTrue(all(row["all_targets_pass"] for row in first[0]))
            self.assertTrue(all(not row["actual_rtx_performed"] for row in first[0]))

    def test_run_writes_states_plan_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry_path = root / "geometry.csv"
            self.make_geometry(geometry_path)
            summary, cells, seeds, plan, summary_path, plot = run(
                geometry_path, root / "output", [11, 12]
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["total_cell_count"], 300)
            for path in (cells, seeds, plan, summary_path, plot):
                self.assertTrue(path.is_file())
            self.assertEqual(
                len(list((root / "output" / "states").rglob("*_state_maps.npz"))),
                12,
            )


if __name__ == "__main__":
    unittest.main()
