import csv
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_vehicle_seed_representative_rtx import planned_cells
from scripts.measurement_cell_selection import parse_measurement_cells


class VehicleRepresentativeCellsTest(unittest.TestCase):
    def test_scanner_adds_center_reference_and_deduplicates(self):
        self.assertEqual(
            parse_measurement_cells("1,1;5,5;1,1", 5),
            [(1, 1), (5, 5), (3, 3)],
        )

    def test_plan_selection_is_seed_and_region_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = [
                {"seed": 7, "region_id": "hood", "grid_row": 1, "grid_column": 2},
                {"seed": 7, "region_id": "roof", "grid_row": 2, "grid_column": 3},
                {"seed": 8, "region_id": "hood", "grid_row": 4, "grid_column": 5},
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(planned_cells(path, 7, "hood"), [(1, 2)])


if __name__ == "__main__":
    unittest.main()
