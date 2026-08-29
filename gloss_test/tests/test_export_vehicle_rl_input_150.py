import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_vehicle_rl_input_150 import export_initial_state  # noqa: E402


class ExportVehicleRlInput150Tests(unittest.TestCase):
    def setUp(self):
        self.geometry = (
            ROOT / "results" / "vehicle_multi_region_local_20"
            / "vehicle_multi_region_local_20.csv"
        )
        self.states = (
            ROOT / "results" / "vehicle_seed_repeatability"
            / "vehicle_seed_repeatability_cells.csv"
        )

    def test_exports_one_complete_initial_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vehicle_150_cells.csv"
            _, summary_path, report_path, rows, summary = export_initial_state(
                self.geometry, self.states, output, seed=20260828
            )
            self.assertEqual(len(rows), 150)
            self.assertEqual(len({row["region_id"] for row in rows}), 6)
            self.assertEqual(len({row["cell_id"] for row in rows}), 150)
            self.assertEqual({row["data_origin"] for row in rows}, {
                "synthetic_initial_state_not_rl"
            })
            self.assertNotIn("force_n", rows[0])
            self.assertNotIn("roughness_after", rows[0])
            self.assertTrue(summary["ready_for_rl_inference_input"])
            self.assertFalse(summary["contains_rl_actions"])
            self.assertFalse(summary["contains_after_state"])
            self.assertTrue(output.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(report_path.is_file())

    def test_unknown_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "seed 999 not found"):
                export_initial_state(
                    self.geometry,
                    self.states,
                    Path(directory) / "vehicle_150_cells.csv",
                    seed=999,
                )

    def test_missing_source_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "states.csv"
            with self.states.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)
            rows = [
                row for row in rows
                if not (
                    row["seed"] == "20260828"
                    and row["region_id"] == "hood"
                    and row["grid_row"] == "1"
                    and row["grid_column"] == "1"
                )
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "expected 150 cells"):
                export_initial_state(
                    self.geometry,
                    source,
                    root / "vehicle_150_cells.csv",
                    seed=20260828,
                )


if __name__ == "__main__":
    unittest.main()
