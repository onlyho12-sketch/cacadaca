import tempfile
import unittest
from pathlib import Path

from scripts.surface_generalization_benchmark import generate_rows, run


class SurfaceGeneralizationBenchmarkTest(unittest.TestCase):
    def test_six_scenarios_hold_local_20_and_preserve_random_clearcoat(self):
        specs, rows = generate_rows(seed=1234)
        self.assertEqual(len(specs), 6)
        self.assertEqual(len(rows), 150)
        self.assertTrue(all(abs(row["incident_angle_deg"] - 20.0) < 1e-6 for row in rows))
        self.assertTrue(all(abs(row["detection_angle_deg"] - 20.0) < 1e-6 for row in rows))
        self.assertTrue(all(row["clearcoat_before_um"] >= 40.0 for row in rows))
        self.assertTrue(all(row["clearcoat_before_um"] <= 50.0 for row in rows))
        self.assertGreater(
            len({round(row["clearcoat_before_um"], 4) for row in rows}), 100
        )

    def test_seed_is_reproducible(self):
        _, first = generate_rows(seed=99, grid=3)
        _, second = generate_rows(seed=99, grid=3)
        self.assertEqual(first, second)

    def test_outputs_mark_synthetic_and_rtx_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path, plot_path, summary = run(
                Path(directory), seed=7, grid=3
            )
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertTrue(plot_path.is_file())
            self.assertTrue(summary["geometry_passed"])
            self.assertEqual(
                summary["data_origin"], "synthetic_surface_generalization_not_rl"
            )
            self.assertTrue(all(
                not item["rtx_measurement_performed"]
                for item in summary["scenarios"]
            ))


if __name__ == "__main__":
    unittest.main()
