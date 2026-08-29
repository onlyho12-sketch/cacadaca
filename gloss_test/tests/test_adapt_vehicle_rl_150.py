import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from adapt_vehicle_rl_150 import (  # noqa: E402
    ADAPTER_REQUIRED_COLUMNS,
    canonical_cell_id,
    create_template,
    validate_and_normalize,
    write_validation_outputs,
)
from gu_proxy import LiteratureGuProxyConfig, relative_gloss_to_gu_proxy  # noqa: E402


class VehicleRl150AdapterTests(unittest.TestCase):
    def setUp(self):
        self.geometry = (
            ROOT / "results" / "vehicle_multi_region_local_20"
            / "vehicle_multi_region_local_20.csv"
        )

    def _complete_csv(self, directory):
        path = Path(directory) / "complete_rl.csv"
        create_template(self.geometry, path, episode_id="validation_ep_001")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        config = LiteratureGuProxyConfig()
        before_relative = 0.20
        after_relative = 0.90
        for row in rows:
            row.update({
                "data_origin": "isaac_lab_ppo_validation_seed_42",
                "force_n": "10.0",
                "rpm": "1500.0",
                "feed_mm_s": "20.0",
                "step_over_ratio": "0.25",
                "pass_count": "2",
                "roughness_before": "0.42",
                "roughness_after": "0.13",
                "scratch_before": "0.70",
                "scratch_after": "0.15",
                "ra_before_um": "0.80",
                "ra_after_um": "0.09",
                "rz_before_um": "4.00",
                "rz_after_um": "0.52",
                "clearcoat_before_um": "45.0",
                "clearcoat_after_um": "42.2",
                "clearcoat_removed_um": "2.8",
                "relative_gloss_before_not_gu": str(before_relative),
                "relative_gloss_after_not_gu": str(after_relative),
                "gu_proxy_before": str(relative_gloss_to_gu_proxy(before_relative, config)),
                "gu_proxy_after": str(relative_gloss_to_gu_proxy(after_relative, config)),
            })
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ADAPTER_REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path, rows

    def _rewrite(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ADAPTER_REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_template_contains_six_complete_regions_and_150_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "template.csv"
            _, rows = create_template(self.geometry, output)
            self.assertEqual(len(rows), 150)
            self.assertEqual(len({row["region_id"] for row in rows}), 6)
            self.assertEqual(len({row["cell_id"] for row in rows}), 150)
            first = rows[0]
            self.assertEqual(
                first["cell_id"],
                canonical_cell_id(
                    first["region_id"], first["grid_row"], first["grid_column"]
                ),
            )
            self.assertEqual(first["force_n"], "")

    def test_complete_150_cell_episode_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._complete_csv(directory)
            rows, summary = validate_and_normalize(path, self.geometry)
            self.assertEqual(len(rows), 150)
            self.assertEqual(summary["region_count"], 6)
            self.assertEqual(summary["gu_proxy_target_pass_count"], 150)
            self.assertFalse(summary["contains_synthetic_origin"])
            self.assertTrue(summary["passed"])

            outputs = write_validation_outputs(Path(directory) / "validated", rows, summary)
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(output.is_file() for output in outputs))

    def test_missing_vehicle_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self._complete_csv(directory)
            self._rewrite(path, rows[:-1])
            with self.assertRaisesRegex(ValueError, "missing 1 vehicle cells"):
                validate_and_normalize(path, self.geometry)

    def test_duplicate_vehicle_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self._complete_csv(directory)
            duplicate = dict(rows[0])
            duplicate["cell_id"] = "different_rl_id"
            rows[-1] = duplicate
            self._rewrite(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate vehicle geometry cell"):
                validate_and_normalize(path, self.geometry)

    def test_position_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self._complete_csv(directory)
            rows[0]["position_x_m"] = str(float(rows[0]["position_x_m"]) + 0.01)
            self._rewrite(path, rows)
            with self.assertRaisesRegex(ValueError, "position mismatch"):
                validate_and_normalize(path, self.geometry)

    def test_gu_proxy_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self._complete_csv(directory)
            rows[0]["gu_proxy_after"] = "1.0"
            self._rewrite(path, rows)
            with self.assertRaisesRegex(ValueError, "reported GU proxy disagrees"):
                validate_and_normalize(path, self.geometry)

    def test_clearcoat_mass_balance_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self._complete_csv(directory)
            rows[0]["clearcoat_removed_um"] = "99"
            self._rewrite(path, rows)
            with self.assertRaisesRegex(ValueError, "mass balance mismatch"):
                validate_and_normalize(path, self.geometry)


if __name__ == "__main__":
    unittest.main()
