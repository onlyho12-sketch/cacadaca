import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rl_output import read_and_validate, run  # noqa: E402


class RlOutputBridgeTests(unittest.TestCase):
    def setUp(self):
        self.example = ROOT / "examples" / "rl_polishing_output_example.csv"

    def test_example_contract_and_gu_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, summary = run(self.example, Path(temporary))
            episode = summary["episodes"]["example_ep_001"]
            self.assertEqual(summary["row_count"], 4)
            self.assertEqual(episode["target_pass_cell_count"], 4)
            self.assertGreater(episode["mean_gu_proxy_after"], 70.0)

    def _write_changed_example(self, destination, column, value):
        with self.example.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0][column] = value
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_rejects_non_unit_normal(self):
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.csv"
            self._write_changed_example(invalid, "normal_z", "2.0")
            with self.assertRaisesRegex(ValueError, "normal length"):
                read_and_validate(invalid)

    def test_rejects_clearcoat_mass_balance_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.csv"
            self._write_changed_example(invalid, "clearcoat_removed_um", "99.0")
            with self.assertRaisesRegex(ValueError, "mass balance mismatch"):
                read_and_validate(invalid)

    def test_rejects_missing_optical_output(self):
        with self.example.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = [
                name for name in rows[0]
                if name != "relative_gloss_after_not_gu"
            ]
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.csv"
            with invalid.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_and_validate(invalid)

    def test_rejects_roughness_outside_normalized_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.csv"
            self._write_changed_example(invalid, "roughness_after", "1.1")
            with self.assertRaisesRegex(ValueError, "must be in \[0, 1\]"):
                read_and_validate(invalid)


if __name__ == "__main__":
    unittest.main()
