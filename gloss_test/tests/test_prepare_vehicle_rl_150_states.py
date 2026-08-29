import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from adapt_vehicle_rl_150 import (  # noqa: E402
    ADAPTER_REQUIRED_COLUMNS,
    create_template,
    validate_and_normalize,
    write_validation_outputs,
)
from gu_proxy import LiteratureGuProxyConfig, relative_gloss_to_gu_proxy  # noqa: E402
from prepare_vehicle_rl_150_states import prepare_states  # noqa: E402


class PrepareVehicleRl150StatesTests(unittest.TestCase):
    def setUp(self):
        self.geometry = (
            ROOT / "results" / "vehicle_multi_region_local_20"
            / "vehicle_multi_region_local_20.csv"
        )

    def _normalized_csv(self, directory):
        raw = Path(directory) / "rl.csv"
        create_template(self.geometry, raw, episode_id="validation_ep_001")
        with raw.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        config = LiteratureGuProxyConfig()
        before_relative, after_relative = 0.20, 0.90
        for row in rows:
            row.update({
                "data_origin": "isaac_lab_ppo_validation_seed_42",
                "force_n": "10", "rpm": "1500", "feed_mm_s": "20",
                "step_over_ratio": "0.25", "pass_count": "2",
                "roughness_before": "0.42", "roughness_after": "0.13",
                "scratch_before": "0.70", "scratch_after": "0.15",
                "ra_before_um": "0.80", "ra_after_um": "0.09",
                "rz_before_um": "4.0", "rz_after_um": "0.52",
                "clearcoat_before_um": "45", "clearcoat_after_um": "42.2",
                "clearcoat_removed_um": "2.8",
                "relative_gloss_before_not_gu": str(before_relative),
                "relative_gloss_after_not_gu": str(after_relative),
                "gu_proxy_before": str(relative_gloss_to_gu_proxy(before_relative, config)),
                "gu_proxy_after": str(relative_gloss_to_gu_proxy(after_relative, config)),
            })
        with raw.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ADAPTER_REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        normalized, summary = validate_and_normalize(raw, self.geometry)
        outputs = write_validation_outputs(Path(directory) / "validated", normalized, summary)
        return outputs[0]

    def test_prepares_six_render_ready_state_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            normalized = self._normalized_csv(directory)
            cells, summary_path, report, summary = prepare_states(
                normalized, Path(directory) / "states"
            )
            self.assertTrue(cells.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(report.is_file())
            self.assertEqual(summary["region_count"], 6)
            self.assertEqual(summary["cell_count"], 150)
            self.assertEqual(summary["totals"]["gu_proxy_target_pass_count"], 150)
            self.assertEqual(summary["totals"]["ra_target_pass_count"], 150)
            self.assertEqual(
                summary["totals"]["all_configured_targets_pass_count"], 150
            )
            self.assertIsNone(summary["totals"]["rz_target_pass_count"])
            self.assertFalse(summary["actual_rtx_performed"])
            for state_path in summary["state_paths"].values():
                with np.load(state_path) as archive:
                    self.assertEqual(archive["roughness_before"].shape, (5, 5))
                    self.assertEqual(archive["force_n"].shape, (5, 5))
                    self.assertTrue(archive["all_configured_targets_pass"].all())

    def test_optional_rz_limit_participates_in_combined_result(self):
        with tempfile.TemporaryDirectory() as directory:
            normalized = self._normalized_csv(directory)
            _, _, _, summary = prepare_states(
                normalized, Path(directory) / "states", rz_max_um=0.50
            )
            self.assertEqual(summary["totals"]["rz_target_pass_count"], 0)
            self.assertEqual(
                summary["totals"]["all_configured_targets_pass_count"], 0
            )


if __name__ == "__main__":
    unittest.main()
