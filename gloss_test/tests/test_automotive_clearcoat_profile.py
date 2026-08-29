import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.automotive_clearcoat_profiles import (  # noqa: E402
    WHITE_AUTOMOTIVE_LITERATURE_COMPOSITE_V1,
)


class AutomotiveClearcoatProfileTests(unittest.TestCase):
    def test_literature_sources_are_explicitly_separate(self):
        profile = WHITE_AUTOMOTIVE_LITERATURE_COMPOSITE_V1
        self.assertAlmostEqual(profile.pristine_ra_um, 0.0805)
        self.assertAlmostEqual(profile.high_gloss_anchor_20deg_gu, 88.8)
        self.assertNotEqual(profile.pristine_ra_source_doi, profile.high_gloss_source_doi)

    def test_renderer_values_are_marked_as_design_values(self):
        metadata = WHITE_AUTOMOTIVE_LITERATURE_COMPOSITE_V1.metadata()
        self.assertTrue(metadata["is_literature_composite"])
        self.assertIn("simulation design values", metadata["evidence_scope"]["renderer_parameters"])
        self.assertFalse(metadata["actual_gloss_meter_calibrated"])


if __name__ == "__main__":
    unittest.main()
