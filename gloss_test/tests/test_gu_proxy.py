import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gu_proxy import (  # noqa: E402
    LiteratureGuProxyConfig,
    gu_proxy_passes,
    relative_gloss_to_gu_proxy,
)


class LiteratureGuProxyTests(unittest.TestCase):
    def setUp(self):
        self.config = LiteratureGuProxyConfig()

    def test_anchor_mapping(self):
        self.assertAlmostEqual(
            relative_gloss_to_gu_proxy(0.0, self.config),
            self.config.defective_anchor_gu,
        )
        self.assertAlmostEqual(
            relative_gloss_to_gu_proxy(1.0, self.config),
            self.config.good_refinish_anchor_gu,
        )

    def test_current_comparison_means(self):
        self.assertAlmostEqual(
            relative_gloss_to_gu_proxy(0.2180435574579558, self.config),
            36.55630854527166,
        )
        self.assertAlmostEqual(
            relative_gloss_to_gu_proxy(0.8814930786790728, self.config),
            71.71913316999086,
        )

    def test_target(self):
        self.assertFalse(gu_proxy_passes(69.999, self.config))
        self.assertTrue(gu_proxy_passes(70.0, self.config))

    def test_relative_input_is_clipped(self):
        self.assertAlmostEqual(
            relative_gloss_to_gu_proxy(-0.5, self.config),
            self.config.defective_anchor_gu,
        )
        self.assertAlmostEqual(
            relative_gloss_to_gu_proxy(1.5, self.config),
            self.config.good_refinish_anchor_gu,
        )


if __name__ == "__main__":
    unittest.main()
