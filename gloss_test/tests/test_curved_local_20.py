import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gloss_geometry import build_measurement_frame  # noqa: E402
from validate_curved_local_20 import print_terminal_report, run  # noqa: E402


class CurvedLocal20Tests(unittest.TestCase):
    def test_path_tangent_is_projected_to_surface(self):
        tangent, bitangent, normal = build_measurement_frame(
            [0.2, -0.3, 0.93], tangent_hint=[1.0, 0.2, 0.4]
        )
        self.assertAlmostEqual(float(tangent @ normal), 0.0, places=12)
        self.assertAlmostEqual(float(bitangent @ normal), 0.0, places=12)
        self.assertAlmostEqual(float(tangent @ bitangent), 0.0, places=12)

    def test_three_curved_profiles_hold_local_20_degrees(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, summary = run(Path(temporary))
            self.assertEqual(summary["sample_count"], 75)
            self.assertTrue(summary["geometry_passed"])
            self.assertLess(summary["max_incident_angle_error_deg"], 1e-6)
            self.assertLess(summary["max_detection_angle_error_deg"], 1e-6)
            self.assertLess(summary["max_specular_reflection_error_deg"], 1e-6)
            self.assertFalse(summary["is_rtx_measurement"])

    def test_terminal_report_explains_measured_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path, _, _, summary = run(Path(temporary), grid=3)
            import csv
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = []
                for row in csv.DictReader(handle):
                    rows.append({
                        "profile": row["profile"],
                        "incident_angle_deg": float(row["incident_angle_deg"]),
                        "detection_angle_deg": float(row["detection_angle_deg"]),
                        "specular_reflection_error_deg": float(row["specular_reflection_error_deg"]),
                        "normal_spread_p95_deg": float(row["normal_spread_p95_deg"]),
                        "footprint_valid": row["footprint_valid"].lower() == "true",
                    })
            output = StringIO()
            with redirect_stdout(output):
                print_terminal_report(rows, summary)
            rendered = output.getvalue()
            self.assertIn("실제 입사각 범위", rendered)
            self.assertIn("광학 세기·GU 측정 여부", rendered)
            self.assertIn("미측정", rendered)


if __name__ == "__main__":
    unittest.main()
