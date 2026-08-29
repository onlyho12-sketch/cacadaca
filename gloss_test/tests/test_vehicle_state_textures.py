import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.vehicle_state_textures import (
    interpolate_grid,
    load_vehicle_state,
    save_scalar_texture,
    save_scratch_normal_texture,
)


class VehicleStateTexturesTest(unittest.TestCase):
    def make_npz(self, path):
        base = np.linspace(0.1, 0.3, 25, dtype=np.float32).reshape(5, 5)
        clearcoat_before = np.linspace(40.0, 50.0, 25, dtype=np.float32).reshape(5, 5)
        np.savez_compressed(
            path,
            roughness_before=base,
            roughness_after=base * 0.5,
            scratch_before=np.clip(base * 2.0, 0.0, 1.0),
            scratch_after=base * 0.1,
            clearcoat_before_um=clearcoat_before,
            clearcoat_after_um=clearcoat_before - 3.0,
        )

    def test_load_preserves_thickness_and_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            self.make_npz(path)
            state = load_vehicle_state(path, "after", safety_limit_um=35.0)
            self.assertEqual(state["grid_size"], 5)
            self.assertAlmostEqual(float(state["clearcoat_thickness_um"].min()), 37.0)
            self.assertFalse(state["clearcoat_safety_failure"].any())
            self.assertTrue(np.allclose(state["clearcoat_integrity"], 1.0))

    def test_below_safety_reduces_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            self.make_npz(path)
            with np.load(path) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["clearcoat_after_um"] = payload["clearcoat_after_um"].copy()
            payload["clearcoat_after_um"][0, 0] = 17.5
            np.savez_compressed(path, **payload)
            state = load_vehicle_state(path, "after", safety_limit_um=35.0)
            self.assertTrue(state["clearcoat_safety_failure"][0, 0])
            self.assertAlmostEqual(float(state["clearcoat_integrity"][0, 0]), 0.5)

    def test_texture_generation(self):
        values = np.arange(25, dtype=np.float32).reshape(5, 5) / 24.0
        field = interpolate_grid(values, resolution=64)
        self.assertEqual(field.shape, (64, 64))
        self.assertGreater(float(field[4, 32]), float(field[-5, 32]))
        with tempfile.TemporaryDirectory() as directory:
            scalar_path, _ = save_scalar_texture(
                Path(directory) / "scalar.png", values, resolution=64
            )
            normal_path, scratch = save_scratch_normal_texture(
                Path(directory) / "normal.png", values, resolution=64
            )
            self.assertTrue(scalar_path.is_file())
            self.assertTrue(normal_path.is_file())
            self.assertEqual(scratch.shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
