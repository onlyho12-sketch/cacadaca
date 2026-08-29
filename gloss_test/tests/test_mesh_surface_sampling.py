import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mesh_surface_sampling import (  # noqa: E402
    area_weighted_vertex_normals,
    sample_planar_grid,
    sample_xy_grid,
    triangulate_faces,
)


class MeshSurfaceSamplingTests(unittest.TestCase):
    def test_quad_triangulation_and_normal(self):
        triangles = triangulate_faces([4], [0, 1, 2, 3])
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 2, 3]])
        points = np.asarray([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        ])
        normals = area_weighted_vertex_normals(points, triangles)
        np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]] * 4)

    def test_grid_hits_tilted_mesh_and_orients_upward(self):
        points = np.asarray([
            [-1.0, -1.0, -0.2], [1.0, -1.0, 0.2],
            [1.0, 1.0, 0.2], [-1.0, 1.0, -0.2],
        ])
        triangles = triangulate_faces([4], [0, 1, 2, 3])
        rows = sample_xy_grid(points, triangles, [0.0, 0.0], 1.0, 5, 2.0)
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(row["hit"] is not None for row in rows))
        center = rows[12]["hit"]
        self.assertAlmostEqual(float(center["point"][2]), 0.0, places=12)
        self.assertGreater(float(center["normal"][2]), 0.9)

    def test_outside_mesh_is_reported_as_missing(self):
        points = np.asarray([
            [-0.1, -0.1, 0.0], [0.1, -0.1, 0.0],
            [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0],
        ])
        triangles = triangulate_faces([4], [0, 1, 2, 3])
        rows = sample_xy_grid(points, triangles, [0.0, 0.0], 1.0, 3, 1.0)
        self.assertEqual(sum(row["hit"] is not None for row in rows), 1)

    def test_planar_grid_hits_vertical_side(self):
        points = np.asarray([
            [0.0, -1.0, -1.0], [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0], [0.0, -1.0, 1.0],
        ])
        triangles = triangulate_faces([4], [0, 1, 2, 3])
        rows = sample_planar_grid(
            points, triangles, [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
            1.0, 5, ray_origin_distance_m=1.0,
        )
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(row["hit"] is not None for row in rows))
        center = rows[12]["hit"]
        np.testing.assert_allclose(center["point"], [0.0, 0.0, 0.0])
        self.assertLess(float(center["normal"][0]), -0.9)

    def test_planar_grid_rejects_nonorthogonal_axes(self):
        points = np.asarray([
            [0.0, -1.0, -1.0], [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0], [0.0, -1.0, 1.0],
        ])
        triangles = triangulate_faces([4], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            sample_planar_grid(
                points, triangles, [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0],
                1.0, 5,
            )


if __name__ == "__main__":
    unittest.main()
