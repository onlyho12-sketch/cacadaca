"""Triangle-mesh surface sampling utilities independent of Isaac Sim."""

import numpy as np


def triangulate_faces(face_vertex_counts, face_vertex_indices):
    """Fan-triangulate polygon faces and return an ``(T, 3)`` index array."""
    counts = np.asarray(face_vertex_counts, dtype=int)
    indices = np.asarray(face_vertex_indices, dtype=int)
    if np.any(counts < 3):
        raise ValueError("Mesh contains a face with fewer than three vertices")
    if int(counts.sum()) != len(indices):
        raise ValueError("faceVertexCounts and faceVertexIndices lengths disagree")
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset:offset + count]
        for corner in range(1, count - 1):
            triangles.append((face[0], face[corner], face[corner + 1]))
        offset += count
    if not triangles:
        raise ValueError("Mesh has no triangles")
    return np.asarray(triangles, dtype=int)


def area_weighted_vertex_normals(points, triangles):
    """Compute smooth vertex normals from triangle cross products."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    tri_points = points[triangles]
    crosses = np.cross(
        tri_points[:, 1] - tri_points[:, 0],
        tri_points[:, 2] - tri_points[:, 0],
    )
    normals = np.zeros_like(points, dtype=float)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], crosses)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError("Mesh contains a vertex without a valid surface normal")
    return normals / lengths[:, None]


def raycast_nearest(points, triangles, vertex_normals, origin, direction):
    """Return nearest Möller–Trumbore hit or ``None``."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    vertex_normals = np.asarray(vertex_normals, dtype=float)
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    tri_points = points[triangles]
    edge1 = tri_points[:, 1] - tri_points[:, 0]
    edge2 = tri_points[:, 2] - tri_points[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    s = origin - tri_points[:, 0]
    bary_u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    bary_v = inverse * np.einsum("j,ij->i", direction, q)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    valid &= bary_u >= -1e-10
    valid &= bary_v >= -1e-10
    valid &= bary_u + bary_v <= 1.0 + 1e-10
    valid &= distance > 1e-8
    if not np.any(valid):
        return None
    candidates = np.where(valid, distance, np.inf)
    triangle_index = int(np.argmin(candidates))
    u = float(bary_u[triangle_index])
    v = float(bary_v[triangle_index])
    weights = np.asarray([1.0 - u - v, u, v])
    point = origin + candidates[triangle_index] * direction
    normal = weights @ vertex_normals[triangles[triangle_index]]
    normal /= np.linalg.norm(normal)
    # Orient the sampled surface toward the ray origin.  The gloss rig must sit
    # outside the visible surface regardless of source mesh winding.
    if np.dot(normal, -direction) < 0.0:
        normal = -normal
    return {
        "point": point,
        "normal": normal,
        "triangle_index": triangle_index,
        "distance": float(candidates[triangle_index]),
        "barycentric": weights,
    }


def sample_xy_grid(
    points,
    triangles,
    center_xy,
    span_m,
    grid,
    ray_origin_z,
    ray_direction=(0.0, 0.0, -1.0),
):
    """Sample a square XY grid by nearest downward triangle intersections."""
    if grid < 3 or grid % 2 == 0:
        raise ValueError("grid must be an odd integer >= 3")
    vertex_normals = area_weighted_vertex_normals(points, triangles)
    offsets = np.linspace(-span_m / 2.0, span_m / 2.0, grid)
    rows = []
    for row_index, y_offset in enumerate(offsets, start=1):
        for column_index, x_offset in enumerate(offsets, start=1):
            x = float(center_xy[0] + x_offset)
            y = float(center_xy[1] + y_offset)
            hit = raycast_nearest(
                points,
                triangles,
                vertex_normals,
                [x, y, ray_origin_z],
                ray_direction,
            )
            rows.append({
                "grid_row": row_index,
                "grid_column": column_index,
                "x_m": x,
                "y_m": y,
                "hit": hit,
            })
    return rows


def sample_planar_grid(
    points,
    triangles,
    center,
    axis_u,
    axis_v,
    ray_direction,
    span_m,
    grid,
    ray_origin_distance_m=0.5,
):
    """Raycast a square grid from an arbitrary inspection plane.

    ``axis_u`` and ``axis_v`` locate the grid cells. ``ray_direction`` points
    from the inspection rig toward the surface. This supports both horizontal
    panels (hood/roof) and vertical panels (doors/fenders).
    """
    if grid < 3 or grid % 2 == 0:
        raise ValueError("grid must be an odd integer >= 3")
    if span_m <= 0.0:
        raise ValueError("span_m must be positive")
    if ray_origin_distance_m <= 0.0:
        raise ValueError("ray_origin_distance_m must be positive")
    center = np.asarray(center, dtype=float)
    axis_u = np.asarray(axis_u, dtype=float)
    axis_v = np.asarray(axis_v, dtype=float)
    ray_direction = np.asarray(ray_direction, dtype=float)
    if center.shape != (3,) or any(
        vector.shape != (3,) for vector in (axis_u, axis_v, ray_direction)
    ):
        raise ValueError("center and inspection vectors must have three components")
    lengths = [np.linalg.norm(vector) for vector in (axis_u, axis_v, ray_direction)]
    if min(lengths) <= 1.0e-12:
        raise ValueError("inspection vectors must be non-zero")
    axis_u = axis_u / lengths[0]
    axis_v = axis_v / lengths[1]
    ray_direction = ray_direction / lengths[2]
    if abs(float(np.dot(axis_u, axis_v))) > 1.0e-8:
        raise ValueError("axis_u and axis_v must be orthogonal")
    if max(
        abs(float(np.dot(axis_u, ray_direction))),
        abs(float(np.dot(axis_v, ray_direction))),
    ) > 1.0e-8:
        raise ValueError("grid axes must be perpendicular to ray_direction")

    vertex_normals = area_weighted_vertex_normals(points, triangles)
    offsets = np.linspace(-span_m / 2.0, span_m / 2.0, grid)
    rows = []
    for row_index, v_offset in enumerate(offsets, start=1):
        for column_index, u_offset in enumerate(offsets, start=1):
            plane_point = center + u_offset * axis_u + v_offset * axis_v
            origin = plane_point - ray_origin_distance_m * ray_direction
            hit = raycast_nearest(
                points, triangles, vertex_normals, origin, ray_direction
            )
            rows.append({
                "grid_row": row_index,
                "grid_column": column_index,
                "plane_point": plane_point,
                "ray_origin": origin,
                "hit": hit,
            })
    return rows
