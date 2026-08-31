"""Pure geometry helpers for the isolated Gate F curved-surface work.

The formulas are selectively ported from ``seb000423/cacadaca``
``learning-bc-v2`` at commit ``cb9ee7e7beca0c40fb6669db659f90681bfcca1c``.
They deliberately do not modify the established ``SurfaceState`` generator or
the shared polishing environments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SURFACE_KINDS = ("flat", "cylinder", "sphere", "freeform")


@dataclass(frozen=True)
class SurfaceGeometrySpec:
    """PT-DESIGN macro-geometry parameters for one polishing map."""

    kind: str = "cylinder"
    patch_size_m: tuple[float, float] = (0.32, 0.32)
    curvature_radius_m: float = 0.60
    freeform_seed: int = 0

    def validate(self) -> None:
        if self.kind not in SURFACE_KINDS:
            raise ValueError(f"kind must be one of {SURFACE_KINDS}, got {self.kind!r}")
        size = np.asarray(self.patch_size_m, dtype=np.float64)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            raise ValueError(f"patch_size_m must contain two finite positive values, got {self.patch_size_m}")
        if self.kind in ("cylinder", "sphere"):
            radius = float(self.curvature_radius_m)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"curvature_radius_m must be finite and positive, got {radius}")
            half = 0.5 * size
            required = half[0] if self.kind == "cylinder" else float(np.linalg.norm(half))
            if radius <= required:
                raise ValueError(
                    f"{self.kind} radius {radius} m must exceed the patch half-span {required} m"
                )


@dataclass(frozen=True)
class SurfaceMesh:
    """Local mesh coordinates and upward-wound triangle indices."""

    vertices_m: np.ndarray
    triangles: np.ndarray
    uv_m: np.ndarray
    normal_xyz: np.ndarray
    grid_size: int
    margin_m: float


def _freeform_parameters(seed: int, patch_size_m: tuple[float, float]) -> tuple[tuple[float, ...], ...]:
    """Return the frozen PT-DESIGN Gaussian-bump parameters."""
    rng = np.random.default_rng(int(seed) + 777_000)
    return tuple(
        (
            float(rng.uniform(0.15, 0.85) * patch_size_m[0]),
            float(rng.uniform(0.15, 0.85) * patch_size_m[1]),
            float(rng.uniform(-0.0025, 0.0025)),
            float(rng.uniform(0.03, 0.06)),
        )
        for _ in range(6)
    )


def surface_height_normal(
    kind: str,
    radius_m: float,
    patch_size_m: tuple[float, float],
    u_m: float | np.ndarray,
    v_m: float | np.ndarray,
    *,
    freeform_seed: int = 0,
) -> tuple[float | np.ndarray, np.ndarray]:
    """Return surface height and unit +outward normal at map coordinates ``u,v``.

    Height is zero at the center for flat/cylinder/sphere and negative toward
    their edges.  Scalar inputs return a Python float height and a ``(3,)``
    normal; broadcast array inputs return matching arrays and ``(..., 3)``.
    """
    spec = SurfaceGeometrySpec(kind, tuple(patch_size_m), float(radius_m), int(freeform_seed))
    spec.validate()
    u, v = np.broadcast_arrays(np.asarray(u_m, dtype=np.float64), np.asarray(v_m, dtype=np.float64))
    scalar = u.ndim == 0
    cu = u - 0.5 * float(patch_size_m[0])
    cv = v - 0.5 * float(patch_size_m[1])

    if kind == "flat":
        height = np.zeros_like(u)
        normal = np.zeros(u.shape + (3,), dtype=np.float64)
        normal[..., 2] = 1.0
    elif kind == "cylinder":
        under = float(radius_m) ** 2 - cu**2
        if np.any(under <= 0.0):
            raise ValueError("cylinder query lies outside the declared radius")
        root = np.sqrt(under)
        height = root - float(radius_m)
        # height = sqrt(R^2-cu^2)-R, so dh/du = -cu/root and the upward
        # graph normal is (-dh/du, 0, 1) = (cu/root, 0, 1).
        normal = np.stack((cu, np.zeros_like(cv), root), axis=-1)
    elif kind == "sphere":
        under = float(radius_m) ** 2 - cu**2 - cv**2
        if np.any(under <= 0.0):
            raise ValueError("sphere query lies outside the declared radius")
        root = np.sqrt(under)
        height = root - float(radius_m)
        # Match the derivatives of the negative-sag height used above.
        normal = np.stack((cu, cv, root), axis=-1)
    else:
        height = np.zeros_like(u)
        du = np.zeros_like(u)
        dv = np.zeros_like(u)
        for center_u, center_v, amplitude, width in _freeform_parameters(
            freeform_seed, tuple(patch_size_m)
        ):
            gaussian = amplitude * np.exp(
                -((u - center_u) ** 2 + (v - center_v) ** 2) / (2.0 * width**2)
            )
            height += gaussian
            du += gaussian * (-(u - center_u) / width**2)
            dv += gaussian * (-(v - center_v) / width**2)
        normal = np.stack((-du, -dv, np.ones_like(u)), axis=-1)

    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    if scalar:
        return float(height), np.asarray(normal, dtype=np.float64).reshape(3)
    return np.asarray(height, dtype=np.float64), np.asarray(normal, dtype=np.float64)


def build_surface_mesh(
    spec: SurfaceGeometrySpec,
    *,
    grid_size: int = 41,
    margin_m: float = 0.02,
) -> SurfaceMesh:
    """Build a deterministic local trimesh covering the map plus a margin."""
    spec.validate()
    if int(grid_size) != grid_size or grid_size < 3:
        raise ValueError(f"grid_size must be an integer >= 3, got {grid_size}")
    if not np.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError(f"margin_m must be finite and non-negative, got {margin_m}")

    u = np.linspace(-margin_m, spec.patch_size_m[0] + margin_m, int(grid_size))
    v = np.linspace(-margin_m, spec.patch_size_m[1] + margin_m, int(grid_size))
    uu, vv = np.meshgrid(u, v, indexing="ij")
    height, normal = surface_height_normal(
        spec.kind,
        spec.curvature_radius_m,
        spec.patch_size_m,
        uu,
        vv,
        freeform_seed=spec.freeform_seed,
    )
    vertices = np.stack(
        (
            uu - 0.5 * spec.patch_size_m[0],
            vv - 0.5 * spec.patch_size_m[1],
            height,
        ),
        axis=-1,
    ).reshape(-1, 3)
    uv = np.stack((uu, vv), axis=-1).reshape(-1, 2)

    faces: list[tuple[int, int, int]] = []
    n = int(grid_size)
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.extend(((a, d, b), (a, c, d)))
    triangles = np.asarray(faces, dtype=np.int32)
    mesh = SurfaceMesh(vertices, triangles, uv, normal.reshape(-1, 3), n, float(margin_m))
    validate_surface_mesh(mesh)
    return mesh


def validate_surface_mesh(mesh: SurfaceMesh) -> dict[str, float | int]:
    """Validate finite arrays, indices, unit normals, and upward winding."""
    vertices = np.asarray(mesh.vertices_m, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.normal_xyz, dtype=np.float64)
    expected_vertices = int(mesh.grid_size) ** 2
    expected_triangles = 2 * (int(mesh.grid_size) - 1) ** 2
    if vertices.shape != (expected_vertices, 3):
        raise ValueError(f"unexpected vertex shape {vertices.shape}")
    if triangles.shape != (expected_triangles, 3):
        raise ValueError(f"unexpected triangle shape {triangles.shape}")
    if normals.shape != vertices.shape:
        raise ValueError(f"unexpected normal shape {normals.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
        raise ValueError("mesh contains NaN or Inf")
    if triangles.min(initial=0) < 0 or triangles.max(initial=0) >= expected_vertices:
        raise ValueError("triangle index is out of range")
    lengths = np.linalg.norm(normals, axis=-1)
    if not np.allclose(lengths, 1.0, atol=1e-10):
        raise ValueError("mesh normals are not unit length")
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    face_cross = np.cross(p1 - p0, p2 - p0)
    face_area2 = np.linalg.norm(face_cross, axis=-1)
    if np.any(face_area2 <= 1e-14):
        raise ValueError("mesh contains a degenerate triangle")
    face_nz = face_cross[:, 2] / face_area2
    if np.any(face_nz <= 0.0):
        raise ValueError("mesh contains a non-upward triangle winding")
    return {
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "normal_length_max_error": float(np.max(np.abs(lengths - 1.0))),
        "minimum_face_normal_z": float(face_nz.min()),
    }


def project_normal_force_n(
    force_xyz_n: np.ndarray,
    normal_xyz: np.ndarray,
) -> np.ndarray:
    """Return ``abs(F dot n)`` with finite/unit-normal validation."""
    force = np.asarray(force_xyz_n, dtype=np.float64)
    normal = np.asarray(normal_xyz, dtype=np.float64)
    if force.shape[-1:] != (3,) or normal.shape[-1:] != (3,):
        raise ValueError("force_xyz_n and normal_xyz must end in dimension 3")
    force, normal = np.broadcast_arrays(force, normal)
    if not np.isfinite(force).all() or not np.isfinite(normal).all():
        raise ValueError("force or normal contains NaN or Inf")
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(length <= 1e-12):
        raise ValueError("normal must have non-zero length")
    unit = normal / length
    return np.abs(np.sum(force * unit, axis=-1))


def vertical_pad_target_z_m(
    work_top_m: float,
    surface_height_m: float | np.ndarray,
    command_clearance_m: float | np.ndarray,
    tracking_compensation_m: float,
) -> float | np.ndarray:
    """Return the vertical-pad face target including Gate F tracking feed-forward.

    The compensation is a small positive world-Z offset that counters the
    measured downward compliance/servo tracking displacement under contact.
    It changes neither the local surface definition nor the force setpoint.
    """
    values = np.broadcast_arrays(
        np.asarray(surface_height_m, dtype=np.float64),
        np.asarray(command_clearance_m, dtype=np.float64),
    )
    if not np.isfinite(float(work_top_m)) or not np.isfinite(float(tracking_compensation_m)):
        raise ValueError("work_top_m and tracking_compensation_m must be finite")
    if float(tracking_compensation_m) < 0.0:
        raise ValueError("tracking_compensation_m must be non-negative")
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("surface height and command clearance must be finite")
    target = float(work_top_m) + values[0] + values[1] + float(tracking_compensation_m)
    return float(target) if target.ndim == 0 else target


def normal_alignment_quaternion_wxyz(normal_xyz: np.ndarray) -> np.ndarray:
    """Return the unique positive-w minimal rotation from ``+Z`` to a normal.

    Gate F surfaces are upward-facing, so the antiparallel singularity cannot
    occur.  The positive-w convention prevents quaternion sign flips along a
    smooth path and leaves a flat normal exactly at identity.
    """
    normal = np.asarray(normal_xyz, dtype=np.float64)
    if normal.shape[-1:] != (3,):
        raise ValueError("normal_xyz must end in dimension 3")
    if not np.isfinite(normal).all():
        raise ValueError("normal_xyz contains NaN or Inf")
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(length <= 1.0e-12):
        raise ValueError("normal_xyz must have non-zero length")
    unit = normal / length
    if np.any(unit[..., 2] <= -1.0 + 1.0e-10):
        raise ValueError("downward antiparallel normals are unsupported")
    denominator = np.sqrt(2.0 * (1.0 + unit[..., 2]))
    quat = np.stack(
        (
            0.5 * denominator,
            -unit[..., 1] / denominator,
            unit[..., 0] / denominator,
            np.zeros_like(unit[..., 2]),
        ),
        axis=-1,
    )
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat


def biased_normal_command(
    normal_xyz: np.ndarray, bias_xy_deg: tuple[float, float]
) -> np.ndarray:
    """Apply a small world-tangent command bias and return unit normals.

    This compensates a measured steady differential-IK orientation bias.  The
    helper is deliberately pure and is not applied to the frozen flat-parity
    command.
    """
    normal = np.asarray(normal_xyz, dtype=np.float64)
    if normal.shape[-1:] != (3,) or not np.isfinite(normal).all():
        raise ValueError("normal_xyz must be finite and end in dimension 3")
    bias = np.asarray(bias_xy_deg, dtype=np.float64)
    if bias.shape != (2,) or not np.isfinite(bias).all():
        raise ValueError("bias_xy_deg must contain two finite values")
    adjusted = normal.copy()
    adjusted[..., :2] += np.tan(np.radians(bias))
    length = np.linalg.norm(adjusted, axis=-1, keepdims=True)
    if np.any(length <= 1.0e-12):
        raise ValueError("biased normal has zero length")
    return adjusted / length


def collected_link6_target_quaternion_xyzw(normal_xyz: np.ndarray) -> np.ndarray:
    """Return the collected M0609 link6 target in Isaac Lab ``xyzw`` order.

    The established flat command is ``[1, 0, 0, 0]`` in ``xyzw`` order, a
    180-degree rotation about link6 X.  Pre-multiplying that frozen baseline by
    the minimal world rotation from ``+Z`` to the requested normal preserves
    the flat pose exactly and rotates link6 local ``-Z`` onto the normal.
    """
    delta_wxyz = normal_alignment_quaternion_wxyz(normal_xyz)
    w = delta_wxyz[..., 0]
    x = delta_wxyz[..., 1]
    y = delta_wxyz[..., 2]
    # q_delta(xyzw) * q_flat(xyzw), with q_flat = [1, 0, 0, 0].
    target_xyzw = np.stack((w, np.zeros_like(w), -y, -x), axis=-1)
    target_xyzw /= np.linalg.norm(target_xyzw, axis=-1, keepdims=True)
    return target_xyzw


def quaternion_step_angle_deg(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return sign-invariant angular changes between consecutive quaternions."""
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quat.ndim < 2 or quat.shape[-1] != 4:
        raise ValueError("quaternion_wxyz must have shape (N, ..., 4)")
    if not np.isfinite(quat).all():
        raise ValueError("quaternion_wxyz contains NaN or Inf")
    length = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(length <= 1.0e-12):
        raise ValueError("quaternion_wxyz contains a zero quaternion")
    unit = quat / length
    dot = np.sum(unit[1:] * unit[:-1], axis=-1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
