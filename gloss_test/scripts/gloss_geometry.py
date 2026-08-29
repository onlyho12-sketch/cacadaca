"""Local-normal measurement geometry for flat and curved surfaces."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if length < 1.0e-9:
        raise ValueError("surface normal must be non-zero")
    return vector / length


def build_measurement_frame(normal, tangent_hint=None):
    """Return a local frame, using the projected path tangent when provided."""
    normal = normalize(normal)
    if tangent_hint is not None:
        tangent_hint = np.asarray(tangent_hint, dtype=float)
        tangent_projected = tangent_hint - np.dot(tangent_hint, normal) * normal
        if float(np.linalg.norm(tangent_projected)) >= 1.0e-9:
            tangent = normalize(tangent_projected)
            bitangent = normalize(np.cross(normal, tangent))
            return tangent, bitangent, normal
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    tangent = normalize(np.cross(helper, normal))
    bitangent = normalize(np.cross(normal, tangent))
    return tangent, bitangent, normal


def measurement_pose(point, normal, angle_deg, distance, tangent_hint=None):
    """Place source and detector symmetrically around the local normal."""
    point = np.asarray(point, dtype=float)
    tangent, bitangent, normal = build_measurement_frame(normal, tangent_hint)
    angle = math.radians(float(angle_deg))
    light_dir = math.cos(angle) * normal + math.sin(angle) * tangent
    detector_dir = math.cos(angle) * normal - math.sin(angle) * tangent
    return {
        "point": point,
        "tangent": tangent,
        "bitangent": bitangent,
        "normal": normal,
        "light_position": point + distance * light_dir,
        "camera_position": point + distance * detector_dir,
        "light_direction_from_surface": light_dir,
        "detector_direction_from_surface": detector_dir,
    }


def quaternion_from_local_z(normal):
    """Quaternion (wxyz) rotating a local +Z panel normal to `normal`."""
    _, bitangent, normal = build_measurement_frame(normal)
    tangent = normalize(np.cross(bitangent, normal))
    matrix = np.column_stack((tangent, bitangent, normal))
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def look_at_quaternion(position, target, up_hint):
    """Quaternion (wxyz) for USD Camera/RectLight whose forward axis is -Z."""
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    back = normalize(position - target)
    up_hint = normalize(up_hint)
    right = normalize(np.cross(up_hint, back))
    up = normalize(np.cross(back, right))
    matrix = np.column_stack((right, up, back))
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
