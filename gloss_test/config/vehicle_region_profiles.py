"""Inspection regions for the bundled BMW Z4 body-paint mesh."""

VEHICLE_REGION_PROFILES = {
    "hood": {
        "label": "보닛",
        "center_m": [0.0, -0.75, 0.68],
        "axis_u": [1.0, 0.0, 0.0],
        "axis_v": [0.0, 1.0, 0.0],
        "ray_direction": [0.0, 0.0, -1.0],
        "span_m": 0.12,
    },
    "roof": {
        "label": "루프",
        "center_m": [0.0, 0.35, 0.96],
        "axis_u": [1.0, 0.0, 0.0],
        "axis_v": [0.0, 1.0, 0.0],
        "ray_direction": [0.0, 0.0, -1.0],
        "span_m": 0.12,
    },
    "negative_x_door": {
        "label": "-X 측 도어",
        "center_m": [-0.64, 0.35, 0.48],
        "axis_u": [0.0, 1.0, 0.0],
        "axis_v": [0.0, 0.0, 1.0],
        "ray_direction": [1.0, 0.0, 0.0],
        "span_m": 0.12,
    },
    "positive_x_door": {
        "label": "+X 측 도어",
        "center_m": [0.64, 0.35, 0.48],
        "axis_u": [0.0, 1.0, 0.0],
        "axis_v": [0.0, 0.0, 1.0],
        "ray_direction": [-1.0, 0.0, 0.0],
        "span_m": 0.12,
    },
    "negative_x_front_fender": {
        "label": "-X 측 앞 펜더 상부",
        "center_m": [-0.64, -0.55, 0.54],
        "axis_u": [0.0, 1.0, 0.0],
        "axis_v": [0.0, 0.0, 1.0],
        "ray_direction": [1.0, 0.0, 0.0],
        "span_m": 0.12,
    },
    "positive_x_front_fender": {
        "label": "+X 측 앞 펜더 상부",
        "center_m": [0.64, -0.55, 0.54],
        "axis_u": [0.0, 1.0, 0.0],
        "axis_v": [0.0, 0.0, 1.0],
        "ray_direction": [-1.0, 0.0, 0.0],
        "span_m": 0.12,
    },
}


def get_vehicle_region_profile(name):
    """Return a copy so callers cannot mutate the shared definitions."""
    if name not in VEHICLE_REGION_PROFILES:
        raise KeyError(f"Unknown vehicle region profile: {name}")
    return dict(VEHICLE_REGION_PROFILES[name])
