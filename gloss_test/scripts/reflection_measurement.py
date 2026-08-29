"""Image persistence and fixed-aperture relative gloss measurement."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def rgb_array(annotator_data):
    payload = annotator_data.get("data") if isinstance(annotator_data, dict) else annotator_data
    array = np.asarray(payload)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"RGB annotator returned invalid data: shape={array.shape}")
    return array[:, :, :3].copy()


def centered_roi(image, fraction):
    height, width = image.shape[:2]
    roi_w = max(2, int(round(width * fraction)))
    roi_h = max(2, int(round(height * fraction)))
    x1 = (width - roi_w) // 2
    y1 = (height - roi_h) // 2
    return image[y1:y1 + roi_h, x1:x1 + roi_w], (x1, y1, x1 + roi_w, y1 + roi_h)


def measure_roi(image, fraction):
    roi, bounds = centered_roi(image, fraction)
    rgb = roi.astype(np.float64)
    if np.issubdtype(image.dtype, np.integer):
        rgb /= float(np.iinfo(image.dtype).max)
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return {
        "roi_mean_intensity": float(luminance.mean()),
        "roi_std_intensity": float(luminance.std()),
        "roi_peak_intensity": float(luminance.max()),
        "saturated_fraction": float(np.mean(luminance >= 0.995)),
        "roi_bounds": bounds,
    }


def save_capture(image, png_path, raw_path, roi_bounds):
    png_path = Path(png_path)
    raw_path = Path(raw_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, image)
    display = image
    if not np.issubdtype(display.dtype, np.integer):
        display = np.clip(display, 0.0, 1.0)
        display = np.round(display * 255.0).astype(np.uint8)
    elif display.dtype != np.uint8:
        display = (display.astype(np.float64) / np.iinfo(display.dtype).max * 255.0).astype(np.uint8)
    output = Image.fromarray(display[:, :, :3], "RGB")
    draw = ImageDraw.Draw(output)
    draw.rectangle(roi_bounds, outline=(255, 40, 40), width=2)
    output.save(png_path)

