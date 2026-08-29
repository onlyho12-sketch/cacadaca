"""Multi-point flat-panel scan for spatial gloss uniformity."""

import csv
import json
import subprocess
import sys

import numpy as np


def run_spatial_scan(
    *, args, cfg, test_root, stage, normal, center_rig, shader,
    light_xform, camera_xform, rep, rgb_annotator, hdr_annotator,
    Gf, measurement_pose, look_at_quaternion, gf_quat,
    set_clearcoat_roughness, measure_roi, rgb_array, save_capture,
):
    output_dir = test_root / "results" / args.tag
    image_dir = output_dir / "spatial_images"
    raw_dir = output_dir / "spatial_raw"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    half_span = cfg.PANEL_SIZE_M / 2.0 - args.edge_margin_m
    if half_span <= 0.0:
        raise ValueError("--edge-margin-m must be smaller than half the panel size")
    coordinates = np.linspace(-half_span, half_span, args.spatial_grid)
    light_ops = light_xform.GetOrderedXformOps()
    camera_ops = camera_xform.GetOrderedXformOps()
    scan_points = []
    rows = []

    set_clearcoat_roughness(shader, args.scan_roughness)
    total = args.spatial_grid * args.spatial_grid
    print(
        f"[Spatial Scan] {args.spatial_grid}x{args.spatial_grid}={total} points, "
        f"roughness={args.scan_roughness:.3f}, edge_margin={args.edge_margin_m * 1000:.1f} mm"
    )
    # Force one sacrificial transform change outside the real 5x5 sequence. With
    # a newly compiled textured material, this Isaac/Replicator combination can
    # return an empty HdrColor payload until the render camera first moves.
    warmup_point = center_rig["point"] + 0.001 * center_rig["tangent"]
    warmup_rig = measurement_pose(
        warmup_point, center_rig["normal"], cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M
    )
    light_ops[0].Set(Gf.Vec3d(*warmup_rig["light_position"]))
    light_ops[1].Set(gf_quat(
        Gf,
        look_at_quaternion(
            warmup_rig["light_position"], warmup_point, warmup_rig["bitangent"]
        ),
    ))
    camera_ops[0].Set(Gf.Vec3d(*warmup_rig["camera_position"]))
    camera_ops[1].Set(gf_quat(
        Gf,
        look_at_quaternion(
            warmup_rig["camera_position"], warmup_point, warmup_rig["bitangent"]
        ),
    ))
    for _ in range(max(12, cfg.SETTLE_FRAMES * 4)):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    warmup_hdr = rgb_array(hdr_annotator.get_data())
    warmup_metrics = measure_roi(warmup_hdr, cfg.ROI_FRACTION)
    print(
        f"[Spatial Scan] discarded sacrificial HDR warmup payload: "
        f"ROI={warmup_metrics['roi_mean_intensity']:.8f}"
    )
    index = 0
    for row_index, v in enumerate(coordinates):
        for column_index, u in enumerate(coordinates):
            index += 1
            point = center_rig["point"] + u * center_rig["tangent"] + v * center_rig["bitangent"]
            rig = measurement_pose(point, normal, cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M)
            scan_points.append(point.copy())

            light_ops[0].Set(Gf.Vec3d(*rig["light_position"]))
            light_ops[1].Set(gf_quat(
                Gf, look_at_quaternion(rig["light_position"], point, rig["bitangent"])
            ))
            camera_ops[0].Set(Gf.Vec3d(*rig["camera_position"]))
            camera_ops[1].Set(gf_quat(
                Gf, look_at_quaternion(rig["camera_position"], point, rig["bitangent"])
            ))

            for capture_attempt in range(1, 4):
                settle_count = cfg.SETTLE_FRAMES if capture_attempt == 1 else 8
                for _ in range(settle_count):
                    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
                ldr_image = rgb_array(rgb_annotator.get_data())
                hdr_image = rgb_array(hdr_annotator.get_data())
                preview = np.clip(
                    hdr_image.astype(np.float32) * cfg.PNG_PREVIEW_EXPOSURE_SCALE, 0.0, 1.0
                )
                hdr_metrics = measure_roi(hdr_image, cfg.ROI_FRACTION)
                preview_metrics = measure_roi(preview, cfg.ROI_FRACTION)
                if hdr_metrics["roi_mean_intensity"] > 0.0:
                    break
                print(
                    f"[Spatial Scan] point {index:02d}: zero HDR during shader warmup; "
                    f"retry {capture_attempt}/3"
                )
            stem = f"row_{row_index + 1:02d}_col_{column_index + 1:02d}"
            save_capture(
                preview,
                image_dir / f"{stem}.png",
                raw_dir / f"{stem}_preview.npy",
                preview_metrics["roi_bounds"],
            )
            np.save(raw_dir / f"{stem}_hdr.npy", hdr_image)
            rows.append({
                "grid_row": row_index + 1,
                "grid_column": column_index + 1,
                "tangent_offset_mm": float(u * 1000.0),
                "bitangent_offset_mm": float(v * 1000.0),
                "world_x_m": float(point[0]),
                "world_y_m": float(point[1]),
                "world_z_m": float(point[2]),
                "clearcoat_roughness": args.scan_roughness,
                "hdr_roi_mean_intensity": hdr_metrics["roi_mean_intensity"],
                "hdr_roi_spatial_std": hdr_metrics["roi_std_intensity"],
                "hdr_roi_peak_intensity": hdr_metrics["roi_peak_intensity"],
                "preview_saturated_fraction": preview_metrics["saturated_fraction"],
            })
            print(
                f"[Spatial Scan] {index:02d}/{total}: row={row_index + 1}, "
                f"col={column_index + 1}, u={u * 1000:+.1f} mm, v={v * 1000:+.1f} mm, "
                f"HDR_ROI={hdr_metrics['roi_mean_intensity']:.8f}"
            )

    center_index = min(
        range(len(rows)),
        key=lambda i: abs(rows[i]["tangent_offset_mm"]) + abs(rows[i]["bitangent_offset_mm"]),
    )
    reference = float(rows[center_index]["hdr_roi_mean_intensity"])
    if reference <= 0.0:
        raise RuntimeError("Center reference intensity is zero")
    for row in rows:
        row["relative_to_center"] = float(row["hdr_roi_mean_intensity"] / reference)

    csv_path = output_dir / "spatial_gloss_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    values = np.asarray([row["hdr_roi_mean_intensity"] for row in rows], dtype=float)
    mean = float(values.mean())
    summary = {
        "mode": "flat_panel_spatial_scan",
        "grid": [args.spatial_grid, args.spatial_grid],
        "point_count": len(rows),
        "clearcoat_roughness": args.scan_roughness,
        "edge_margin_mm": args.edge_margin_m * 1000.0,
        "measurement_aov": "HdrColor",
        "is_gu": False,
        "hdr_mean": mean,
        "hdr_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "hdr_cv": float(values.std(ddof=1) / mean) if len(values) > 1 and mean > 0 else 0.0,
        "hdr_min": float(values.min()),
        "hdr_max": float(values.max()),
        "relative_to_center_min": float(min(row["relative_to_center"] for row in rows)),
        "relative_to_center_max": float(max(row["relative_to_center"] for row in rows)),
        "all_preview_saturated_fraction_below_1pct": all(
            row["preview_saturated_fraction"] < 0.01 for row in rows
        ),
        "all_values_finite_and_positive": bool(np.all(np.isfinite(values)) and np.all(values > 0)),
    }
    minimum_row = min(rows, key=lambda row: float(row["hdr_roi_mean_intensity"]))
    summary["detected_minimum_cell"] = [
        int(minimum_row["grid_row"]), int(minimum_row["grid_column"])
    ]
    if args.defect_cell:
        expected_cell = [int(value.strip()) for value in args.defect_cell.split(",")]
        summary["intentional_defect"] = {
            "expected_cell": expected_cell,
            "roughness": args.defect_roughness,
            "size_mm": args.defect_size_m * 1000.0,
        }
        summary["defect_detected"] = (
            summary["detected_minimum_cell"] == expected_cell
            and float(minimum_row["relative_to_center"]) < 0.95
        )
        summary["expected_defect_present"] = args.expect_defect
        summary["recovered_to_at_least_95pct"] = (
            float(minimum_row["relative_to_center"]) >= 0.95
        )
    summary["passed"] = (
        summary["all_preview_saturated_fraction_below_1pct"]
        and summary["all_values_finite_and_positive"]
        and (
            not args.defect_cell
            or (args.expect_defect and summary["defect_detected"])
            or (not args.expect_defect and summary["recovered_to_at_least_95pct"])
        )
    )
    summary_path = output_dir / "spatial_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stage.Export(str(output_dir / "spatial_scan_scene.usda"))
    print(f"[Spatial Scan] summary: {summary}")
    print(f"[Spatial Scan] CSV saved: {csv_path}")

    plot_script = test_root / "scripts" / "plot_spatial_scan.py"
    subprocess.run([sys.executable, str(plot_script), str(csv_path)], check=True)
    if not summary["passed"]:
        raise RuntimeError(f"Spatial scan validation failed; see {summary_path}")
    return scan_points
