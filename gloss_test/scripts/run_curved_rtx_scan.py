#!/usr/bin/env python3
"""Measure an analytic curved clearcoat patch at local 20-degree RTX poses."""

import argparse
import csv
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from config import gloss_config as cfg
from config.automotive_clearcoat_profiles import PROFILES, get_clearcoat_profile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="curved_freeform_rtx_5x5")
    parser.add_argument(
        "--surface-profile",
        choices=(
            "plane", "cylinder", "concave_cylinder", "sphere",
            "freeform", "freeform_strong",
        ),
        default="freeform",
    )
    parser.add_argument(
        "--material-profile",
        choices=tuple(PROFILES),
        default="white_automotive_literature_composite_v1",
    )
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--mesh-resolution", type=int, default=51)
    parser.add_argument("--scan-span-m", type=float, default=0.16)
    parser.add_argument("--edge-margin-m", type=float, default=0.02)
    parser.add_argument("--roughness", type=float, default=0.10)
    parser.add_argument(
        "--distributed-roughness", choices=("initial", "improved"), default=None
    )
    parser.add_argument("--roughness-seed", type=int, default=20260827)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()
    if args.grid < 3 or args.grid % 2 == 0:
        parser.error("--grid must be an odd integer >= 3")
    if args.mesh_resolution < 11:
        parser.error("--mesh-resolution must be >= 11")
    if not 0.0 <= args.roughness <= 1.0:
        parser.error("--roughness must be in [0, 1]")
    if args.scan_span_m <= 0.0 or args.edge_margin_m <= 0.0:
        parser.error("--scan-span-m and --edge-margin-m must be positive")
    if args.keep_open and args.headless:
        parser.error("--keep-open requires --no-headless")
    return args


args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "renderer": "PathTracing",
    "width": 512,
    "height": 512,
    "samples_per_pixel_per_frame": cfg.PATH_TRACING_SPP,
    "denoiser": False,
})


def gf_quat(Gf, quat):
    return Gf.Quatf(float(quat[0]), Gf.Vec3f(*[float(value) for value in quat[1:]]))


def create_analytic_mesh(
    stage, UsdGeom, Gf, Sdf, path, span_m, resolution, surface, surface_profile
):
    coordinates = np.linspace(-span_m / 2.0, span_m / 2.0, resolution)
    points = []
    normals = []
    for v in coordinates:
        for u in coordinates:
            point, normal, _ = surface(surface_profile, float(u), float(v))
            points.append(Gf.Vec3f(*[float(value) for value in point]))
            normals.append(Gf.Vec3f(*[float(value) for value in normal]))
    counts = []
    indices = []
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            lower_left = row * resolution + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * resolution + column
            upper_right = upper_left + 1
            counts.append(4)
            indices.extend([lower_left, lower_right, upper_right, upper_left])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([
        Gf.Vec2f(float(column / (resolution - 1)), float(row / (resolution - 1)))
        for row in range(resolution)
        for column in range(resolution)
    ])
    return mesh


def save_heatmap(path, rows, grid, surface_profile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray([row["relative_gloss_to_center_not_gu"] for row in rows]).reshape(grid, grid)
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axis.imshow(values, origin="lower", cmap="viridis")
    for row in range(grid):
        for column in range(grid):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    axis.set_title(
        f"{surface_profile.capitalize()} panel\n"
        "RTX relative gloss (center = 1.0, not GU)"
    )
    axis.set_xlabel("grid column")
    axis.set_ylabel("grid row")
    axis.set_xticks(range(grid), range(1, grid + 1))
    axis.set_yticks(range(grid), range(1, grid + 1))
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def add_bright_overview(stage, points_and_normals, Gf, Sdf, UsdGeom, UsdLux,
                        UsdShade, look_at_quaternion):
    """Add post-measurement-only bright inspection assets and overview camera."""
    root = "/World/CurvedGlossTest/GUI_ONLY_PostMeasurement"
    UsdGeom.Xform.Define(stage, root)

    backdrop_material = UsdShade.Material.Define(stage, root + "/BackdropMaterial")
    backdrop_shader = UsdShade.Shader.Define(stage, root + "/BackdropMaterial/Shader")
    backdrop_shader.CreateIdAttr("UsdPreviewSurface")
    backdrop_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.22, 0.25, 0.30)
    )
    backdrop_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    backdrop_material.CreateSurfaceOutput().ConnectToSource(
        backdrop_shader.ConnectableAPI(), "surface"
    )
    backdrop = UsdGeom.Cube.Define(stage, root + "/WhiteBackdrop")
    backdrop.CreateSizeAttr(1.0)
    backdrop_xform = UsdGeom.Xformable(backdrop)
    backdrop_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.055))
    backdrop_xform.AddScaleOp().Set(Gf.Vec3d(0.65, 0.65, 0.015))
    UsdShade.MaterialBindingAPI.Apply(backdrop.GetPrim()).Bind(backdrop_material)

    dome = UsdLux.DomeLight.Define(stage, root + "/BrightEnvironment")
    # Moderate neutral fill retains the white paint while avoiding a white-on-white
    # washout in the post-measurement GUI.
    dome.CreateIntensityAttr(260.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    camera_position = np.array([0.0, -0.42, 0.30], dtype=float)
    target = np.array([0.0, 0.0, 0.015], dtype=float)
    camera = UsdGeom.Camera.Define(stage, root + "/BrightOverviewCamera")
    camera.CreateFocalLengthAttr(48.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_position))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(camera_position, target, [0.0, 0.0, 1.0])
    ))

    pink_material = UsdShade.Material.Define(stage, root + "/PinkReferenceMaterial")
    pink_shader = UsdShade.Shader.Define(stage, root + "/PinkReferenceMaterial/Shader")
    pink_shader.CreateIdAttr("UsdPreviewSurface")
    pink_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.01, 0.38)
    )
    pink_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(5.0, 0.02, 1.8)
    )
    pink_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08)
    pink_material.CreateSurfaceOutput().ConnectToSource(
        pink_shader.ConnectableAPI(), "surface"
    )
    for index, (point, normal) in enumerate(points_and_normals, start=1):
        camera_direction = camera_position - point
        camera_direction /= np.linalg.norm(camera_direction)
        tangent_component = camera_direction - np.dot(camera_direction, normal) * normal
        object_direction = -tangent_component + np.dot(camera_direction, normal) * normal
        object_direction /= np.linalg.norm(object_direction)
        sphere_position = point + 0.030 * object_direction
        sphere = UsdGeom.Sphere.Define(stage, root + f"/PinkReferenceSphere_{index:02d}")
        sphere.CreateRadiusAttr(0.005)
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*sphere_position))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(pink_material)
        # USD PreviewSurface emission is not consistently visible in RTX Real-Time
        # reflections.  A co-located physical sphere light makes the specular pink
        # response deterministic while remaining a GUI-only inspection aid.
        marker_light = UsdLux.SphereLight.Define(
            stage, root + f"/PinkReflectionLight_{index:02d}"
        )
        marker_light.CreateRadiusAttr(0.005)
        marker_light.CreateIntensityAttr(75.0)
        marker_light.CreateColorAttr(Gf.Vec3f(1.0, 0.01, 0.30))
        UsdGeom.Xformable(marker_light).AddTranslateOp().Set(
            Gf.Vec3d(*sphere_position)
        )
    return camera.GetPath()


def main():
    import carb.settings
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from scripts.clearcoat_material import create_clearcoat_material, set_clearcoat_roughness
    from scripts.distributed_roughness import (
        create_continuous_severity_map,
        create_distributed_scratch_normal_map,
        create_severity_grid,
        save_severity_grid,
    )
    from scripts.gloss_geometry import look_at_quaternion, measurement_pose
    from scripts.reflection_measurement import measure_roi, rgb_array, save_capture
    from scripts.validate_curved_local_20 import angle_deg, surface
    from scripts.masked_defect_material import bind_masked_clearcoat_material

    output_dir = TEST_ROOT / "results" / args.tag
    image_dir = output_dir / "images"
    raw_dir = output_dir / "raw"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/CurvedGlossTest")

    material_profile = get_clearcoat_profile(args.material_profile)
    mesh = create_analytic_mesh(
        stage, UsdGeom, Gf, Sdf,
        f"/World/CurvedGlossTest/{args.surface_profile.capitalize()}Panel",
        args.scan_span_m + 2.0 * args.edge_margin_m,
        args.mesh_resolution, surface, args.surface_profile,
    )
    severity = None
    if args.distributed_roughness:
        pristine_cell = (3, 3)
        residual_factor = 1.0 if args.distributed_roughness == "initial" else 0.08
        scratch_strength = 1.2 if args.distributed_roughness == "initial" else 0.08
        severity = create_severity_grid(5, args.roughness_seed, pristine_cell)
        asset_dir = output_dir / "assets"
        save_severity_grid(asset_dir, severity, args.roughness_seed, pristine_cell)
        mask_path, severity_field = create_continuous_severity_map(
            asset_dir / "roughness_severity_map.png", severity,
            residual_factor=residual_factor, pristine_cell=pristine_cell,
        )
        normal_map_path = create_distributed_scratch_normal_map(
            asset_dir / "scratch_normal_map.png", severity_field,
            seed=args.roughness_seed, strength=scratch_strength,
        )
        bind_masked_clearcoat_material(
            stage, mesh, "/World/CurvedGlossTest/DistributedClearcoatMaterial",
            mask_path, material_profile.BASE_COLOR,
            material_profile.BASE_ROUGHNESS, material_profile.CLEARCOAT_WEIGHT,
            args.roughness, 0.35, material_profile.IOR, normal_map_path,
        )
    else:
        material, shader = create_clearcoat_material(
            stage, "/World/CurvedGlossTest/ClearcoatMaterial", material_profile
        )
        set_clearcoat_roughness(shader, args.roughness)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    center_point, center_normal, center_tangent = surface(
        args.surface_profile, 0.0, 0.0
    )
    rig = measurement_pose(
        center_point, center_normal, cfg.INCIDENT_ANGLE_DEG,
        cfg.RIG_DISTANCE_M, tangent_hint=center_tangent,
    )
    light = UsdLux.RectLight.Define(stage, "/World/CurvedGlossTest/LightSource")
    light.CreateWidthAttr(cfg.LIGHT_WIDTH_M)
    light.CreateHeightAttr(cfg.LIGHT_HEIGHT_M)
    light.CreateIntensityAttr(cfg.LIGHT_INTENSITY)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddTranslateOp().Set(Gf.Vec3d(*rig["light_position"]))
    light_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(rig["light_position"], rig["point"], rig["bitangent"])
    ))

    camera = UsdGeom.Camera.Define(stage, "/World/CurvedGlossTest/Camera")
    camera.CreateFocalLengthAttr(50.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*rig["camera_position"]))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(rig["camera_position"], rig["point"], rig["bitangent"])
    ))
    camera_prim = camera.GetPrim()
    camera_prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
    camera_prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
    for name, value in {
        "exposure": 0.0, "exposure:fStop": 1.0, "exposure:iso": 0.0,
        "exposure:responsivity": 1.0, "exposure:time": 1.0,
    }.items():
        camera_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    camera_prim.CreateAttribute(
        "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool
    ).Set(False)

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", cfg.PATH_TRACING_SPP)
    settings.set("/rtx/pathtracing/totalSpp", cfg.PATH_TRACING_SPP)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)
    render_product = rep.create.render_product(camera.GetPath(), cfg.RESOLUTION)
    hdr_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
    hdr_annotator.attach(render_product)
    rep.orchestrator.set_capture_on_play(False)

    coordinates = np.linspace(
        -args.scan_span_m / 2.0, args.scan_span_m / 2.0, args.grid
    )
    light_ops = light_xform.GetOrderedXformOps()
    camera_ops = camera_xform.GetOrderedXformOps()
    rows = []
    total = args.grid * args.grid
    print("")
    print("=" * 76)
    print(f"{args.surface_profile} 곡면 RTX Path Tracing Local 20° 실제 반사 측정")
    surface_label = "distributed-defect" if args.distributed_roughness else "uniform"
    print(
        f"측정 대상: {surface_label} clearcoat {args.surface_profile} mesh, "
        f"base roughness={args.roughness:.3f}"
    )
    print(f"재질 프로필: {material_profile.display_name}")
    print(
        f"문헌 기준  : pristine Ra={material_profile.pristine_ra_um:.4f} um, "
        f"high-gloss anchor={material_profile.high_gloss_anchor_20deg_gu:.1f} GU"
    )
    if args.distributed_roughness:
        print(
            f"표면 상태: 분산 결함 {args.distributed_roughness}, "
            f"seed={args.roughness_seed}, 중앙 (3,3)은 자연 정상부"
        )
    print(
        f"패널/측정 폭: {(args.scan_span_m + 2.0 * args.edge_margin_m) * 1000:.0f} mm / "
        f"{args.scan_span_m * 1000:.0f} mm (가장자리 {args.edge_margin_m * 1000:.0f} mm 제외)"
    )
    print(f"측정 위치: {args.grid}x{args.grid} = {total}개")
    print("측정값   : HDR 정반사 ROI 평균 및 중앙 대비 상대광택 (GU 아님)")
    print("=" * 76)

    # A textured material can return an empty first HdrColor payload while RTX
    # compiles it. Move to a sacrificial pose, render, read, and discard it so
    # the real 5x5 sequence always starts from a valid payload.
    warmup_point, warmup_normal, warmup_tangent = surface(
        args.surface_profile, 0.001, 0.0
    )
    warmup_rig = measurement_pose(
        warmup_point, warmup_normal, cfg.INCIDENT_ANGLE_DEG,
        cfg.RIG_DISTANCE_M, tangent_hint=warmup_tangent,
    )
    light_ops[0].Set(Gf.Vec3d(*warmup_rig["light_position"]))
    light_ops[1].Set(gf_quat(Gf, look_at_quaternion(
        warmup_rig["light_position"], warmup_point, warmup_rig["bitangent"]
    )))
    camera_ops[0].Set(Gf.Vec3d(*warmup_rig["camera_position"]))
    camera_ops[1].Set(gf_quat(Gf, look_at_quaternion(
        warmup_rig["camera_position"], warmup_point, warmup_rig["bitangent"]
    )))
    for _ in range(max(16, cfg.SETTLE_FRAMES * 5)):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    warmup_hdr = rgb_array(hdr_annotator.get_data())
    warmup_value = measure_roi(warmup_hdr, cfg.ROI_FRACTION)["roi_mean_intensity"]
    print(f"[warm-up] 버리는 RTX 준비 프레임 HDR_ROI={warmup_value:.8f}")

    scan_points_and_normals = []
    for index, (row_index, column_index, v, u) in enumerate(
        ((r, c, v, u) for r, v in enumerate(coordinates, 1)
         for c, u in enumerate(coordinates, 1)), start=1
    ):
        point, normal, path_tangent = surface(
            args.surface_profile, float(u), float(v)
        )
        rig = measurement_pose(
            point, normal, cfg.INCIDENT_ANGLE_DEG,
            cfg.RIG_DISTANCE_M, tangent_hint=path_tangent,
        )
        scan_points_and_normals.append((point.copy(), normal.copy()))
        light_ops[0].Set(Gf.Vec3d(*rig["light_position"]))
        light_ops[1].Set(gf_quat(
            Gf, look_at_quaternion(rig["light_position"], point, rig["bitangent"])
        ))
        camera_ops[0].Set(Gf.Vec3d(*rig["camera_position"]))
        camera_ops[1].Set(gf_quat(
            Gf, look_at_quaternion(rig["camera_position"], point, rig["bitangent"])
        ))
        hdr_image = None
        metrics = None
        for attempt in range(1, 4):
            settle_frames = max(6, cfg.SETTLE_FRAMES * 2) if index == 1 else cfg.SETTLE_FRAMES
            for _ in range(settle_frames):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
            hdr_image = rgb_array(hdr_annotator.get_data())
            metrics = measure_roi(hdr_image, cfg.ROI_FRACTION)
            if metrics["roi_mean_intensity"] > 0.0:
                break
            print(f"  [{index:02d}/{total}] HDR=0, renderer warmup retry {attempt}/3")
        preview = np.clip(
            hdr_image.astype(np.float32) * cfg.PNG_PREVIEW_EXPOSURE_SCALE, 0.0, 1.0
        )
        stem = f"row_{row_index:02d}_col_{column_index:02d}"
        save_capture(
            preview, image_dir / f"{stem}.png", raw_dir / f"{stem}_preview.npy",
            metrics["roi_bounds"],
        )
        np.save(raw_dir / f"{stem}_hdr.npy", hdr_image)
        incident_angle = angle_deg(rig["light_direction_from_surface"], normal)
        detection_angle = angle_deg(rig["detector_direction_from_surface"], normal)
        rows.append({
            "grid_row": row_index,
            "grid_column": column_index,
            "u_m": float(u), "v_m": float(v),
            "position_x_m": float(point[0]), "position_y_m": float(point[1]),
            "position_z_m": float(point[2]),
            "normal_x": float(normal[0]), "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "incident_angle_deg": incident_angle,
            "detection_angle_deg": detection_angle,
            "clearcoat_roughness": args.roughness,
            "surface_profile": args.surface_profile,
            "material_profile": material_profile.profile_id,
            "initial_severity": (
                float(severity[row_index - 1, column_index - 1])
                if severity is not None else 0.0
            ),
            "hdr_roi_mean_intensity": metrics["roi_mean_intensity"],
            "hdr_roi_spatial_std": metrics["roi_std_intensity"],
            "hdr_roi_peak_intensity": metrics["roi_peak_intensity"],
        })
        print(
            f"  [{index:02d}/{total}] cell=({row_index},{column_index}) "
            f"z={point[2] * 1000:+6.2f} mm, normal=({normal[0]:+.3f},"
            f"{normal[1]:+.3f},{normal[2]:+.3f}), "
            f"입사/검출={incident_angle:.3f}°/{detection_angle:.3f}°, "
            f"HDR_ROI={metrics['roi_mean_intensity']:.8f}"
        )

    center = next(
        row for row in rows
        if row["grid_row"] == args.grid // 2 + 1
        and row["grid_column"] == args.grid // 2 + 1
    )
    reference = center["hdr_roi_mean_intensity"]
    if reference <= 0.0:
        raise RuntimeError("Center HDR reference is zero")
    for row in rows:
        row["relative_gloss_to_center_not_gu"] = row["hdr_roi_mean_intensity"] / reference

    csv_path = output_dir / "curved_rtx_gloss_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([row["hdr_roi_mean_intensity"] for row in rows])
    relative = np.asarray([row["relative_gloss_to_center_not_gu"] for row in rows])
    summary = {
        "mode": f"{args.surface_profile}_curved_mesh_rtx_path_tracing_local_20",
        "surface_profile": args.surface_profile,
        "material_profile": material_profile.metadata(),
        "surface_state": args.distributed_roughness or "uniform",
        "renderer": "RTX Path Tracing",
        "measurement_aov": "HdrColor",
        "sample_count": len(rows),
        "is_gu": False,
        "clearcoat_roughness": args.roughness,
        "panel_span_m": args.scan_span_m + 2.0 * args.edge_margin_m,
        "scan_span_m": args.scan_span_m,
        "edge_margin_m": args.edge_margin_m,
        "hdr_mean": float(values.mean()),
        "hdr_min": float(values.min()),
        "hdr_max": float(values.max()),
        "relative_to_center_min": float(relative.min()),
        "relative_to_center_max": float(relative.max()),
        "uniform_surface_relative_tolerance": 0.03,
        "max_absolute_relative_deviation_from_center": float(
            np.max(np.abs(relative - 1.0))
        ),
        "all_values_finite_and_positive": bool(
            np.all(np.isfinite(values)) and np.all(values > 0.0)
        ),
        "max_incident_angle_error_deg": max(
            abs(row["incident_angle_deg"] - cfg.INCIDENT_ANGLE_DEG) for row in rows
        ),
        "max_detection_angle_error_deg": max(
            abs(row["detection_angle_deg"] - cfg.INCIDENT_ANGLE_DEG) for row in rows
        ),
    }
    summary["passed"] = (
        summary["all_values_finite_and_positive"]
        and summary["max_incident_angle_error_deg"] < 1e-6
        and summary["max_detection_angle_error_deg"] < 1e-6
        and (
            args.distributed_roughness is not None
            or summary["max_absolute_relative_deviation_from_center"]
            <= summary["uniform_surface_relative_tolerance"]
        )
    )
    summary_path = output_dir / "curved_rtx_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_path = output_dir / "curved_rtx_relative_gloss_heatmap.png"
    save_heatmap(plot_path, rows, args.grid, args.surface_profile)
    stage.Export(str(output_dir / "curved_rtx_scene.usda"))
    print("")
    print("-" * 76)
    positive_count = int(np.sum(np.isfinite(values) & (values > 0.0)))
    print(f"실제 측정 완료: HDR 양수·유한값 {positive_count}/{len(rows)}")
    print(f"HDR 범위      : {values.min():.8f} ~ {values.max():.8f}")
    print(f"상대광택 범위 : {relative.min():.6f} ~ {relative.max():.6f} (중앙=1.0, GU 아님)")
    if args.distributed_roughness is None:
        print(
            f"균일면 최대편차: {summary['max_absolute_relative_deviation_from_center'] * 100:.3f}% "
            f"(허용 {summary['uniform_surface_relative_tolerance'] * 100:.1f}% 이내)"
        )
    print(f"최종 판정     : {'통과' if summary['passed'] else '실패'}")
    print(f"CSV           : {csv_path}")
    print(f"요약 JSON     : {summary_path}")
    print(f"히트맵        : {plot_path}")
    print("-" * 76)
    if not summary["passed"]:
        raise RuntimeError(f"Curved RTX scan failed; see {summary_path}")

    if args.keep_open:
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        light.GetIntensityAttr().Set(0.0)
        overview_path = add_bright_overview(
            stage, scan_points_and_normals, Gf, Sdf, UsdGeom, UsdLux,
            UsdShade, look_at_quaternion,
        )
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(str(overview_path))
        for _ in range(16):
            simulation_app.update()
        print("GUI: RTX Real-Time 2.0 + 고대비 전체 곡면 overview")
        print("회색 배경·분홍 기준구/구형광원은 측정 완료 후에만 추가되어 수치에 영향 없음")
        print("분홍 기준구 반사의 흐림 차이로 결함 분포를 확인할 수 있습니다.")
        print("창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            simulation_app.update()


run_status_path = TEST_ROOT / "results" / args.tag / "run_status.json"
failure = False
try:
    main()
except Exception:
    traceback.print_exc()
    failure = True
finally:
    run_status_path.parent.mkdir(parents=True, exist_ok=True)
    run_status_path.write_text(json.dumps({"success": not failure}, indent=2), encoding="utf-8")
    simulation_app.close()

if failure:
    raise SystemExit(1)
