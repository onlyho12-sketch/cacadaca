#!/usr/bin/env python3
"""Run the isolated Isaac Sim 6.0.1 20-degree relative gloss sweep."""

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

# Load our configuration before SimulationApp imports OpenCV's own top-level
# module named ``config`` into sys.modules.
from config import gloss_config as cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", default="0,0,1", help="surface normal as x,y,z")
    parser.add_argument("--tag", default="normal_z", help="result subdirectory name")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=cfg.REPEAT_COUNT)
    parser.add_argument(
        "--spatial-grid",
        type=int,
        default=0,
        help="run an odd NxN flat-panel spatial scan instead of a roughness sweep",
    )
    parser.add_argument("--scan-roughness", type=float, default=0.10)
    parser.add_argument("--edge-margin-m", type=float, default=0.020)
    parser.add_argument(
        "--defect-cell",
        default=None,
        help="optional 1-based row,column containing a physical rough-clearcoat patch",
    )
    parser.add_argument("--defect-roughness", type=float, default=0.30)
    parser.add_argument("--defect-size-m", type=float, default=0.030)
    parser.add_argument("--scratch-strength", type=float, default=2.8)
    parser.add_argument(
        "--distributed-roughness",
        choices=("initial", "improved"),
        default=None,
        help="use deterministic varied whole-panel roughness, before or after polishing",
    )
    parser.add_argument("--roughness-seed", type=int, default=20260827)
    parser.add_argument(
        "--expect-defect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require defect detection, or with --no-expect-defect require >=95% recovery",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep the Isaac Sim GUI open after measurement; requires --no-headless",
    )
    parsed = parser.parse_args()
    if parsed.repeats < 1:
        parser.error("--repeats must be at least 1")
    if parsed.keep_open and parsed.headless:
        parser.error("--keep-open requires --no-headless")
    if parsed.spatial_grid and (parsed.spatial_grid < 3 or parsed.spatial_grid % 2 == 0):
        parser.error("--spatial-grid must be zero or an odd integer of at least 3")
    if not 0.0 <= parsed.scan_roughness <= 1.0:
        parser.error("--scan-roughness must be between 0 and 1")
    if not 0.0 <= parsed.defect_roughness <= 1.0:
        parser.error("--defect-roughness must be between 0 and 1")
    if parsed.defect_cell and not parsed.spatial_grid:
        parser.error("--defect-cell requires --spatial-grid")
    if parsed.distributed_roughness and parsed.spatial_grid != 5:
        parser.error("--distributed-roughness requires --spatial-grid 5")
    if parsed.distributed_roughness and parsed.defect_cell:
        parser.error("--distributed-roughness cannot be combined with --defect-cell")
    if parsed.defect_size_m <= 0.0:
        parser.error("--defect-size-m must be positive")
    if parsed.scratch_strength < 0.0:
        parser.error("--scratch-strength must be non-negative")
    return parsed


def parse_vector(text):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 3:
        raise ValueError("--normal must contain x,y,z")
    return np.asarray(values, dtype=float)


args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "renderer": "PathTracing",
    "width": 512,
    "height": 512,
    "samples_per_pixel_per_frame": 128,
    "denoiser": False,
})


def gf_quat(Gf, quat):
    return Gf.Quatf(float(quat[0]), Gf.Vec3f(*[float(v) for v in quat[1:]]))


def add_gui_overview(
    stage, rig, shader, look_at_quaternion, Gf, Sdf, UsdGeom, UsdLux, UsdShade,
    scan_points=None, show_rig_markers=True,
):
    """Add an observer view after measurement, so it cannot affect saved results."""
    set_gui_roughness = shader.GetInput("clearcoatRoughness")
    if set_gui_roughness:
        set_gui_roughness.Set(0.10)

    gui_root = "/World/GlossTest/GUI_ONLY_PostMeasurement"
    UsdGeom.Xform.Define(stage, gui_root)

    dome = UsdLux.DomeLight.Define(stage, f"{gui_root}/OverviewFillLight")
    dome.CreateIntensityAttr(350.0)
    dome.CreateColorAttr(Gf.Vec3f(0.55, 0.62, 0.75))

    def make_emissive_marker(name, position, color, radius):
        sphere = UsdGeom.Sphere.Define(stage, f"{gui_root}/{name}")
        sphere.CreateRadiusAttr(radius)
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*position))

        material = UsdShade.Material.Define(stage, f"{gui_root}/{name}Material")
        marker_shader = UsdShade.Shader.Define(stage, f"{gui_root}/{name}Material/Shader")
        marker_shader.CreateIdAttr("UsdPreviewSurface")
        marker_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        marker_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        marker_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        material.CreateSurfaceOutput().ConnectToSource(marker_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)

    # Yellow = light, cyan = detector camera, magenta = measurement point.
    if show_rig_markers:
        make_emissive_marker(
            "YELLOW_LightMarker", rig["light_position"], (4.0, 2.8, 0.05), 0.035
        )
        make_emissive_marker(
            "CYAN_CameraMarker", rig["camera_position"], (0.05, 2.5, 4.0), 0.035
        )
    if scan_points is not None:
        for index, point in enumerate(scan_points, start=1):
            marker_position = point + 0.012 * rig["normal"]
            make_emissive_marker(
                f"MAGENTA_ScanPoint_{index:02d}", marker_position, (4.0, 0.05, 1.8), 0.006
            )
    else:
        marker_position = rig["point"] + 0.012 * rig["normal"]
        make_emissive_marker(
            "MAGENTA_MeasurementPoint", marker_position, (4.0, 0.05, 1.8), 0.012
        )

    overview_position = (
        rig["point"]
        - 0.85 * rig["bitangent"]
        + 0.45 * rig["normal"]
    )
    overview = UsdGeom.Camera.Define(stage, f"{gui_root}/OverviewCamera")
    overview.CreateFocalLengthAttr(32.0)
    overview.CreateHorizontalApertureAttr(36.0)
    overview.CreateVerticalApertureAttr(24.0)
    overview.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    overview_xform = UsdGeom.Xformable(overview)
    overview_xform.AddTranslateOp().Set(Gf.Vec3d(*overview_position))
    overview_xform.AddOrientOp().Set(gf_quat(
        Gf,
        look_at_quaternion(overview_position, rig["point"], rig["normal"]),
    ))
    return overview.GetPath()


def add_pink_ball_inspection_view(
    stage, rig, scan_points, look_at_quaternion, Gf, Sdf, UsdGeom, UsdShade
):
    """Reflect a grid of real pink reference spheres at all scan locations."""
    root = "/World/GlossTest/GUI_ONLY_PostMeasurement/PinkBallGridInspection"
    UsdGeom.Xform.Define(stage, root)
    camera_position = rig["point"] - 0.85 * rig["bitangent"] + 0.45 * rig["normal"]
    camera = UsdGeom.Camera.Define(stage, f"{root}/PinkBallGridCamera")
    camera.CreateFocalLengthAttr(55.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_position))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(camera_position, rig["point"], rig["normal"])
    ))

    material = UsdShade.Material.Define(stage, f"{root}/PinkEmissiveMaterial")
    pink_shader = UsdShade.Shader.Define(stage, f"{root}/PinkEmissiveMaterial/Shader")
    pink_shader.CreateIdAttr("UsdPreviewSurface")
    pink_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.01, 0.45)
    )
    pink_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(4.0, 0.02, 1.8)
    )
    pink_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
    material.CreateSurfaceOutput().ConnectToSource(pink_shader.ConnectableAPI(), "surface")

    for index, point in enumerate(scan_points, start=1):
        camera_direction = camera_position - point
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        tangent_component = (
            camera_direction - np.dot(camera_direction, rig["normal"]) * rig["normal"]
        )
        reflected_object_direction = -tangent_component + np.dot(
            camera_direction, rig["normal"]
        ) * rig["normal"]
        reflected_object_direction /= np.linalg.norm(reflected_object_direction)
        sphere_position = point + 0.020 * reflected_object_direction
        sphere = UsdGeom.Sphere.Define(stage, f"{root}/PinkReferenceSphere_{index:02d}")
        sphere.CreateRadiusAttr(0.0045)
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*sphere_position))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    return camera.GetPath()


def main():
    import carb.settings
    import isaacsim.core.version as version_api
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from scripts.clearcoat_material import create_clearcoat_material, set_clearcoat_roughness
    from scripts.gloss_geometry import look_at_quaternion, measurement_pose, quaternion_from_local_z
    from scripts.distributed_roughness import (
        create_continuous_severity_map,
        create_distributed_scratch_normal_map,
        create_severity_grid,
        save_severity_grid,
    )
    from scripts.masked_defect_material import (
        create_masked_clearcoat_overlay,
        create_localized_scratch_normal_map,
        create_soft_square_mask,
    )
    from scripts.reflection_measurement import measure_roi, rgb_array, save_capture
    from scripts.spatial_scan import run_spatial_scan

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
    UsdGeom.Xform.Define(stage, "/World/GlossTest")

    normal = parse_vector(args.normal)
    rig = measurement_pose(np.zeros(3), normal, cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M)

    panel = UsdGeom.Cube.Define(stage, "/World/GlossTest/Panel")
    panel.CreateSizeAttr(1.0)
    panel_xform = UsdGeom.Xformable(panel)
    panel_xform.AddOrientOp().Set(gf_quat(Gf, quaternion_from_local_z(rig["normal"])))
    panel_xform.AddScaleOp().Set(Gf.Vec3d(cfg.PANEL_SIZE_M, cfg.PANEL_SIZE_M, cfg.PANEL_THICKNESS_M))

    material, shader = create_clearcoat_material(stage, "/World/GlossTest/ClearcoatMaterial", cfg)
    UsdShade.MaterialBindingAPI.Apply(panel.GetPrim()).Bind(material)

    defect_point = None
    masked_defect_texture = None
    distributed_mode = args.distributed_roughness is not None
    if distributed_mode:
        pristine_cell = (3, 3)
        residual_factor = 1.0 if args.distributed_roughness == "initial" else 0.15
        scratch_strength = 1.2 if args.distributed_roughness == "initial" else 0.18
        severity = create_severity_grid(
            grid_size=5, seed=args.roughness_seed, pristine_cell=pristine_cell
        )
        asset_dir = output_dir / "assets"
        save_severity_grid(asset_dir, severity, args.roughness_seed, pristine_cell)
        mask_path, severity_field = create_continuous_severity_map(
            asset_dir / "roughness_severity_map.png",
            severity,
            residual_factor=residual_factor,
            pristine_cell=pristine_cell,
        )
        normal_map_path = create_distributed_scratch_normal_map(
            asset_dir / "scratch_normal_map.png",
            severity_field,
            seed=args.roughness_seed,
            strength=scratch_strength,
        )
        _, masked_defect_texture = create_masked_clearcoat_overlay(
            stage=stage,
            path="/World/GlossTest/DistributedClearcoatSurface",
            mask_path=mask_path,
            panel_size_m=cfg.PANEL_SIZE_M,
            panel_thickness_m=cfg.PANEL_THICKNESS_M,
            base_color=cfg.BASE_COLOR,
            base_roughness=cfg.BASE_ROUGHNESS,
            clearcoat_weight=cfg.CLEARCOAT_WEIGHT,
            normal_clearcoat_roughness=args.scan_roughness,
            defect_clearcoat_roughness=0.35,
            ior=cfg.IOR,
            orient_quaternion=gf_quat(Gf, quaternion_from_local_z(rig["normal"])),
            normal_map_path=normal_map_path,
        )
        print(
            f"[Distributed Roughness] state={args.distributed_roughness}, "
            f"seed={args.roughness_seed}, pristine_cell={pristine_cell}, "
            f"residual_factor={residual_factor:.2f}, scratch_strength={scratch_strength:.2f}"
        )
    elif args.defect_cell:
        try:
            defect_row, defect_column = [int(value.strip()) for value in args.defect_cell.split(",")]
        except (TypeError, ValueError) as exc:
            raise ValueError("--defect-cell must be formatted as row,column") from exc
        if not (1 <= defect_row <= args.spatial_grid and 1 <= defect_column <= args.spatial_grid):
            raise ValueError("--defect-cell row and column must be inside the spatial grid")
        half_span = cfg.PANEL_SIZE_M / 2.0 - args.edge_margin_m
        grid_coordinates = np.linspace(-half_span, half_span, args.spatial_grid)
        defect_u = grid_coordinates[defect_column - 1]
        defect_v = grid_coordinates[defect_row - 1]
        defect_point = (
            rig["point"] + defect_u * rig["tangent"] + defect_v * rig["bitangent"]
        )
        defect_uv = (
            float(defect_u / cfg.PANEL_SIZE_M + 0.5),
            float(defect_v / cfg.PANEL_SIZE_M + 0.5),
        )
        mask_path = create_soft_square_mask(
            output_dir / "assets" / "roughness_mask.png",
            center_uv=defect_uv,
            size_uv=args.defect_size_m / cfg.PANEL_SIZE_M,
            feather_uv=0.025,
        )
        normal_map_path = create_localized_scratch_normal_map(
            output_dir / "assets" / "scratch_normal_map.png",
            center_uv=defect_uv,
            size_uv=args.defect_size_m / cfg.PANEL_SIZE_M,
            feather_uv=0.025,
            strength=args.scratch_strength,
        )
        _, masked_defect_texture = create_masked_clearcoat_overlay(
            stage=stage,
            path="/World/GlossTest/MaskedClearcoatSurface",
            mask_path=mask_path,
            panel_size_m=cfg.PANEL_SIZE_M,
            panel_thickness_m=cfg.PANEL_THICKNESS_M,
            base_color=cfg.BASE_COLOR,
            base_roughness=cfg.BASE_ROUGHNESS,
            clearcoat_weight=cfg.CLEARCOAT_WEIGHT,
            normal_clearcoat_roughness=args.scan_roughness,
            defect_clearcoat_roughness=args.defect_roughness,
            ior=cfg.IOR,
            orient_quaternion=gf_quat(Gf, quaternion_from_local_z(rig["normal"])),
            normal_map_path=normal_map_path,
        )
        print(
            f"[Defect Test] soft roughness mask: row={defect_row}, col={defect_column}, "
            f"roughness={args.defect_roughness:.3f}, size={args.defect_size_m * 1000:.1f} mm, "
            f"feather=5.0 mm, scratch normal strength={args.scratch_strength:.3f}"
        )

    light = UsdLux.RectLight.Define(stage, "/World/GlossTest/LightSource")
    light.CreateWidthAttr(cfg.LIGHT_WIDTH_M)
    light.CreateHeightAttr(cfg.LIGHT_HEIGHT_M)
    light.CreateIntensityAttr(cfg.LIGHT_INTENSITY)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddTranslateOp().Set(Gf.Vec3d(*rig["light_position"]))
    light_xform.AddOrientOp().Set(gf_quat(
        Gf,
        look_at_quaternion(rig["light_position"], rig["point"], rig["bitangent"]),
    ))

    camera = UsdGeom.Camera.Define(stage, "/World/GlossTest/Camera")
    camera.CreateFocalLengthAttr(50.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*rig["camera_position"]))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf,
        look_at_quaternion(rig["camera_position"], rig["point"], rig["bitangent"]),
    ))
    camera_prim = camera.GetPrim()
    camera_prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
    camera_prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
    for name, value in {
        "exposure": 0.0,
        "exposure:fStop": 1.0,
        "exposure:iso": 0.0,
        "exposure:responsivity": 1.0,
        "exposure:time": 1.0,
    }.items():
        camera_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    camera_prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", cfg.PATH_TRACING_SPP)
    settings.set("/rtx/pathtracing/totalSpp", cfg.PATH_TRACING_SPP)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    render_product = rep.create.render_product(camera.GetPath(), cfg.RESOLUTION, name="GlossRenderProduct")
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)
    hdr_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
    hdr_annotator.attach(render_product)
    rep.orchestrator.set_capture_on_play(False)

    version = getattr(version_api, "__version__", "6.0.1")
    print("[Gloss Test]")
    print(f"Isaac Sim Version : {version}")
    print("Renderer          : RTX Path Tracing")
    print(f"Incident Angle    : {cfg.INCIDENT_ANGLE_DEG:.1f} deg from local normal")
    print(f"Detection Angle   : {cfg.INCIDENT_ANGLE_DEG:.1f} deg from local normal")
    print(f"Surface Normal    : {rig['normal'].tolist()}")
    print(f"Light Position    : {rig['light_position'].tolist()}")
    print(f"Camera Position   : {rig['camera_position'].tolist()}")
    print("Auto Exposure     : OFF")
    print("Measurement AOV   : HdrColor (float, pre-tonemap)")
    print("Preview Output    : fixed-scale HdrColor PNG")
    print(f"Repeat Count      : {args.repeats}")

    if args.spatial_grid:
        scan_points = run_spatial_scan(
            args=args,
            cfg=cfg,
            test_root=TEST_ROOT,
            stage=stage,
            normal=normal,
            center_rig=rig,
            shader=shader,
            light_xform=light_xform,
            camera_xform=camera_xform,
            rep=rep,
            rgb_annotator=rgb_annotator,
            hdr_annotator=hdr_annotator,
            Gf=Gf,
            measurement_pose=measurement_pose,
            look_at_quaternion=look_at_quaternion,
            gf_quat=gf_quat,
            set_clearcoat_roughness=set_clearcoat_roughness,
            measure_roi=measure_roi,
            rgb_array=rgb_array,
            save_capture=save_capture,
        )
        light_ops = light_xform.GetOrderedXformOps()
        camera_ops = camera_xform.GetOrderedXformOps()
        light_ops[0].Set(Gf.Vec3d(*rig["light_position"]))
        light_ops[1].Set(gf_quat(
            Gf, look_at_quaternion(rig["light_position"], rig["point"], rig["bitangent"])
        ))
        camera_ops[0].Set(Gf.Vec3d(*rig["camera_position"]))
        camera_ops[1].Set(gf_quat(
            Gf, look_at_quaternion(rig["camera_position"], rig["point"], rig["bitangent"])
        ))
        if args.keep_open:
            settings.set("/rtx/rendermode", "RealTimePathTracing")
            overview_camera_path = add_gui_overview(
                stage, rig, shader, look_at_quaternion, Gf, Sdf, UsdGeom, UsdLux, UsdShade,
                scan_points=[] if (defect_point is not None or distributed_mode) else scan_points,
                show_rig_markers=defect_point is None and not distributed_mode,
            )
            active_camera_path = overview_camera_path
            if defect_point is not None or distributed_mode:
                # Inspection-only reference gloss. Numerical captures above remain
                # at --scan-roughness and the defect material remains unchanged.
                if defect_point is not None:
                    set_clearcoat_roughness(shader, 0.05)
                    masked_defect_texture.GetInput("scale").Set(Gf.Vec4f(
                        args.defect_roughness - 0.05,
                        args.defect_roughness - 0.05,
                        args.defect_roughness - 0.05,
                        1.0,
                    ))
                    masked_defect_texture.GetInput("bias").Set(
                        Gf.Vec4f(0.05, 0.05, 0.05, 0.0)
                    )
                light.GetIntensityAttr().Set(0.0)
                fill_prim = stage.GetPrimAtPath(
                    "/World/GlossTest/GUI_ONLY_PostMeasurement/OverviewFillLight"
                )
                fill = UsdLux.DomeLight(fill_prim)
                fill.GetIntensityAttr().Set(350.0)
                fill.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
                active_camera_path = add_pink_ball_inspection_view(
                    stage, rig, scan_points, look_at_quaternion, Gf, Sdf, UsdGeom, UsdShade
                )
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac Sim GUI viewport를 찾지 못했습니다.")
            viewport.set_active_camera(str(active_camera_path))
            for _ in range(12):
                simulation_app.update()
            print("[Spatial Scan] GUI renderer: RTX Real-Time 2.0 (post-measurement only)")
            if defect_point is not None:
                print("[Defect Test] physical 5x5 pink-ball reflection grid camera active")
                print("[Defect Test] visual reference only: surrounding roughness 0.05, defect unchanged")
                print("[Defect Test] 정상 24곳보다 결함 1곳의 분홍 반사상만 흐리고 약하게 보입니다.")
            elif distributed_mode:
                print("[Distributed Roughness] physical 5x5 pink-ball reflection grid camera active")
                print("[Distributed Roughness] 분홍 반사상의 선명도 차이로 전체 결함 분포를 확인합니다.")
            else:
                print("[Spatial Scan] 자홍색 점 25개가 실제 패널 측정 위치입니다.")
            print("[Spatial Scan] keep-open: Isaac Sim 창을 닫으면 테스트가 종료됩니다.")
            while simulation_app.is_running():
                simulation_app.update()
        return

    measurement_rows = []
    aggregate_rows = []
    formats_logged = False
    for roughness in cfg.ROUGHNESS_VALUES:
        print(f"[Gloss Test] clearcoat_roughness={roughness:.3f}")
        roughness_measurements = []
        for repeat_index in range(1, args.repeats + 1):
            set_clearcoat_roughness(shader, roughness)
            for _ in range(cfg.SETTLE_FRAMES):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
            ldr_image = rgb_array(rgb_annotator.get_data())
            hdr_image = rgb_array(hdr_annotator.get_data())
            preview_image = np.clip(hdr_image.astype(np.float32) * cfg.PNG_PREVIEW_EXPOSURE_SCALE, 0.0, 1.0)
            if not formats_logged:
                print(
                    f"LDR format         : shape={ldr_image.shape}, dtype={ldr_image.dtype}, "
                    f"range=[{float(ldr_image.min()):.6f}, {float(ldr_image.max()):.6f}]"
                )
                print(
                    f"HDR format         : shape={hdr_image.shape}, dtype={hdr_image.dtype}, "
                    f"range=[{float(hdr_image.min()):.6f}, {float(hdr_image.max()):.6f}]"
                )
                formats_logged = True
            hdr_metrics = measure_roi(hdr_image, cfg.ROI_FRACTION)
            ldr_metrics = measure_roi(ldr_image, cfg.ROI_FRACTION)
            preview_metrics = measure_roi(preview_image, cfg.ROI_FRACTION)
            stem = f"roughness_{roughness:.3f}_rep_{repeat_index:02d}"
            save_capture(
                preview_image,
                image_dir / f"{stem}.png",
                raw_dir / f"{stem}_preview.npy",
                preview_metrics["roi_bounds"],
            )
            np.save(raw_dir / f"{stem}_ldr.npy", ldr_image)
            np.save(raw_dir / f"{stem}_hdr.npy", hdr_image)
            measurement = {
                "clearcoat_roughness": roughness,
                "repeat_index": repeat_index,
                "hdr_roi_mean_intensity": hdr_metrics["roi_mean_intensity"],
                "hdr_roi_spatial_std": hdr_metrics["roi_std_intensity"],
                "hdr_roi_peak_intensity": hdr_metrics["roi_peak_intensity"],
                "ldr_roi_mean_intensity": ldr_metrics["roi_mean_intensity"],
                "ldr_saturated_fraction": ldr_metrics["saturated_fraction"],
                "preview_saturated_fraction": preview_metrics["saturated_fraction"],
            }
            measurement_rows.append(measurement)
            roughness_measurements.append(measurement)
            print(
                f"  repeat {repeat_index:02d}/{args.repeats}: "
                f"HDR_ROI={measurement['hdr_roi_mean_intensity']:.8f}, "
                f"Preview_sat={measurement['preview_saturated_fraction']:.6f}"
            )

        hdr_values = np.asarray(
            [item["hdr_roi_mean_intensity"] for item in roughness_measurements], dtype=float
        )
        aggregate_rows.append({
            "clearcoat_roughness": roughness,
            "incident_angle_deg": cfg.INCIDENT_ANGLE_DEG,
            "detection_angle_deg": cfg.INCIDENT_ANGLE_DEG,
            "surface_normal_x": rig["normal"][0],
            "surface_normal_y": rig["normal"][1],
            "surface_normal_z": rig["normal"][2],
            "light_intensity": cfg.LIGHT_INTENSITY,
            "repeat_count": args.repeats,
            "hdr_roi_mean_intensity": float(hdr_values.mean()),
            "hdr_roi_repeat_std": float(hdr_values.std(ddof=1)) if args.repeats > 1 else 0.0,
            "hdr_roi_repeat_cv": (
                float(hdr_values.std(ddof=1) / hdr_values.mean())
                if args.repeats > 1 and hdr_values.mean() > 0.0 else 0.0
            ),
            "max_preview_saturated_fraction": max(
                item["preview_saturated_fraction"] for item in roughness_measurements
            ),
        })

    reference = aggregate_rows[0]["hdr_roi_mean_intensity"]
    if reference <= 0.0:
        raise RuntimeError("Reference ROI intensity is zero; check Light/Camera/Material geometry")
    for row in aggregate_rows:
        row["reference_roughness"] = cfg.ROUGHNESS_VALUES[0]
        row["relative_gloss"] = row["hdr_roi_mean_intensity"] / reference
        row["relative_gloss_std"] = row["hdr_roi_repeat_std"] / reference

    csv_path = output_dir / "gloss_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    measurement_csv_path = output_dir / "gloss_measurements.csv"
    with measurement_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurement_rows[0].keys()))
        writer.writeheader()
        writer.writerows(measurement_rows)

    stage.Export(str(output_dir / "gloss_test_scene.usda"))
    for row in aggregate_rows:
        print(
            f"Roughness {row['clearcoat_roughness']:.2f} -> "
            f"Relative Gloss {row['relative_gloss']:.6f} ± {row['relative_gloss_std']:.6f} "
            f"(HDR_ROI={row['hdr_roi_mean_intensity']:.8f}, "
            f"CV={row['hdr_roi_repeat_cv']:.6f}, "
            f"Preview_sat={row['max_preview_saturated_fraction']:.6f})"
        )
    print(f"[Gloss Test] CSV saved: {csv_path}")
    print(f"[Gloss Test] raw measurements saved: {measurement_csv_path}")

    relative_values = [float(row["relative_gloss"]) for row in aggregate_rows]
    monotonic = all(
        current <= previous * 1.01
        for previous, current in zip(relative_values, relative_values[1:])
    )
    unsaturated = all(
        float(row["max_preview_saturated_fraction"]) < 0.01 for row in aggregate_rows
    )
    repeatable = all(
        float(row["hdr_roi_repeat_cv"]) <= cfg.MAX_REPEAT_CV for row in aggregate_rows
    )
    validation = {
        "metric_name": "relative_gloss",
        "measurement_aov": "HdrColor",
        "is_gu": False,
        "repeat_count": args.repeats,
        "monotonic_nonincreasing_with_1pct_tolerance": monotonic,
        "all_preview_saturated_fraction_below_1pct": unsaturated,
        "all_repeat_cv_below_2pct": repeatable,
        "passed": monotonic and unsaturated and repeatable,
    }
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"[Gloss Test] validation: {validation}")
    if not validation["passed"]:
        raise RuntimeError(f"Gloss response validation failed; see {validation_path}")

    plot_script = TEST_ROOT / "scripts" / "plot_results.py"
    subprocess.run([sys.executable, str(plot_script), str(csv_path)], check=True)

    if args.keep_open:
        # Measurements above stay on PathTracing. The persistent observer GUI does
        # not need 128 spp and is much more responsive in Isaac Sim 6 Real-Time 2.0.
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        overview_camera_path = add_gui_overview(
            stage, rig, shader, look_at_quaternion, Gf, Sdf, UsdGeom, UsdLux, UsdShade
        )
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim GUI viewport를 찾지 못했습니다.")
        viewport.set_active_camera(str(overview_camera_path))
        for _ in range(12):
            simulation_app.update()
        print("[Gloss Test] GUI renderer switched: RealTimePathTracing (post-measurement only)")
        print(f"[Gloss Test] GUI overview camera active: {overview_camera_path}")
        print("[Gloss Test] 노랑=광원, 청록=센서 카메라, 자홍=측정점, 중앙 사각형=20 cm 패널")
        print("[Gloss Test] 실제 측정 화면은 상단 카메라 메뉴에서 /World/GlossTest/Camera 선택")
        print("[Gloss Test] keep-open: Isaac Sim 창을 닫으면 테스트가 종료됩니다.")
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
    run_status_path.write_text(
        json.dumps({"success": not failure}, indent=2), encoding="utf-8"
    )
    simulation_app.close()
