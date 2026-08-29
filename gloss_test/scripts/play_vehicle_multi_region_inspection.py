#!/usr/bin/env python3
"""Play the completed BMW multi-region inspection as a visible light animation."""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from config import gloss_config as cfg
from config.vehicle_region_profiles import VEHICLE_REGION_PROFILES


REGION_ORDER = tuple(VEHICLE_REGION_PROFILES)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dwell-seconds", type=float, default=0.65)
    parser.add_argument("--transition-seconds", type=float, default=0.30)
    parser.add_argument("--region-pause-seconds", type=float, default=0.8)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.asset.is_file():
        parser.error(f"--asset not found: {args.asset}")
    if not args.cells_csv.is_file():
        parser.error(f"--cells-csv not found: {args.cells_csv}")
    for name in ("dwell_seconds", "transition_seconds", "region_pause_seconds"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RealTimePathTracing",
    "width": 1280,
    "height": 720,
})


def gf_quat(Gf, quaternion):
    return Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(*[float(value) for value in quaternion[1:]]),
    )


def load_cells(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cells = []
    for region_id in REGION_ORDER:
        region_rows = [row for row in rows if row["region_id"] == region_id]
        region_rows.sort(key=lambda row: (int(row["grid_row"]), int(row["grid_column"])))
        if len(region_rows) != 25:
            raise ValueError(f"{region_id}: expected 25 cells, found {len(region_rows)}")
        for row in region_rows:
            cells.append({
                "region_id": region_id,
                "region_label": VEHICLE_REGION_PROFILES[region_id]["label"],
                "grid_row": int(row["grid_row"]),
                "grid_column": int(row["grid_column"]),
                "point": np.asarray([
                    float(row["position_x_m"]),
                    float(row["position_y_m"]),
                    float(row["position_z_m"]),
                ]),
                "normal": np.asarray([
                    float(row["normal_x"]),
                    float(row["normal_y"]),
                    float(row["normal_z"]),
                ]),
                "tangent_hint": np.asarray(
                    VEHICLE_REGION_PROFILES[region_id]["axis_u"], dtype=float
                ),
            })
    return cells


def wait_with_updates(seconds):
    deadline = time.perf_counter() + seconds
    while simulation_app.is_running() and time.perf_counter() < deadline:
        simulation_app.update()


def main():
    import carb.settings
    import omni.replicator.core as rep
    import omni.ui as ui
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    from scripts.gloss_geometry import look_at_quaternion, measurement_pose
    from scripts.reflection_measurement import rgb_array

    cells = load_cells(args.cells_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    vehicle = stage.DefinePrim("/World/Vehicle", "Xform")
    vehicle.GetReferences().AddReference(str(args.asset.resolve()))
    stage.Load()
    for _ in range(6):
        simulation_app.update()

    disabled_lights = 0
    for prim in Usd.PrimRange(vehicle):
        if prim.HasAPI(UsdLux.LightAPI):
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(0.0)
            disabled_lights += 1

    root = "/World/InspectionPlayback"
    UsdGeom.Xform.Define(stage, root)

    ground_material = UsdShade.Material.Define(stage, root + "/GroundMaterial")
    ground_shader = UsdShade.Shader.Define(stage, root + "/GroundMaterial/Shader")
    ground_shader.CreateIdAttr("UsdPreviewSurface")
    ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.035, 0.045, 0.060)
    )
    ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    ground_material.CreateSurfaceOutput().ConnectToSource(
        ground_shader.ConnectableAPI(), "surface"
    )
    ground = UsdGeom.Cube.Define(stage, root + "/Ground")
    ground.CreateSizeAttr(1.0)
    ground_xform = UsdGeom.Xformable(ground)
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.04))
    ground_xform.AddScaleOp().Set(Gf.Vec3d(5.0, 5.0, 0.06))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

    environment = UsdLux.DomeLight.Define(stage, root + "/DimEnvironment")
    # Preserve only a faint silhouette; the moving inspection source should
    # create the dominant visible illumination.
    environment.CreateIntensityAttr(0.0)
    environment.CreateColorAttr(Gf.Vec3f(0.10, 0.13, 0.18))

    soft_fill = UsdLux.RectLight.Define(stage, root + "/SoftFill")
    soft_fill.CreateWidthAttr(3.0)
    soft_fill.CreateHeightAttr(3.0)
    soft_fill.CreateIntensityAttr(0.0)
    soft_fill.CreateColorAttr(Gf.Vec3f(0.42, 0.52, 0.70))
    fill_position = np.asarray([-2.2, -2.0, 2.8])
    fill_target = np.asarray([0.0, 0.0, 0.45])
    fill_xform = UsdGeom.Xformable(soft_fill)
    fill_xform.AddTranslateOp().Set(Gf.Vec3d(*fill_position))
    fill_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(fill_position, fill_target, [0.0, 0.0, 1.0])
    ))

    key_light = UsdLux.RectLight.Define(stage, root + "/VehicleKeyLight")
    key_light.CreateWidthAttr(2.4)
    key_light.CreateHeightAttr(2.4)
    key_light.CreateIntensityAttr(0.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.88, 0.72))
    key_position = np.asarray([2.2, -2.6, 3.0])
    key_xform = UsdGeom.Xformable(key_light)
    key_xform.AddTranslateOp().Set(Gf.Vec3d(*key_position))
    key_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(key_position, fill_target, [0.0, 0.0, 1.0])
    ))

    inspection_light = UsdLux.RectLight.Define(stage, root + "/MovingInspectionLight")
    inspection_light.CreateWidthAttr(0.080)
    inspection_light.CreateHeightAttr(0.080)
    inspection_light.CreateIntensityAttr(250000.0)
    inspection_light.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.42))
    light_xform = UsdGeom.Xformable(inspection_light)
    light_translate = light_xform.AddTranslateOp()
    light_orient = light_xform.AddOrientOp()

    source_material = UsdShade.Material.Define(stage, root + "/SourceMaterial")
    source_shader = UsdShade.Shader.Define(stage, root + "/SourceMaterial/Shader")
    source_shader.CreateIdAttr("UsdPreviewSurface")
    source_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.55, 0.03)
    )
    source_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(8.0, 2.8, 0.08)
    )
    source_material.CreateSurfaceOutput().ConnectToSource(
        source_shader.ConnectableAPI(), "surface"
    )
    source_marker = UsdGeom.Sphere.Define(stage, root + "/VisibleLightSource")
    source_marker.CreateRadiusAttr(0.026)
    source_xform = UsdGeom.Xformable(source_marker)
    source_translate = source_xform.AddTranslateOp()
    UsdShade.MaterialBindingAPI.Apply(source_marker.GetPrim()).Bind(source_material)

    target_material = UsdShade.Material.Define(stage, root + "/TargetMaterial")
    target_shader = UsdShade.Shader.Define(stage, root + "/TargetMaterial/Shader")
    target_shader.CreateIdAttr("UsdPreviewSurface")
    target_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.08, 0.01)
    )
    target_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(5.0, 0.05, 0.01)
    )
    target_material.CreateSurfaceOutput().ConnectToSource(
        target_shader.ConnectableAPI(), "surface"
    )
    target_marker = UsdGeom.Sphere.Define(stage, root + "/ActiveTarget")
    target_marker.CreateRadiusAttr(0.010)
    target_xform = UsdGeom.Xformable(target_marker)
    target_translate = target_xform.AddTranslateOp()
    UsdShade.MaterialBindingAPI.Apply(target_marker.GetPrim()).Bind(target_material)

    camera_position = np.asarray([2.75, -3.75, 2.15])
    camera_target = np.asarray([0.0, -0.05, 0.48])
    camera = UsdGeom.Camera.Define(stage, root + "/OverviewCamera")
    camera.CreateFocalLengthAttr(52.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    camera_prim = camera.GetPrim()
    camera_prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
    camera_prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
    camera_prim.CreateAttribute(
        "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool
    ).Set(False)
    camera_prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(1.35)
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_position))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(camera_position, camera_target, [0.0, 0.0, 1.0])
    ))

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim GUI viewport를 찾지 못했습니다.")
    viewport.set_active_camera(str(camera.GetPath()))

    status_window = ui.Window("검사 위치", width=400, height=145)
    with status_window.frame:
        with ui.VStack(spacing=5):
            title_label = ui.Label("BMW Z4 다중 영역 광택 검사 시각화")
            region_label_ui = ui.Label("영역 준비 중")
            cell_label_ui = ui.Label("셀 준비 중")
            angle_label_ui = ui.Label("Local 입사각 20° (검출기는 화면에서 숨김)")

    def pose_for(cell):
        return measurement_pose(
            cell["point"], cell["normal"], cfg.INCIDENT_ANGLE_DEG,
            cfg.RIG_DISTANCE_M, tangent_hint=cell["tangent_hint"],
        )

    def set_visual_pose(light_position, target_point, up_hint, normal):
        light_translate.Set(Gf.Vec3d(*[float(value) for value in light_position]))
        light_orient.Set(gf_quat(
            Gf, look_at_quaternion(light_position, target_point, up_hint)
        ))
        source_translate.Set(Gf.Vec3d(*[float(value) for value in light_position]))
        marker_point = target_point + 0.008 * normal
        target_translate.Set(Gf.Vec3d(*[float(value) for value in marker_point]))

    first = cells[0]
    first_pose = pose_for(first)
    set_visual_pose(
        first_pose["light_position"], first["point"],
        first_pose["bitangent"], first["normal"],
    )
    for _ in range(16):
        simulation_app.update()

    # Capture and immediately release a verification render product so the
    # live GUI keeps only one viewport and remains responsive.
    capture_product = rep.create.render_product(camera.GetPath(), (1280, 720))
    capture_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
    capture_annotator.attach(capture_product)
    for _ in range(10):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    capture = rgb_array(capture_annotator.get_data())
    if np.issubdtype(capture.dtype, np.integer):
        capture_u8 = capture.astype(np.uint8)
    else:
        capture_u8 = np.round(np.clip(capture, 0.0, 1.0) * 255).astype(np.uint8)
    from PIL import Image
    capture_path = args.output_dir / "inspection_playback_first_frame.png"
    Image.fromarray(capture_u8, "RGB").save(capture_path)
    capture_annotator.detach([capture_product])
    capture_product.destroy()
    for _ in range(6):
        simulation_app.update()

    metadata = {
        "mode": "visual_playback_of_existing_actual_rtx_measurement_points",
        "asset": str(args.asset.resolve()),
        "source_cells_csv": str(args.cells_csv.resolve()),
        "region_order": list(REGION_ORDER),
        "sample_count": len(cells),
        "renderer": "RTX Real-Time 2.0",
        "moving_light_angle_deg": cfg.INCIDENT_ANGLE_DEG,
        "detector_visible": False,
        "display_only_spotlight": True,
        "fixed_key_light_intensity": 0.0,
        "fixed_fill_light_intensity": 0.0,
        "silhouette_dome_light_intensity": 0.0,
        "inspection_light_type": "moving_rect_light",
        "inspection_light_width_m": 0.080,
        "inspection_light_height_m": 0.080,
        "inspection_light_intensity": 250000.0,
        "inspection_light_cone_deg": None,
        "changes_measurement_results": False,
        "first_frame": str(capture_path),
    }
    (args.output_dir / "inspection_playback_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    luminance = (
        0.2126 * capture_u8[:, :, 0]
        + 0.7152 * capture_u8[:, :, 1]
        + 0.0722 * capture_u8[:, :, 2]
    )
    print("")
    print("=" * 80)
    print("BMW Z4 다중 영역 검사 시각화 플레이어")
    print(f"측정점       : {len(cells)}개, {len(REGION_ORDER)}영역")
    print("재생 순서     : " + " -> ".join(REGION_ORDER))
    print("광원          : 각 표면 법선 기준 Local 20° 위치로 이동")
    print("검출기        : 계산에는 존재하지만 화면에서는 숨김")
    print("주황 작은 점   : 현재 검사 위치")
    print("노란 발광 구   : 현재 검사 광원 위치")
    print("좁은 밝은 영역 : 시각화 전용 보조광, 기존 측정 결과에 영향 없음")
    print(
        f"첫 프레임 검증: luminance min={luminance.min():.1f}, "
        f"mean={luminance.mean():.1f}, max={luminance.max():.1f}"
    )
    print(f"첫 프레임 PNG : {capture_path}")
    print("창을 닫으면 종료됩니다.")
    print("=" * 80)

    current_light = np.asarray(first_pose["light_position"], dtype=float)
    current_target = np.asarray(first["point"], dtype=float)
    current_normal = np.asarray(first["normal"], dtype=float)
    pass_index = 0
    while simulation_app.is_running():
        pass_index += 1
        for index, cell in enumerate(cells, start=1):
            if not simulation_app.is_running():
                break
            pose = pose_for(cell)
            next_light = np.asarray(pose["light_position"], dtype=float)
            next_target = np.asarray(cell["point"], dtype=float)
            next_normal = np.asarray(cell["normal"], dtype=float)
            transition_start = time.perf_counter()
            while simulation_app.is_running():
                elapsed = time.perf_counter() - transition_start
                alpha = 1.0 if args.transition_seconds == 0.0 else min(
                    elapsed / args.transition_seconds, 1.0
                )
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                light_position = (1.0 - smooth) * current_light + smooth * next_light
                target_point = (1.0 - smooth) * current_target + smooth * next_target
                normal = (1.0 - smooth) * current_normal + smooth * next_normal
                normal /= np.linalg.norm(normal)
                set_visual_pose(light_position, target_point, pose["bitangent"], normal)
                simulation_app.update()
                if alpha >= 1.0:
                    break
            current_light = next_light
            current_target = next_target
            current_normal = next_normal
            region_label_ui.text = (
                f"현재 영역: {cell['region_label']} ({cell['region_id']})"
            )
            cell_label_ui.text = (
                f"현재 셀: ({cell['grid_row']}, {cell['grid_column']})  "
                f"전체 {index}/{len(cells)}  반복 {pass_index}"
            )
            print(
                f"[visual] pass={pass_index}, region={cell['region_id']}, "
                f"cell=({cell['grid_row']},{cell['grid_column']}), "
                f"light=({next_light[0]:+.3f},{next_light[1]:+.3f},"
                f"{next_light[2]:+.3f})",
                flush=True,
            )
            wait_with_updates(args.dwell_seconds)
            if index < len(cells) and cells[index]["region_id"] != cell["region_id"]:
                wait_with_updates(args.region_pause_seconds)
        if not args.loop:
            print("1회 재생 완료. 창을 닫으면 종료됩니다.")
            while simulation_app.is_running():
                simulation_app.update()
            break


failed = False
try:
    main()
except Exception:
    traceback.print_exc()
    failed = True
finally:
    simulation_app.close()

if failed:
    raise SystemExit(1)
