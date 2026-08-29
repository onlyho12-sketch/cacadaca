#!/usr/bin/env python3
"""Visualize received BMW RL before/after states with the moving inspection light."""

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
PHASES = ("before", "after")
STATE_COLUMNS = (
    "force_n", "rpm", "feed_mm_s", "step_over_ratio", "pass_count",
    "roughness_before", "roughness_after", "scratch_before", "scratch_after",
    "ra_before_um", "ra_after_um", "rz_before_um", "rz_after_um",
    "clearcoat_before_um", "clearcoat_after_um", "clearcoat_removed_um",
    "gu_proxy_before", "gu_proxy_after",
)


def load_cells(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: header is missing")
        required = {
            "region_id", "region_label", "grid_row", "grid_column",
            "position_x_m", "position_y_m", "position_z_m",
            "normal_x", "normal_y", "normal_z", *STATE_COLUMNS,
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing render columns: {missing}")
        rows = list(reader)
    if len(rows) != 150:
        raise ValueError(f"expected 150 render cells, found {len(rows)}")

    cells = []
    seen = set()
    for region_id in REGION_ORDER:
        selected = [row for row in rows if row["region_id"] == region_id]
        selected.sort(key=lambda row: (int(row["grid_row"]), int(row["grid_column"])))
        if len(selected) != 25:
            raise ValueError(f"{region_id}: expected 25 cells, found {len(selected)}")
        for row in selected:
            key = (region_id, int(row["grid_row"]), int(row["grid_column"]))
            if key in seen:
                raise ValueError(f"duplicate render cell: {key}")
            seen.add(key)
            point = np.asarray([float(row[f"position_{axis}_m"]) for axis in "xyz"])
            normal = np.asarray([float(row[f"normal_{axis}"]) for axis in "xyz"])
            normal_length = np.linalg.norm(normal)
            if not np.isfinite(point).all() or not np.isfinite(normal).all():
                raise ValueError(f"non-finite position/normal at {key}")
            if abs(normal_length - 1.0) > 1.0e-3:
                raise ValueError(f"non-unit normal at {key}: {normal_length}")
            values = {name: float(row[name]) for name in STATE_COLUMNS}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite RL state at {key}")
            cells.append({
                "key": key,
                "region_id": region_id,
                "region_label": row.get("region_label") or VEHICLE_REGION_PROFILES[region_id]["label"],
                "grid_row": key[1],
                "grid_column": key[2],
                "point": point,
                "normal": normal / normal_length,
                "tangent_hint": np.asarray(
                    VEHICLE_REGION_PROFILES[region_id]["axis_u"], dtype=float
                ),
                "values": values,
            })
    if len(seen) != 150:
        raise ValueError(f"expected 150 unique cells, found {len(seen)}")
    return cells


def visual_state(cell, phase, clearcoat_safety_limit_um=35.0):
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if clearcoat_safety_limit_um <= 0.0:
        raise ValueError("clearcoat safety limit must be positive")
    values = cell["values"]
    roughness = float(values[f"roughness_{phase}"])
    scratch = float(values[f"scratch_{phase}"])
    clearcoat_um = float(values[f"clearcoat_{phase}_um"])
    if not 0.0 <= roughness <= 1.0 or not 0.0 <= scratch <= 1.0:
        raise ValueError(f"roughness/scratch outside [0,1] at {cell['key']}")
    if clearcoat_um < 0.0:
        raise ValueError(f"negative clearcoat at {cell['key']}")
    return {
        "clearcoat_roughness": roughness,
        # Scratch is shown as isotropic base-layer blur in this display player.
        # Numerical RTX evidence remains a separate Path Tracing measurement.
        "base_roughness": float(np.clip(0.08 + 0.55 * scratch, 0.0, 1.0)),
        "clearcoat_weight": float(np.clip(clearcoat_um / clearcoat_safety_limit_um, 0.0, 1.0)),
        "clearcoat_um": clearcoat_um,
        "gu_proxy": float(values[f"gu_proxy_{phase}"]),
        "ra_um": float(values[f"ra_{phase}_um"]),
        "rz_um": float(values[f"rz_{phase}_um"]),
        "roughness": roughness,
        "scratch": scratch,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after", "cycle"), default="cycle")
    parser.add_argument("--dwell-seconds", type=float, default=0.55)
    parser.add_argument("--transition-seconds", type=float, default=0.25)
    parser.add_argument("--region-pause-seconds", type=float, default=0.7)
    parser.add_argument("--cell-patch-size-m", type=float, default=0.026)
    parser.add_argument("--clearcoat-safety-limit-um", type=float, default=35.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if not args.asset.is_file():
        parser.error(f"--asset not found: {args.asset}")
    if not args.cells_csv.is_file():
        parser.error(f"--cells-csv not found: {args.cells_csv}")
    for name in ("dwell_seconds", "transition_seconds", "region_pause_seconds"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.cell_patch_size_m <= 0.0:
        parser.error("--cell-patch-size-m must be positive")
    if args.clearcoat_safety_limit_um <= 0.0:
        parser.error("--clearcoat-safety-limit-um must be positive")
    if args.headless and args.loop:
        parser.error("--headless requires --no-loop")
    return args


args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "renderer": "RealTimePathTracing",
    "width": 1280,
    "height": 720,
})


def gf_quat(Gf, quaternion):
    return Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(*[float(value) for value in quaternion[1:]]),
    )


def wait_with_updates(seconds):
    deadline = time.perf_counter() + seconds
    while simulation_app.is_running() and time.perf_counter() < deadline:
        simulation_app.update()


def tangent_frame(cell):
    normal = cell["normal"]
    tangent = cell["tangent_hint"] - np.dot(cell["tangent_hint"], normal) * normal
    if np.linalg.norm(tangent) < 1.0e-8:
        fallback = np.asarray([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, normal)) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0])
        tangent = fallback - np.dot(fallback, normal) * normal
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return tangent, bitangent


def main():
    import carb.settings
    import omni.ui as ui
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    from scripts.gloss_geometry import look_at_quaternion, measurement_pose

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

    root = "/World/RLInspectionPlayback"
    UsdGeom.Xform.Define(stage, root)
    patch_root = root + "/ClearcoatCellPatches"
    UsdGeom.Xform.Define(stage, patch_root)
    patch_inputs = []
    half = args.cell_patch_size_m / 2.0
    for index, cell in enumerate(cells):
        tangent, bitangent = tangent_frame(cell)
        center = cell["point"] + 3.5e-4 * cell["normal"]
        points = [
            center - half * tangent - half * bitangent,
            center + half * tangent - half * bitangent,
            center + half * tangent + half * bitangent,
            center - half * tangent + half * bitangent,
        ]
        patch_path = patch_root + f"/Cell_{index + 1:03d}"
        mesh = UsdGeom.Mesh.Define(stage, patch_path)
        mesh.CreatePointsAttr([Gf.Vec3f(*[float(value) for value in point]) for point in points])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(False)

        material_path = patch_path + "Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.33, 0.35, 0.38)
        )
        base_roughness = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
        clearcoat = shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float)
        clearcoat_roughness = shader.CreateInput(
            "clearcoatRoughness", Sdf.ValueTypeNames.Float
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        patch_inputs.append((base_roughness, clearcoat, clearcoat_roughness))

    ground_material = UsdShade.Material.Define(stage, root + "/GroundMaterial")
    ground_shader = UsdShade.Shader.Define(stage, root + "/GroundMaterial/Surface")
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

    inspection_light = UsdLux.RectLight.Define(stage, root + "/MovingInspectionLight")
    inspection_light.CreateWidthAttr(0.080)
    inspection_light.CreateHeightAttr(0.080)
    inspection_light.CreateIntensityAttr(250000.0)
    inspection_light.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.42))
    light_xform = UsdGeom.Xformable(inspection_light)
    light_translate = light_xform.AddTranslateOp()
    light_orient = light_xform.AddOrientOp()

    source_material = UsdShade.Material.Define(stage, root + "/SourceMaterial")
    source_shader = UsdShade.Shader.Define(stage, root + "/SourceMaterial/Surface")
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
    source_translate = UsdGeom.Xformable(source_marker).AddTranslateOp()
    UsdShade.MaterialBindingAPI.Apply(source_marker.GetPrim()).Bind(source_material)

    target_material = UsdShade.Material.Define(stage, root + "/TargetMaterial")
    target_shader = UsdShade.Shader.Define(stage, root + "/TargetMaterial/Surface")
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
    target_translate = UsdGeom.Xformable(target_marker).AddTranslateOp()
    UsdShade.MaterialBindingAPI.Apply(target_marker.GetPrim()).Bind(target_material)

    camera_position = np.asarray([2.75, -3.75, 2.15])
    camera_target = np.asarray([0.0, -0.05, 0.48])
    camera = UsdGeom.Camera.Define(stage, root + "/OverviewCamera")
    camera.CreateFocalLengthAttr(52.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_position))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(camera_position, camera_target, [0.0, 0.0, 1.0])
    ))
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.set_active_camera(str(camera.GetPath()))

    phase_label = region_label = cell_label = quality_label = None
    if not args.headless:
        status_window = ui.Window("RL 검사 상태", width=520, height=205)
        with status_window.frame:
            with ui.VStack(spacing=5):
                ui.Label("BMW Z4 RL 6영역 150셀 Clearcoat 검사")
                phase_label = ui.Label("상태 준비 중")
                region_label = ui.Label("영역 준비 중")
                cell_label = ui.Label("셀 준비 중")
                quality_label = ui.Label("품질값 준비 중")
                ui.Label("Local 입사각 20° / 이동 RectLight")

    def apply_phase(phase):
        for cell, inputs in zip(cells, patch_inputs):
            state = visual_state(cell, phase, args.clearcoat_safety_limit_um)
            inputs[0].Set(state["base_roughness"])
            inputs[1].Set(state["clearcoat_weight"])
            inputs[2].Set(state["clearcoat_roughness"])
        if phase_label is not None:
            phase_label.text = "현재 상태: 폴리싱 " + ("전" if phase == "before" else "후")
        print(f"[RL visual] phase={phase} material maps applied", flush=True)
        for _ in range(8):
            simulation_app.update()

    def pose_for(cell):
        return measurement_pose(
            cell["point"], cell["normal"], cfg.INCIDENT_ANGLE_DEG,
            cfg.RIG_DISTANCE_M, tangent_hint=cell["tangent_hint"],
        )

    def set_pose(light_position, target_point, up_hint, normal):
        light_translate.Set(Gf.Vec3d(*[float(value) for value in light_position]))
        light_orient.Set(gf_quat(
            Gf, look_at_quaternion(light_position, target_point, up_hint)
        ))
        source_translate.Set(Gf.Vec3d(*[float(value) for value in light_position]))
        target_translate.Set(Gf.Vec3d(*[float(value) for value in target_point + 0.008 * normal]))

    first = cells[0]
    first_pose = pose_for(first)
    set_pose(first_pose["light_position"], first["point"], first_pose["bitangent"], first["normal"])
    active_phases = list(PHASES) if args.phase == "cycle" else [args.phase]
    apply_phase(active_phases[0])

    metadata = {
        "mode": "rl_6region_150_clearcoat_state_visualization",
        "asset": str(args.asset.resolve()),
        "source_cells_csv": str(args.cells_csv.resolve()),
        "sample_count": len(cells),
        "region_order": list(REGION_ORDER),
        "phase_mode": args.phase,
        "renderer": "RTX Real-Time 2.0",
        "moving_light_angle_deg": cfg.INCIDENT_ANGLE_DEG,
        "cell_patch_size_m": args.cell_patch_size_m,
        "clearcoat_visual_mapping": {
            "clearcoatRoughness": "roughness_before_or_after",
            "base_roughness": "0.08 + 0.55 * scratch_before_or_after",
            "clearcoat_weight": "clip(clearcoat_um / safety_limit_um, 0, 1)",
        },
        "display_only": True,
        "actual_path_tracing_measurement_performed": False,
        "disabled_asset_lights": disabled_lights,
    }
    (args.output_dir / "rl_inspection_playback_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("")
    print("=" * 80)
    print("BMW Z4 RL 6영역 150셀 검사광 시각화")
    print(f"입력: {args.cells_csv}")
    print(f"상태 모드: {args.phase}")
    print("표시: Roughness + Scratch blur + Clearcoat integrity")
    print("주의: GUI 표시이며 실제 Path Tracing 수치 측정은 수행하지 않음")
    print("=" * 80)

    current_light = np.asarray(first_pose["light_position"], dtype=float)
    current_target = np.asarray(first["point"], dtype=float)
    current_normal = np.asarray(first["normal"], dtype=float)
    pass_index = 0
    while simulation_app.is_running():
        phase = active_phases[pass_index % len(active_phases)]
        apply_phase(phase)
        pass_index += 1
        for index, cell in enumerate(cells, start=1):
            if not simulation_app.is_running():
                break
            pose = pose_for(cell)
            next_light = np.asarray(pose["light_position"], dtype=float)
            next_target = np.asarray(cell["point"], dtype=float)
            next_normal = np.asarray(cell["normal"], dtype=float)
            started = time.perf_counter()
            while simulation_app.is_running():
                elapsed = time.perf_counter() - started
                alpha = 1.0 if args.transition_seconds == 0.0 else min(
                    elapsed / args.transition_seconds, 1.0
                )
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                normal = (1.0 - smooth) * current_normal + smooth * next_normal
                normal /= np.linalg.norm(normal)
                set_pose(
                    (1.0 - smooth) * current_light + smooth * next_light,
                    (1.0 - smooth) * current_target + smooth * next_target,
                    pose["bitangent"], normal,
                )
                simulation_app.update()
                if alpha >= 1.0:
                    break
            current_light, current_target, current_normal = next_light, next_target, next_normal
            state = visual_state(cell, phase, args.clearcoat_safety_limit_um)
            if region_label is not None:
                region_label.text = f"현재 영역: {cell['region_label']} ({cell['region_id']})"
                cell_label.text = (
                    f"현재 셀: ({cell['grid_row']}, {cell['grid_column']})  "
                    f"전체 {index}/150  반복 {pass_index}"
                )
                quality_label.text = (
                    f"Rough={state['roughness']:.3f}, Scratch={state['scratch']:.3f}, "
                    f"Clearcoat={state['clearcoat_um']:.2f} um, GU proxy={state['gu_proxy']:.2f}"
                )
            values = cell["values"]
            print(
                f"[RL visual] phase={phase}, region={cell['region_id']}, "
                f"cell=({cell['grid_row']},{cell['grid_column']}), "
                f"F={values['force_n']:.3f}N, RPM={values['rpm']:.1f}, "
                f"Feed={values['feed_mm_s']:.3f}mm/s, StepOver={values['step_over_ratio']:.3f}, "
                f"Pass={int(values['pass_count'])}, Rough={state['roughness']:.4f}, "
                f"Scratch={state['scratch']:.4f}, Ra={state['ra_um']:.4f}um, "
                f"Rz={state['rz_um']:.4f}um, Clearcoat={state['clearcoat_um']:.4f}um, "
                f"GU_proxy={state['gu_proxy']:.4f}",
                flush=True,
            )
            wait_with_updates(args.dwell_seconds)
            if index < len(cells) and cells[index]["region_id"] != cell["region_id"]:
                wait_with_updates(args.region_pause_seconds)
        if not args.loop:
            break
    if not args.headless and simulation_app.is_running():
        print("1회 재생 완료. 창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            simulation_app.update()


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
