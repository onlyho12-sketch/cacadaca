#!/usr/bin/env python3
"""Run a local-20-degree RTX scan on an actual USD vehicle mesh patch."""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from config import gloss_config as cfg
from config.automotive_clearcoat_profiles import get_clearcoat_profile
from config.vehicle_region_profiles import (
    VEHICLE_REGION_PROFILES,
    get_vehicle_region_profile,
)
from scripts.measurement_cell_selection import parse_measurement_cells


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument(
        "--target-prim-suffix",
        default="bmw_z4_car_007_color_polySurface37",
        help="Unique suffix of the UsdGeom.Mesh prim to inspect",
    )
    parser.add_argument("--center-x-m", type=float, default=0.0)
    parser.add_argument("--center-y-m", type=float, default=-0.75)
    parser.add_argument("--scan-span-m", type=float, default=0.16)
    parser.add_argument(
        "--region-profile", choices=tuple(VEHICLE_REGION_PROFILES), default=None,
        help="Bundled BMW region; overrides center, span, grid axes, and ray direction",
    )
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--roughness", type=float, default=0.10)
    parser.add_argument(
        "--distributed-roughness", choices=("initial", "improved"), default=None
    )
    parser.add_argument("--roughness-seed", type=int, default=20260827)
    parser.add_argument(
        "--vehicle-state-npz", type=Path, default=None,
        help="prepared 5x5 vehicle_rl_state_maps.npz to bind to the BMW hood",
    )
    parser.add_argument(
        "--vehicle-state-phase", choices=("before", "after"), default="after",
    )
    parser.add_argument("--clearcoat-safety-limit-um", type=float, default=35.0)
    parser.add_argument("--patch-resolution", type=int, default=51)
    parser.add_argument(
        "--measurement-cells", default=None,
        help=(
            "optional semicolon-separated 1-based cells such as "
            "'1,1;3,3;5,5'; center is automatically included as the "
            "relative-HDR reference"
        ),
    )
    parser.add_argument("--tag", default="vehicle_mesh_hood_rtx_5x5")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()
    if not args.asset.is_file():
        parser.error(f"--asset not found: {args.asset}")
    if args.grid < 3 or args.grid % 2 == 0:
        parser.error("--grid must be an odd integer >= 3")
    if args.scan_span_m <= 0.0:
        parser.error("--scan-span-m must be positive")
    if not 0.0 <= args.roughness <= 1.0:
        parser.error("--roughness must be in [0, 1]")
    if args.patch_resolution < 11 or args.patch_resolution % 2 == 0:
        parser.error("--patch-resolution must be an odd integer >= 11")
    if args.keep_open and args.headless:
        parser.error("--keep-open requires --no-headless")
    if args.vehicle_state_npz is not None and not args.vehicle_state_npz.is_file():
        parser.error(f"--vehicle-state-npz not found: {args.vehicle_state_npz}")
    if args.vehicle_state_npz is not None and args.distributed_roughness is not None:
        parser.error("--vehicle-state-npz cannot be combined with --distributed-roughness")
    if args.clearcoat_safety_limit_um <= 0.0:
        parser.error("--clearcoat-safety-limit-um must be positive")
    try:
        args.measurement_cells_parsed = parse_measurement_cells(
            args.measurement_cells, args.grid
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
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


def find_unique_mesh(stage, UsdGeom, suffix):
    matches = [
        prim for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mesh ending with {suffix!r}, found "
            f"{len(matches)}: {[str(prim.GetPath()) for prim in matches]}"
        )
    return matches[0]


def world_mesh_arrays(prim, UsdGeom, Gf, triangulate_faces):
    mesh = UsdGeom.Mesh(prim)
    local_points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    points = np.asarray([
        tuple(transform.Transform(Gf.Vec3d(*[float(value) for value in point])))
        for point in local_points
    ], dtype=float)
    triangles = triangulate_faces(counts, indices)
    return mesh, points, triangles


def save_heatmap(path, rows, grid, region_label="hood"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.full((grid, grid), np.nan, dtype=float)
    for row in rows:
        values[int(row["grid_row"]) - 1, int(row["grid_column"]) - 1] = float(
            row["relative_gloss_to_center_not_gu"]
        )
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = axis.imshow(values, origin="lower", cmap="viridis")
    for row in range(grid):
        for column in range(grid):
            if np.isfinite(values[row, column]):
                axis.text(
                    column, row, f"{values[row, column]:.3f}",
                    ha="center", va="center", fontsize=8,
                )
    axis.set_title(
        f"BMW Z4 {region_label} mesh\n"
        "RTX relative gloss (center = 1.0, not GU)"
    )
    axis.set_xlabel("grid column")
    axis.set_ylabel("grid row")
    axis.set_xticks(range(grid), range(1, grid + 1))
    axis.set_yticks(range(grid), range(1, grid + 1))
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_sampled_clearcoat_patch(
    stage, dense_rows, resolution, UsdGeom, Gf, Sdf, path, offset_m=2.0e-4
):
    """Create a UV patch following dense ray hits on the source vehicle mesh."""
    if len(dense_rows) != resolution * resolution:
        raise ValueError("Dense vehicle patch grid is incomplete")
    points = []
    vertex_normals = []
    for row in dense_rows:
        hit = row["hit"]
        if hit is None:
            raise ValueError(
                f"Vehicle patch misses source mesh at "
                f"({row['grid_row']}, {row['grid_column']})"
            )
        point = hit["point"] + offset_m * hit["normal"]
        points.append(Gf.Vec3f(*[float(value) for value in point]))
        vertex_normals.append(
            Gf.Vec3f(*[float(value) for value in hit["normal"]])
        )
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
    # RTX's normal-mapped PreviewSurface path consumes face-varying normals and
    # UVs.  Expand the per-vertex data in the exact faceVertexIndices order.
    face_varying_normals = [vertex_normals[index] for index in indices]
    vertex_st = [
        Gf.Vec2f(float(column / (resolution - 1)), float(row / (resolution - 1)))
        for row in range(resolution)
        for column in range(resolution)
    ]
    face_varying_st = [vertex_st[index] for index in indices]
    if len(face_varying_normals) != len(indices) or len(face_varying_st) != len(indices):
        raise RuntimeError("Face-varying normal/UV array size mismatch")
    mesh.CreateNormalsAttr(face_varying_normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(face_varying_st)
    if len(mesh.GetNormalsAttr().Get()) != len(indices) or len(st.Get()) != len(indices):
        raise RuntimeError("Authored face-varying primvar validation failed")
    return mesh


def add_vehicle_overview(
    stage, samples, Gf, Sdf, UsdGeom, UsdLux, UsdShade, look_at_quaternion
):
    root = "/World/GUI_ONLY_PostMeasurement"
    UsdGeom.Xform.Define(stage, root)

    ground_material = UsdShade.Material.Define(stage, root + "/GroundMaterial")
    ground_shader = UsdShade.Shader.Define(stage, root + "/GroundMaterial/Shader")
    ground_shader.CreateIdAttr("UsdPreviewSurface")
    ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.18, 0.21, 0.25)
    )
    ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    ground_material.CreateSurfaceOutput().ConnectToSource(
        ground_shader.ConnectableAPI(), "surface"
    )
    ground = UsdGeom.Cube.Define(stage, root + "/Ground")
    ground.CreateSizeAttr(1.0)
    ground_xform = UsdGeom.Xformable(ground)
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.035))
    ground_xform.AddScaleOp().Set(Gf.Vec3d(5.0, 5.0, 0.05))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

    # Keep the environment deliberately dim.  The previous 500-intensity dome
    # made the white vehicle and background converge to the same display value.
    dome = UsdLux.DomeLight.Define(stage, root + "/Environment")
    dome.CreateIntensityAttr(18.0)
    dome.CreateColorAttr(Gf.Vec3f(0.12, 0.16, 0.22))

    vehicle_target = np.asarray([0.0, -0.10, 0.42], dtype=float)
    for name, position, intensity, color in (
        ("KeyLight", [2.2, -2.6, 3.0], 1800.0, [1.0, 0.92, 0.82]),
        ("FillLight", [-2.0, -0.3, 1.8], 900.0, [0.68, 0.80, 1.0]),
    ):
        overview_light = UsdLux.RectLight.Define(stage, root + "/" + name)
        overview_light.CreateWidthAttr(2.2)
        overview_light.CreateHeightAttr(2.2)
        overview_light.CreateIntensityAttr(intensity)
        overview_light.CreateColorAttr(Gf.Vec3f(*color))
        overview_light_xform = UsdGeom.Xformable(overview_light)
        overview_light_xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        overview_light_xform.AddOrientOp().Set(gf_quat(
            Gf,
            look_at_quaternion(position, vehicle_target, [0.0, 0.0, 1.0]),
        ))

    marker_material = UsdShade.Material.Define(stage, root + "/MarkerMaterial")
    marker_shader = UsdShade.Shader.Define(stage, root + "/MarkerMaterial/Shader")
    marker_shader.CreateIdAttr("UsdPreviewSurface")
    marker_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.18, 0.01)
    )
    marker_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(3.0, 0.25, 0.01)
    )
    marker_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    marker_material.CreateSurfaceOutput().ConnectToSource(
        marker_shader.ConnectableAPI(), "surface"
    )
    for index, sample in enumerate(samples, start=1):
        position = sample["point"] + 0.004 * sample["normal"]
        marker = UsdGeom.Sphere.Define(stage, root + f"/ScanPoint_{index:02d}")
        marker.CreateRadiusAttr(0.004)
        UsdGeom.Xformable(marker).AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(marker_material)

    camera_position = np.asarray([2.65, -3.65, 2.05], dtype=float)
    target = vehicle_target
    camera = UsdGeom.Camera.Define(stage, root + "/VehicleOverviewCamera")
    camera.CreateFocalLengthAttr(52.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_position))
    camera_xform.AddOrientOp().Set(gf_quat(
        Gf, look_at_quaternion(camera_position, target, [0.0, 0.0, 1.0])
    ))
    return camera.GetPath()


def main():
    import carb.settings
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    from scripts.clearcoat_material import create_clearcoat_material, set_clearcoat_roughness
    from scripts.distributed_roughness import (
        create_continuous_severity_map,
        create_distributed_scratch_normal_map,
        create_severity_grid,
        save_severity_grid,
    )
    from scripts.gloss_geometry import look_at_quaternion, measurement_pose
    from scripts.mesh_surface_sampling import sample_planar_grid, triangulate_faces
    from scripts.masked_defect_material import (
        bind_masked_clearcoat_material,
        bind_vehicle_state_clearcoat_material,
    )
    from scripts.reflection_measurement import measure_roi, rgb_array, save_capture
    from scripts.validate_curved_local_20 import angle_deg
    from scripts.vehicle_state_textures import (
        load_vehicle_state,
        save_clearcoat_thickness_texture,
        save_scalar_texture,
        save_scratch_normal_texture,
    )

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
    asset_root = stage.DefinePrim("/World/Vehicle", "Xform")
    asset_root.GetReferences().AddReference(str(args.asset.resolve()))
    stage.Load()
    for _ in range(4):
        simulation_app.update()

    # Imported presentation assets can carry their own studio/environment
    # lights.  They would contaminate the local 20-degree measurement and, in
    # this BMW asset, saturate the white body under a 1000-intensity DomeLight.
    # Author local overrides before creating the dedicated measurement light.
    disabled_asset_lights = []
    for prim in Usd.PrimRange(asset_root):
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        light_api = UsdLux.LightAPI(prim)
        previous_intensity = float(light_api.GetIntensityAttr().Get() or 0.0)
        light_api.GetIntensityAttr().Set(0.0)
        disabled_asset_lights.append({
            "prim_path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "previous_intensity": previous_intensity,
        })

    target_prim = find_unique_mesh(stage, UsdGeom, args.target_prim_suffix)
    mesh, points, triangles = world_mesh_arrays(
        target_prim, UsdGeom, Gf, triangulate_faces
    )
    if args.region_profile is not None:
        inspection = get_vehicle_region_profile(args.region_profile)
        inspection_center = np.asarray(inspection["center_m"], dtype=float)
        inspection_axis_u = np.asarray(inspection["axis_u"], dtype=float)
        inspection_axis_v = np.asarray(inspection["axis_v"], dtype=float)
        inspection_ray_direction = np.asarray(
            inspection["ray_direction"], dtype=float
        )
        inspection_span_m = float(inspection["span_m"])
        region_label = inspection["label"]
        region_id = args.region_profile
    else:
        inspection_center = np.asarray(
            [args.center_x_m, args.center_y_m, points[:, 2].max()], dtype=float
        )
        inspection_axis_u = np.asarray([1.0, 0.0, 0.0])
        inspection_axis_v = np.asarray([0.0, 1.0, 0.0])
        inspection_ray_direction = np.asarray([0.0, 0.0, -1.0])
        inspection_span_m = args.scan_span_m
        region_label = "보닛 사용자 좌표"
        region_id = "custom_top_xy"
    sampled_all = sample_planar_grid(
        points, triangles, inspection_center, inspection_axis_u,
        inspection_axis_v, inspection_ray_direction, inspection_span_m,
        args.grid,
    )
    missing = [
        (row["grid_row"], row["grid_column"])
        for row in sampled_all if row["hit"] is None
    ]
    if missing:
        raise RuntimeError(
            f"Scan region leaves target mesh at cells {missing}; change center/span"
        )
    samples_all = [dict(row["hit"]) for row in sampled_all]

    material_profile = get_clearcoat_profile(
        "white_automotive_literature_composite_v1"
    )
    material, shader = create_clearcoat_material(
        stage, "/World/VehicleInspectionClearcoat", material_profile
    )
    set_clearcoat_roughness(shader, args.roughness)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    severity = None
    vehicle_state = None
    state_asset_paths = None
    if args.distributed_roughness or args.vehicle_state_npz is not None:
        patch_span_m = inspection_span_m + 0.04
        dense_rows = sample_planar_grid(
            points, triangles, inspection_center, inspection_axis_u,
            inspection_axis_v, inspection_ray_direction, patch_span_m,
            args.patch_resolution,
        )
        patch_mesh = create_sampled_clearcoat_patch(
            stage, dense_rows, args.patch_resolution, UsdGeom, Gf, Sdf,
            "/World/VehicleInspection/ClearcoatDefectPatch",
        )
        asset_dir = output_dir / "assets"
        if args.vehicle_state_npz is not None:
            vehicle_state = load_vehicle_state(
                args.vehicle_state_npz, args.vehicle_state_phase,
                args.clearcoat_safety_limit_um,
            )
            if vehicle_state["grid_size"] != args.grid:
                raise ValueError(
                    f"vehicle state grid is {vehicle_state['grid_size']}x"
                    f"{vehicle_state['grid_size']}, but --grid is {args.grid}"
                )
            roughness_map_path, _ = save_scalar_texture(
                asset_dir / f"roughness_{args.vehicle_state_phase}.png",
                vehicle_state["roughness"],
            )
            integrity_map_path, _ = save_scalar_texture(
                asset_dir / f"clearcoat_integrity_{args.vehicle_state_phase}.png",
                vehicle_state["clearcoat_integrity"],
            )
            thickness_map_path, _ = save_clearcoat_thickness_texture(
                asset_dir / f"clearcoat_thickness_{args.vehicle_state_phase}.png",
                vehicle_state["clearcoat_thickness_um"],
            )
            normal_map_path, _ = save_scratch_normal_texture(
                asset_dir / f"scratch_normal_{args.vehicle_state_phase}.png",
                vehicle_state["scratch"], seed=args.roughness_seed,
                strength=1.2,
            )
            bind_vehicle_state_clearcoat_material(
                stage, patch_mesh,
                "/World/VehicleInspection/VehicleStateClearcoatMaterial",
                roughness_map_path, integrity_map_path, normal_map_path,
                material_profile.BASE_COLOR, material_profile.BASE_ROUGHNESS,
                material_profile.IOR,
            )
            state_asset_paths = {
                "roughness_map": str(roughness_map_path),
                "scratch_normal_map": str(normal_map_path),
                "clearcoat_integrity_map": str(integrity_map_path),
                "clearcoat_thickness_diagnostic_map": str(thickness_map_path),
            }
        else:
            pristine_cell = (3, 3)
            severity = create_severity_grid(5, args.roughness_seed, pristine_cell)
            residual_factor = 1.0 if args.distributed_roughness == "initial" else 0.08
            scratch_strength = 1.2 if args.distributed_roughness == "initial" else 0.08
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
                stage, patch_mesh, "/World/VehicleInspection/DefectClearcoatMaterial",
                mask_path, material_profile.BASE_COLOR,
                material_profile.BASE_ROUGHNESS, material_profile.CLEARCOAT_WEIGHT,
                args.roughness, 0.35, material_profile.IOR, normal_map_path,
            )
        # The measurement locations lie on the middle 80% of the 200 mm patch.
        # Move the optical rig to the physical overlay rather than the source body.
        for sample in samples_all:
            sample["point"] = sample["point"] + 2.0e-4 * sample["normal"]

    if args.measurement_cells_parsed is None:
        sampled = sampled_all
        samples = samples_all
    else:
        requested = set(args.measurement_cells_parsed)
        pairs = [
            (grid_row, sample)
            for grid_row, sample in zip(sampled_all, samples_all)
            if (grid_row["grid_row"], grid_row["grid_column"]) in requested
        ]
        sampled = [pair[0] for pair in pairs]
        samples = [pair[1] for pair in pairs]
        if len(pairs) != len(requested):
            raise RuntimeError("requested representative measurement cells are incomplete")

    center_index = next(
        index for index, row in enumerate(sampled)
        if row["grid_row"] == args.grid // 2 + 1
        and row["grid_column"] == args.grid // 2 + 1
    )
    center_sample = samples[center_index]
    center_rig = measurement_pose(
        center_sample["point"], center_sample["normal"],
        cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M,
        tangent_hint=inspection_axis_u,
    )
    light = UsdLux.RectLight.Define(stage, "/World/GlossMeasurement/Light")
    light.CreateWidthAttr(cfg.LIGHT_WIDTH_M)
    light.CreateHeightAttr(cfg.LIGHT_HEIGHT_M)
    light.CreateIntensityAttr(cfg.LIGHT_INTENSITY)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddTranslateOp().Set(Gf.Vec3d(*center_rig["light_position"]))
    light_xform.AddOrientOp().Set(gf_quat(Gf, look_at_quaternion(
        center_rig["light_position"], center_rig["point"], center_rig["bitangent"]
    )))

    camera = UsdGeom.Camera.Define(stage, "/World/GlossMeasurement/Camera")
    camera.CreateFocalLengthAttr(50.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*center_rig["camera_position"]))
    camera_xform.AddOrientOp().Set(gf_quat(Gf, look_at_quaternion(
        center_rig["camera_position"], center_rig["point"], center_rig["bitangent"]
    )))
    camera_prim = camera.GetPrim()
    camera_prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
    camera_prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
    camera_prim.CreateAttribute(
        "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool
    ).Set(False)
    for name, value in {
        "exposure": 0.0, "exposure:fStop": 1.0, "exposure:iso": 0.0,
        "exposure:responsivity": 1.0, "exposure:time": 1.0,
    }.items():
        camera_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", cfg.PATH_TRACING_SPP)
    settings.set("/rtx/pathtracing/totalSpp", cfg.PATH_TRACING_SPP)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)
    render_product = rep.create.render_product(camera.GetPath(), cfg.RESOLUTION)
    hdr_annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
    hdr_annotator.attach(render_product)
    rep.orchestrator.set_capture_on_play(False)
    light_ops = light_xform.GetOrderedXformOps()
    camera_ops = camera_xform.GetOrderedXformOps()

    print("")
    print("=" * 80)
    print("실제 BMW Z4 Mesh 다중 영역 RTX Path Tracing Local 20° 검사")
    print(f"자산      : {args.asset.resolve()}")
    print(f"대상 Prim : {target_prim.GetPath()}")
    print(
        f"내장 조명 : {len(disabled_asset_lights)}개 비활성화 "
        f"(전용 20° 측정광만 사용)"
    )
    print(
        f"Mesh      : vertices={len(points)}, triangles={len(triangles)}, "
        f"roughness={args.roughness:.3f}"
    )
    print(
        f"검사 영역 : {region_label} ({region_id}), "
        f"center=({inspection_center[0]:.3f}, {inspection_center[1]:.3f}, "
        f"{inspection_center[2]:.3f}) m, "
        f"span={inspection_span_m * 1000:.0f} mm, {args.grid}x{args.grid}"
    )
    if args.measurement_cells_parsed is not None:
        print(
            "대표 측정 : "
            + ";".join(f"{row},{column}" for row, column in args.measurement_cells_parsed)
            + f" ({len(samples)}/{args.grid * args.grid}셀 Path Tracing)"
        )
    if args.vehicle_state_npz is not None:
        thickness = vehicle_state["clearcoat_thickness_um"]
        print(
            f"표면 상태 : vehicle state {args.vehicle_state_phase}, "
            f"grid={vehicle_state['grid_size']}x{vehicle_state['grid_size']}"
        )
        print(
            f"Clearcoat : {thickness.min():.3f}~{thickness.max():.3f} um, "
            f"안전선 {args.clearcoat_safety_limit_um:.3f} um 미만="
            f"{int(vehicle_state['clearcoat_safety_failure'].sum())}셀"
        )
        print("두께 표현 : 안전선 이상은 광학층 유지, 미만만 integrity 감소 (PT-DESIGN)")
        print(f"상태 출처 : {args.vehicle_state_npz.resolve()}")
    elif args.distributed_roughness:
        print(
            f"표면 상태 : distributed {args.distributed_roughness}, "
            f"seed={args.roughness_seed}, (3,3)=pristine"
        )
    print("측정값    : HDR 정반사 ROI 및 중앙 대비 상대광택 (GU 아님)")
    print("=" * 80)

    first = samples[0]
    warmup_rig = measurement_pose(
        first["point"], first["normal"], cfg.INCIDENT_ANGLE_DEG,
        cfg.RIG_DISTANCE_M, tangent_hint=inspection_axis_u,
    )
    light_ops[0].Set(Gf.Vec3d(*warmup_rig["light_position"]))
    light_ops[1].Set(gf_quat(Gf, look_at_quaternion(
        warmup_rig["light_position"], first["point"], warmup_rig["bitangent"]
    )))
    camera_ops[0].Set(Gf.Vec3d(*warmup_rig["camera_position"]))
    camera_ops[1].Set(gf_quat(Gf, look_at_quaternion(
        warmup_rig["camera_position"], first["point"], warmup_rig["bitangent"]
    )))
    for _ in range(max(16, cfg.SETTLE_FRAMES * 5)):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    warmup_hdr = rgb_array(hdr_annotator.get_data())
    print(
        f"[warm-up] discarded HDR_ROI="
        f"{measure_roi(warmup_hdr, cfg.ROI_FRACTION)['roi_mean_intensity']:.8f}"
    )

    rows = []
    total = len(samples)
    for index, (grid_row, sample) in enumerate(zip(sampled, samples), start=1):
        point = sample["point"]
        normal = sample["normal"]
        rig = measurement_pose(
            point, normal, cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M,
            tangent_hint=inspection_axis_u,
        )
        light_ops[0].Set(Gf.Vec3d(*rig["light_position"]))
        light_ops[1].Set(gf_quat(Gf, look_at_quaternion(
            rig["light_position"], point, rig["bitangent"]
        )))
        camera_ops[0].Set(Gf.Vec3d(*rig["camera_position"]))
        camera_ops[1].Set(gf_quat(Gf, look_at_quaternion(
            rig["camera_position"], point, rig["bitangent"]
        )))
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
            print(f"  [{index:02d}/{total}] HDR=0 retry {attempt}/3")
        preview = np.clip(
            hdr_image.astype(np.float32) * cfg.PNG_PREVIEW_EXPOSURE_SCALE,
            0.0, 1.0,
        )
        stem = f"row_{grid_row['grid_row']:02d}_col_{grid_row['grid_column']:02d}"
        save_capture(
            preview, image_dir / f"{stem}.png", raw_dir / f"{stem}_preview.npy",
            metrics["roi_bounds"],
        )
        np.save(raw_dir / f"{stem}_hdr.npy", hdr_image)
        incident = angle_deg(rig["light_direction_from_surface"], normal)
        detection = angle_deg(rig["detector_direction_from_surface"], normal)
        row = {
            "grid_row": grid_row["grid_row"],
            "grid_column": grid_row["grid_column"],
            "position_x_m": float(point[0]),
            "position_y_m": float(point[1]),
            "position_z_m": float(point[2]),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "triangle_index": sample["triangle_index"],
            "surface_state": (
                f"vehicle_state_{args.vehicle_state_phase}"
                if vehicle_state is not None else args.distributed_roughness or "uniform"
            ),
            "initial_severity": (
                float(severity[grid_row["grid_row"] - 1, grid_row["grid_column"] - 1])
                if severity is not None else 0.0
            ),
            "state_roughness": (
                float(vehicle_state["roughness"][
                    grid_row["grid_row"] - 1, grid_row["grid_column"] - 1
                ]) if vehicle_state is not None else None
            ),
            "state_scratch": (
                float(vehicle_state["scratch"][
                    grid_row["grid_row"] - 1, grid_row["grid_column"] - 1
                ]) if vehicle_state is not None else None
            ),
            "state_clearcoat_um": (
                float(vehicle_state["clearcoat_thickness_um"][
                    grid_row["grid_row"] - 1, grid_row["grid_column"] - 1
                ]) if vehicle_state is not None else None
            ),
            "incident_angle_deg": incident,
            "detection_angle_deg": detection,
            "azimuth_retry_used": False,
            "measurement_tangent_hint": "+axis_u",
            "hdr_roi_mean_intensity": metrics["roi_mean_intensity"],
            "hdr_roi_spatial_std": metrics["roi_std_intensity"],
            "hdr_roi_peak_intensity": metrics["roi_peak_intensity"],
        }
        rows.append(row)
        print(
            f"  [{index:02d}/{total}] cell=({row['grid_row']},{row['grid_column']}) "
            f"p=({point[0]:+.3f},{point[1]:+.3f},{point[2]:+.3f}) m, "
            f"n=({normal[0]:+.3f},{normal[1]:+.3f},{normal[2]:+.3f}), "
            f"입사/검출={incident:.3f}°/{detection:.3f}°, "
            f"HDR_ROI={metrics['roi_mean_intensity']:.8f}"
        )

    # A real vehicle can occlude one azimuth even though the local point and
    # normal are valid.  Re-measure zero cells in other tangent directions while
    # preserving the same local 20-degree incidence/detection geometry.
    tangent_candidates = (
        ("+axis_v", inspection_axis_v),
        ("-axis_u", -inspection_axis_u),
        ("-axis_v", -inspection_axis_v),
    )
    zero_indices = [
        index for index, row in enumerate(rows)
        if row["hdr_roi_mean_intensity"] <= 0.0
    ]
    for row_index in zero_indices:
        row = rows[row_index]
        sample = samples[row_index]
        point = sample["point"]
        normal = sample["normal"]
        print(
            f"[azimuth retry] cell=({row['grid_row']},{row['grid_column']}) "
            "initial +axis_u HDR=0"
        )
        for tangent_name, tangent_hint in tangent_candidates:
            rig = measurement_pose(
                point, normal, cfg.INCIDENT_ANGLE_DEG, cfg.RIG_DISTANCE_M,
                tangent_hint=tangent_hint,
            )
            light_ops[0].Set(Gf.Vec3d(*rig["light_position"]))
            light_ops[1].Set(gf_quat(Gf, look_at_quaternion(
                rig["light_position"], point, rig["bitangent"]
            )))
            camera_ops[0].Set(Gf.Vec3d(*rig["camera_position"]))
            camera_ops[1].Set(gf_quat(Gf, look_at_quaternion(
                rig["camera_position"], point, rig["bitangent"]
            )))
            for _ in range(max(10, cfg.SETTLE_FRAMES * 3)):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
            hdr_image = rgb_array(hdr_annotator.get_data())
            metrics = measure_roi(hdr_image, cfg.ROI_FRACTION)
            print(
                f"  tangent={tangent_name}, "
                f"HDR_ROI={metrics['roi_mean_intensity']:.8f}"
            )
            if metrics["roi_mean_intensity"] <= 0.0:
                continue
            preview = np.clip(
                hdr_image.astype(np.float32) * cfg.PNG_PREVIEW_EXPOSURE_SCALE,
                0.0, 1.0,
            )
            stem = f"row_{row['grid_row']:02d}_col_{row['grid_column']:02d}"
            save_capture(
                preview, image_dir / f"{stem}.png",
                raw_dir / f"{stem}_preview.npy", metrics["roi_bounds"],
            )
            np.save(raw_dir / f"{stem}_hdr.npy", hdr_image)
            row.update({
                "incident_angle_deg": angle_deg(
                    rig["light_direction_from_surface"], normal
                ),
                "detection_angle_deg": angle_deg(
                    rig["detector_direction_from_surface"], normal
                ),
                "azimuth_retry_used": True,
                "measurement_tangent_hint": tangent_name,
                "hdr_roi_mean_intensity": metrics["roi_mean_intensity"],
                "hdr_roi_spatial_std": metrics["roi_std_intensity"],
                "hdr_roi_peak_intensity": metrics["roi_peak_intensity"],
            })
            break

    center = next(
        row for row in rows
        if row["grid_row"] == args.grid // 2 + 1
        and row["grid_column"] == args.grid // 2 + 1
    )
    reference = center["hdr_roi_mean_intensity"]
    if reference <= 0.0:
        raise RuntimeError("Center HDR reference is zero")
    for row in rows:
        row["relative_gloss_to_center_not_gu"] = (
            row["hdr_roi_mean_intensity"] / reference
        )

    csv_path = output_dir / "vehicle_mesh_rtx_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([row["hdr_roi_mean_intensity"] for row in rows])
    relative = np.asarray([row["relative_gloss_to_center_not_gu"] for row in rows])
    non_pristine_relative = np.asarray([
        row["relative_gloss_to_center_not_gu"]
        for row in rows if row["initial_severity"] > 0.0
    ])
    optical_state_consistent = True
    if args.distributed_roughness == "initial":
        optical_state_consistent = bool(
            non_pristine_relative.size == args.grid * args.grid - 1
            and float(non_pristine_relative.mean()) < 0.90
            and int(np.sum(non_pristine_relative < 0.95)) >= 20
        )
    elif args.distributed_roughness == "improved":
        optical_state_consistent = bool(
            non_pristine_relative.size == args.grid * args.grid - 1
            and float(non_pristine_relative.mean()) > 0.85
        )
    summary = {
        "mode": "arbitrary_usd_vehicle_mesh_rtx_local_20",
        "asset": str(args.asset.resolve()),
        "target_prim": str(target_prim.GetPath()),
        "region_id": region_id,
        "region_label": region_label,
        "material_profile": material_profile.metadata(),
        "surface_state": (
            f"vehicle_state_{args.vehicle_state_phase}"
            if vehicle_state is not None else args.distributed_roughness or "uniform"
        ),
        "vehicle_state_npz": (
            str(args.vehicle_state_npz.resolve())
            if args.vehicle_state_npz is not None else None
        ),
        "vehicle_state_phase": args.vehicle_state_phase if vehicle_state is not None else None,
        "vehicle_state_texture_assets": state_asset_paths,
        "clearcoat_thickness_range_um": (
            [
                float(vehicle_state["clearcoat_thickness_um"].min()),
                float(vehicle_state["clearcoat_thickness_um"].max()),
            ] if vehicle_state is not None else None
        ),
        "clearcoat_safety_limit_um": (
            args.clearcoat_safety_limit_um if vehicle_state is not None else None
        ),
        "clearcoat_safety_failure_cell_count": (
            int(vehicle_state["clearcoat_safety_failure"].sum())
            if vehicle_state is not None else None
        ),
        "clearcoat_visualization_rule": (
            "sound_above_safety_limit_then_integrity_reduces_below_limit_PT-DESIGN"
            if vehicle_state is not None else None
        ),
        "roughness_seed": args.roughness_seed if severity is not None else None,
        "disabled_asset_lights": disabled_asset_lights,
        "pristine_cell": [3, 3] if severity is not None else None,
        "clearcoat_patch_source": (
            "dense_51x51_raycast_resampling_of_actual_vehicle_mesh"
            if severity is not None else None
        ),
        "renderer": "RTX Path Tracing",
        "measurement_aov": "HdrColor",
        "sample_count": len(rows),
        "full_grid_cell_count": args.grid * args.grid,
        "valid_mesh_hit_count": len(samples_all),
        "representative_measurement": args.measurement_cells_parsed is not None,
        "measurement_cells": (
            [list(cell) for cell in args.measurement_cells_parsed]
            if args.measurement_cells_parsed is not None else None
        ),
        "is_gu": False,
        "scan_center_xyz_m": inspection_center.tolist(),
        "scan_axis_u": inspection_axis_u.tolist(),
        "scan_axis_v": inspection_axis_v.tolist(),
        "scan_ray_direction": inspection_ray_direction.tolist(),
        "scan_span_m": inspection_span_m,
        "relative_to_center_min": float(relative.min()),
        "relative_to_center_max": float(relative.max()),
        "non_pristine_relative_mean": (
            float(non_pristine_relative.mean())
            if non_pristine_relative.size else None
        ),
        "optical_state_consistent": optical_state_consistent,
        "all_values_finite_and_positive": bool(
            np.all(np.isfinite(values)) and np.all(values > 0.0)
        ),
        "azimuth_retry_count": sum(row["azimuth_retry_used"] for row in rows),
        "max_incident_angle_error_deg": max(
            abs(row["incident_angle_deg"] - cfg.INCIDENT_ANGLE_DEG) for row in rows
        ),
        "max_detection_angle_error_deg": max(
            abs(row["detection_angle_deg"] - cfg.INCIDENT_ANGLE_DEG) for row in rows
        ),
    }
    summary["passed"] = (
        summary["valid_mesh_hit_count"] == args.grid * args.grid
        and summary["all_values_finite_and_positive"]
        and summary["max_incident_angle_error_deg"] < 1e-6
        and summary["max_detection_angle_error_deg"] < 1e-6
        and summary["optical_state_consistent"]
    )
    summary_path = output_dir / "vehicle_mesh_rtx_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    heatmap_path = output_dir / "vehicle_mesh_relative_gloss_heatmap.png"
    save_heatmap(heatmap_path, rows, args.grid, region_label=region_id)
    stage.Export(str(output_dir / "vehicle_mesh_rtx_scene.usda"))

    print("")
    print("-" * 80)
    print(f"실제 Mesh 교차점 : {len(samples_all)}/{args.grid * args.grid}")
    print(f"Path Tracing 측정 : {len(rows)}/{args.grid * args.grid} (대표 셀 모드 가능)")
    print(f"HDR 양수·유한값 : {int(np.sum(np.isfinite(values) & (values > 0.0)))}/{len(rows)}")
    print(f"상대광택 범위   : {relative.min():.6f} ~ {relative.max():.6f} (GU 아님)")
    if non_pristine_relative.size:
        print(f"결함부 평균     : {non_pristine_relative.mean():.6f} (정상 셀=1, GU 아님)")
    print(f"최종 판정       : {'통과' if summary['passed'] else '실패'}")
    print(f"CSV             : {csv_path}")
    print(f"요약 JSON       : {summary_path}")
    print(f"히트맵          : {heatmap_path}")
    print("-" * 80)
    if not summary["passed"]:
        raise RuntimeError(f"Vehicle mesh RTX validation failed; see {summary_path}")

    if args.keep_open:
        # The measurement render product is no longer needed in the interactive
        # overview.  Leaving it alive makes RTX render the hidden 512x512 camera
        # in addition to the main viewport on every frame.
        hdr_annotator.detach([render_product])
        render_product.destroy()
        for _ in range(4):
            simulation_app.update()
        print("GUI 자원 정리: 측정용 512x512 Render Product 해제")

        settings.set("/rtx/rendermode", "RealTimePathTracing")
        light.GetIntensityAttr().Set(0.0)
        overview = add_vehicle_overview(
            stage, samples, Gf, Sdf, UsdGeom, UsdLux, UsdShade,
            look_at_quaternion,
        )
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(str(overview))
        overview_product = rep.create.render_product(overview, (1280, 720))
        overview_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        overview_annotator.attach(overview_product)
        # A newly-created product needs explicit Replicator capture frames after
        # the measurement product has been destroyed; app updates alone do not
        # populate a fresh LdrColor annotator.
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
        overview_image = rgb_array(overview_annotator.get_data())
        if np.issubdtype(overview_image.dtype, np.integer):
            overview_u8 = overview_image.astype(np.uint8)
        else:
            overview_u8 = np.round(
                np.clip(overview_image.astype(np.float32), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        from PIL import Image
        overview_path = output_dir / "gui_vehicle_overview.png"
        Image.fromarray(overview_u8, "RGB").save(overview_path)
        overview_luminance = (
            0.2126 * overview_u8[:, :, 0]
            + 0.7152 * overview_u8[:, :, 1]
            + 0.0722 * overview_u8[:, :, 2]
        )
        print(
            "GUI 캡처 검증: "
            f"min={overview_luminance.min():.1f}, "
            f"mean={overview_luminance.mean():.1f}, "
            f"max={overview_luminance.max():.1f}, "
            f"white={(overview_luminance >= 250).mean():.3%}"
        )
        print(f"GUI 확인 PNG: {overview_path}")
        # The high-resolution verification camera is only needed for the saved
        # PNG.  Destroy it immediately so the live GUI renders one viewport.
        overview_annotator.detach([overview_product])
        overview_product.destroy()
        for _ in range(8):
            simulation_app.update()
        print("GUI 자원 정리: PNG용 1280x720 Render Product 해제")

        import time
        performance_frames = 90
        performance_start = time.perf_counter()
        for _ in range(performance_frames):
            simulation_app.update()
        performance_elapsed = time.perf_counter() - performance_start
        overview_fps = performance_frames / max(performance_elapsed, 1.0e-9)
        print(
            f"GUI 단일 뷰포트 업데이트: {overview_fps:.1f} FPS "
            f"({performance_frames} frames / {performance_elapsed:.2f} s)"
        )
        print("GUI: 실제 BMW Z4 전체 + 보닛 5x5 주황 측정점")
        print("주황 마커·환경광·바닥은 RTX 측정 완료 후에만 추가됩니다.")
        print("창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            simulation_app.update()


status_path = TEST_ROOT / "results" / args.tag / "run_status.json"
failed = False
try:
    main()
except Exception:
    traceback.print_exc()
    failed = True
finally:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({"success": not failed}, indent=2), encoding="utf-8"
    )
    simulation_app.close()

if failed:
    raise SystemExit(1)
