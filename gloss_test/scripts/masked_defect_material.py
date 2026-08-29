"""Soft roughness-mask defect surface for the flat clearcoat panel."""

from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdShade


def create_soft_square_mask(path, center_uv, size_uv, resolution=1024, feather_uv=0.025):
    """Save a soft-edged square mask: black normal area, white defect area."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    uu, vv = np.meshgrid(axis, axis)
    # Image rows run top-down, while the panel's V coordinate runs bottom-up.
    vv = 1.0 - vv
    half_size = size_uv / 2.0
    signed_distance = np.maximum(
        np.abs(uu - center_uv[0]) - half_size,
        np.abs(vv - center_uv[1]) - half_size,
    )
    transition = np.clip(0.5 - signed_distance / max(feather_uv, 1.0e-6), 0.0, 1.0)
    smooth = transition * transition * (3.0 - 2.0 * transition)
    Image.fromarray(np.round(smooth * 255.0).astype(np.uint8), "L").save(path)
    return path


def create_localized_scratch_normal_map(
    path,
    center_uv,
    size_uv,
    resolution=1024,
    feather_uv=0.025,
    seed=20260826,
    strength=2.8,
):
    """Create deterministic fine scratches and swirl arcs inside the defect mask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    uu, vv_image = np.meshgrid(axis, axis)
    vv = 1.0 - vv_image
    local_u = (uu - center_uv[0]) / size_uv
    local_v = (vv - center_uv[1]) / size_uv
    half_size = size_uv / 2.0
    signed_distance = np.maximum(
        np.abs(uu - center_uv[0]) - half_size,
        np.abs(vv - center_uv[1]) - half_size,
    )
    transition = np.clip(0.5 - signed_distance / max(feather_uv, 1.0e-6), 0.0, 1.0)
    window = transition * transition * (3.0 - 2.0 * transition)

    rng = np.random.default_rng(seed)
    height = np.zeros_like(uu, dtype=np.float32)
    # Hairline scratches with slightly different directions and curvature.
    for _ in range(14):
        angle = rng.uniform(-0.65, 0.65)
        offset = rng.uniform(-0.42, 0.42)
        curve = rng.uniform(-0.10, 0.10)
        transverse = -np.sin(angle) * local_u + np.cos(angle) * local_v
        longitudinal = np.cos(angle) * local_u + np.sin(angle) * local_v
        distance = transverse - offset - curve * longitudinal * longitudinal
        width = rng.uniform(0.006, 0.013)
        height -= rng.uniform(0.35, 0.75) * np.exp(-0.5 * (distance / width) ** 2)

    # Partial swirl marks, typical of rotary polishing or washing damage.
    radius = np.sqrt((local_u + 0.10) ** 2 + (local_v - 0.04) ** 2)
    theta = np.arctan2(local_v - 0.04, local_u + 0.10)
    for target_radius, phase in ((0.20, -0.9), (0.31, -0.2), (0.42, 0.55)):
        arc_gate = np.exp(-0.5 * ((theta - phase) / 1.15) ** 2)
        height -= 0.45 * np.exp(-0.5 * ((radius - target_radius) / 0.010) ** 2) * arc_gate

    height *= window
    gradient_v, gradient_u = np.gradient(height)
    nx = -strength * gradient_u
    ny = strength * gradient_v
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack((nx / length, ny / length, nz / length), axis=-1)
    encoded = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), "RGB").save(path)
    return path


def create_masked_clearcoat_overlay(
    stage,
    path,
    mask_path,
    panel_size_m,
    panel_thickness_m,
    base_color,
    base_roughness,
    clearcoat_weight,
    normal_clearcoat_roughness,
    defect_clearcoat_roughness,
    ior,
    orient_quaternion,
    normal_map_path=None,
):
    """Create a full-panel UV mesh whose clearcoat roughness is texture-masked."""
    half = panel_size_m / 2.0
    surface_z = panel_thickness_m / 2.0 + 2.0e-5
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-half, -half, surface_z),
        Gf.Vec3f(half, -half, surface_z),
        Gf.Vec3f(half, half, surface_z),
        Gf.Vec3f(-half, half, surface_z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([
        Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0),
    ])
    UsdGeom.Xformable(mesh).AddOrientOp().Set(orient_quaternion)

    material = UsdShade.Material.Define(stage, path + "Material")
    surface = UsdShade.Shader.Define(stage, path + "Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(base_roughness))
    surface.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat_weight))
    surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))

    reader = UsdShade.Shader.Define(stage, path + "Material/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, path + "Material/RoughnessMask")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(mask_path)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
    delta = float(defect_clearcoat_roughness - normal_clearcoat_roughness)
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(delta, delta, delta, 1.0))
    texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(normal_clearcoat_roughness, normal_clearcoat_roughness,
                 normal_clearcoat_roughness, 0.0)
    )
    texture_r = texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    surface.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).ConnectToSource(texture_r)

    if normal_map_path is not None:
        normal_texture = UsdShade.Shader.Define(stage, path + "Material/ScratchNormalMap")
        normal_texture.CreateIdAttr("UsdUVTexture")
        normal_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(normal_map_path))
        )
        normal_texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        normal_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        normal_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        normal_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
        normal_texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
        )
        normal_texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
        )
        normal_rgb = normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal_rgb)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh, texture


def bind_masked_clearcoat_material(
    stage,
    mesh,
    material_path,
    mask_path,
    base_color,
    base_roughness,
    clearcoat_weight,
    normal_clearcoat_roughness,
    defect_clearcoat_roughness,
    ior,
    normal_map_path=None,
):
    """Bind the same roughness/scratch network to an existing UV-mapped mesh."""
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, material_path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(base_roughness))
    surface.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat_weight))
    surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))

    reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, material_path + "/RoughnessMask")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(mask_path)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
    delta = float(defect_clearcoat_roughness - normal_clearcoat_roughness)
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(delta, delta, delta, 1.0)
    )
    texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(
        normal_clearcoat_roughness, normal_clearcoat_roughness,
        normal_clearcoat_roughness, 0.0,
    ))
    surface.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )

    if normal_map_path is not None:
        normal_texture = UsdShade.Shader.Define(stage, material_path + "/ScratchNormalMap")
        normal_texture.CreateIdAttr("UsdUVTexture")
        normal_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(normal_map_path))
        )
        normal_texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        normal_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        normal_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        normal_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
        normal_texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
        )
        normal_texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
        )
        surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return material, texture


def bind_vehicle_state_clearcoat_material(
    stage,
    mesh,
    material_path,
    roughness_map_path,
    clearcoat_integrity_map_path,
    normal_map_path,
    base_color,
    base_roughness,
    ior,
):
    """Bind direct RL/state maps to the sampled vehicle clearcoat patch."""
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, material_path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*base_color)
    )
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(base_roughness))
    surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))

    reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(name, file_path):
        node = UsdShade.Shader.Define(stage, material_path + "/" + name)
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(file_path))
        )
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
        return node

    roughness_texture = texture("RoughnessStateMap", roughness_map_path)
    surface.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness_texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )

    integrity_texture = texture("ClearcoatIntegrityMap", clearcoat_integrity_map_path)
    surface.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).ConnectToSource(
        integrity_texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )

    normal_texture = texture("ScratchNormalMap", normal_map_path)
    normal_texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
    )
    normal_texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
    )
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )

    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return material
