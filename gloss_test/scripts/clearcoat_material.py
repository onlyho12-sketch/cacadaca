"""USD Preview Surface clearcoat material creation and updates."""

from pxr import Gf, Sdf, UsdShade


def create_clearcoat_material(stage, path, config):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*config.BASE_COLOR))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(config.BASE_ROUGHNESS)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(config.CLEARCOAT_WEIGHT)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(config.ROUGHNESS_VALUES[0])
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(config.IOR)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material, shader


def set_clearcoat_roughness(shader, value):
    shader.GetInput("clearcoatRoughness").Set(float(value))

