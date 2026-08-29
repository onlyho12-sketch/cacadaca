#!/usr/bin/env python3
"""List mesh prims and world-space bounds from a USD asset in Isaac Sim."""

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


args = parse_args()
if not args.asset.is_file():
    raise SystemExit(f"Asset not found: {args.asset}")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})


def vec3(value):
    return [float(value[index]) for index in range(3)]


def main():
    import omni.usd
    from pxr import Usd, UsdGeom, UsdLux

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    root = stage.DefinePrim("/World/ImportedAsset", "Xform")
    root.GetReferences().AddReference(str(args.asset.resolve()))
    stage.Load()
    for _ in range(4):
        simulation_app.update()

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    rows = []
    lights = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdLux.LightAPI):
            light_api = UsdLux.LightAPI(prim)
            color = light_api.GetColorAttr().Get()
            lights.append({
                "prim_path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "intensity": float(light_api.GetIntensityAttr().Get() or 0.0),
                "exposure": float(light_api.GetExposureAttr().Get() or 0.0),
                "color": vec3(color) if color is not None else None,
            })
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = vec3(world_range.GetMin())
        maximum = vec3(world_range.GetMax())
        rows.append({
            "prim_path": str(prim.GetPath()),
            "vertex_count": len(points),
            "face_count": len(counts),
            "world_min": minimum,
            "world_max": maximum,
            "world_size": [maximum[i] - minimum[i] for i in range(3)],
        })
    rows.sort(key=lambda row: (-row["world_max"][2], -row["face_count"]))
    payload = {
        "asset": str(args.asset.resolve()),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "mesh_count": len(rows),
        "light_count": len(lights),
        "lights": lights,
        "meshes": rows,
    }
    print(f"USD mesh inspection: {payload['asset']}")
    print(
        f"up={payload['stage_up_axis']}, meters/unit={payload['stage_meters_per_unit']}, "
        f"meshes={len(rows)}, lights={len(lights)}"
    )
    for index, row in enumerate(lights, start=1):
        print(
            f"[LIGHT {index:02d}] {row['prim_path']} type={row['type']} "
            f"intensity={row['intensity']} exposure={row['exposure']} "
            f"color={row['color']}"
        )
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index:03d}] {row['prim_path']} vertices={row['vertex_count']} "
            f"faces={row['face_count']} min={row['world_min']} "
            f"max={row['world_max']}"
        )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {args.output_json}")
    if not rows:
        raise RuntimeError("No UsdGeom.Mesh prims found in asset")


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
