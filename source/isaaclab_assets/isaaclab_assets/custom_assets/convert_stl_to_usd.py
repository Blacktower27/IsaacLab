# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert STL assets (mm) to USD (meters) for Isaac Lab custom assets.

Per-asset settings (mass, collision type, etc.) are read from a
``conversion_config.yaml`` file placed next to the STL files.
If no config is found, CLI defaults are used.

Usage — batch-convert a directory (reads conversion_config.yaml if present):
    ./isaaclab.sh -p \\
        source/isaaclab_assets/isaaclab_assets/custom_assets/convert_stl_to_usd.py \\
        --dir source/isaaclab_assets/isaaclab_assets/custom_assets/box/middle

Usage — single file (uses CLI args, ignores config):
    ./isaaclab.sh -p \\
        source/isaaclab_assets/isaaclab_assets/custom_assets/convert_stl_to_usd.py \\
        <input.stl> <output.usd> [--mass FLOAT] [--collision-approximation STR]

conversion_config.yaml format:
    default:
      scale: 0.001
      make_instanceable: true
      collision_approximation: convexDecomposition
    assets:
      Small_Box.stl:
        mass: 0.1
      Lid_Yellow.stl:
        mass: 0.02
        collision_approximation: convexHull
"""

"""
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/custom_assets/convert_stl_to_usd.py source/isaaclab_assets/isaaclab_assets/custom_assets/box/middle/Small_Box.stl
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/custom_assets/convert_stl_to_usd.py source/isaaclab_assets/isaaclab_assets/custom_assets/box/middle/Lid_Yellow.stl
"""

import argparse
import os

from isaaclab.app import AppLauncher

_COLLISION_CHOICES = [
    "convexDecomposition",
    "convexHull",
    "triangleMesh",
    "meshSimplification",
    "sdf",
    "boundingCube",
    "boundingSphere",
    "none",
]

parser = argparse.ArgumentParser(
    description="Convert STL (mm) to USD (m) for Isaac Lab custom assets."
)
parser.add_argument(
    "input", nargs="?", default=None, help="Path to input .stl file."
)
parser.add_argument(
    "output", nargs="?", default=None, help="Path to output .usd file."
)
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    help="Batch-convert all .stl files found in this directory.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=0.001,
    help="Uniform scale factor (default: 0.001 = mm to meters).",
)
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for multi-env cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=_COLLISION_CHOICES,
    help="Collision mesh approximation method (default: convexDecomposition).",
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="Mass in kg (default: None).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after Isaac Sim starts ---
import yaml  # noqa: E402

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402

_COLLISION_CFG_MAP = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    "sdf": schemas_cfg.SDFMeshPropertiesCfg,
    "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
    "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
    "none": None,
}

_CONFIG_FILENAME = "conversion_config.yaml"


def load_dir_config(directory: str) -> dict:
    """Load conversion_config.yaml from directory, return empty dict if absent."""
    cfg_path = os.path.join(directory, _CONFIG_FILENAME)
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    print(f"[config] Loaded: {cfg_path}")
    return data


def resolve_asset_cfg(fname: str, dir_cfg: dict) -> dict:
    """Merge default section with per-asset overrides from config."""
    defaults = dir_cfg.get("default", {})
    overrides = (dir_cfg.get("assets") or {}).get(fname, {})
    merged = {**defaults, **overrides}
    # fall back to CLI args for anything not set in config
    return {
        "scale": merged.get("scale", args_cli.scale),
        "make_instanceable": merged.get(
            "make_instanceable", args_cli.make_instanceable
        ),
        "collision_approximation": merged.get(
            "collision_approximation", args_cli.collision_approximation
        ),
        "mass": merged.get("mass", args_cli.mass),
        # Collision contact/rest offsets (None → PhysX default)
        "contact_offset": merged.get("contact_offset", None),
        "rest_offset": merged.get("rest_offset", None),
        # ConvexDecomposition precision params (None → PhysX default)
        "hull_vertex_limit": merged.get("hull_vertex_limit", None),
        "max_convex_hulls": merged.get("max_convex_hulls", None),
        "voxel_resolution": merged.get("voxel_resolution", None),
        "error_percentage": merged.get("error_percentage", None),
        "shrink_wrap": merged.get("shrink_wrap", None),
        "min_thickness": merged.get("min_thickness", None),
    }


def convert_one(stl_path: str, usd_path: str, cfg: dict) -> None:
    """Convert a single STL to USD using the given per-asset config."""
    stl_path = os.path.abspath(stl_path)
    usd_path = os.path.abspath(usd_path)

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"STL not found: {stl_path}")

    s = float(cfg["scale"])
    scale = (s, s, s)
    mass_val = cfg["mass"]
    collision_approx = cfg["collision_approximation"]

    mass_props = (
        schemas_cfg.MassPropertiesCfg(mass=mass_val)
        if mass_val is not None
        else None
    )
    rigid_props = (
        schemas_cfg.RigidBodyPropertiesCfg() if mass_props is not None else None
    )
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=(collision_approx != "none"),
        contact_offset=cfg.get("contact_offset"),
        rest_offset=cfg.get("rest_offset"),
    )
    cfg_cls = _COLLISION_CFG_MAP[collision_approx]
    if cfg_cls is None:
        collision_cfg = None
    elif cfg_cls is schemas_cfg.ConvexDecompositionPropertiesCfg:
        # Build kwargs, omitting keys that are None so PhysX defaults are preserved
        # for any parameter the user did not explicitly set.
        cd_kwargs = {
            k: cfg[k]
            for k in (
                "hull_vertex_limit",
                "max_convex_hulls",
                "voxel_resolution",
                "error_percentage",
                "shrink_wrap",
                "min_thickness",
            )
            if cfg.get(k) is not None
        }
        collision_cfg = cfg_cls(**cd_kwargs)
    else:
        collision_cfg = cfg_cls()

    mesh_cfg = MeshConverterCfg(
        asset_path=stl_path,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        force_usd_conversion=True,
        make_instanceable=cfg["make_instanceable"],
        scale=scale,
        mass_props=mass_props,  # type: ignore[arg-type]
        rigid_props=rigid_props,  # type: ignore[arg-type]
        collision_props=collision_props,
        mesh_collision_props=collision_cfg,  # type: ignore[arg-type]
    )

    print(f"[convert] {os.path.basename(stl_path)}")
    print(f"          scale={scale}, mass={mass_val} kg, "
          f"collision={collision_approx}, "
          f"instanceable={cfg['make_instanceable']}")
    if collision_approx == "convexDecomposition":
        cd_info = {k: cfg[k] for k in
                   ("hull_vertex_limit", "max_convex_hulls", "voxel_resolution",
                    "error_percentage", "shrink_wrap", "min_thickness")
                   if cfg.get(k) is not None}
        if cd_info:
            print(f"          convexDecomp params: {cd_info}")
    contact_info = {k: cfg[k] for k in ("contact_offset", "rest_offset")
                    if cfg.get(k) is not None}
    if contact_info:
        print(f"          collision offsets: {contact_info}")
    converter = MeshConverter(mesh_cfg)

    # Apply ArticulationRootAPI so IsaacLab's ArticulationCfg can load this USD.
    # MeshConverter only adds RigidBodyAPI/CollisionAPI; ArticulationRootAPI must
    # be added manually or spawn_from_usd silently skips it (modify vs define).
    actual_usd_path = converter.usd_path
    stage = Usd.Stage.Open(actual_usd_path)
    articulation_root_prim = None
    for prim in stage.Traverse():
        if UsdPhysics.RigidBodyAPI(prim):
            articulation_root_prim = prim
            break
    if articulation_root_prim is None:
        # Fallback: apply to the default prim
        articulation_root_prim = stage.GetDefaultPrim()
    if articulation_root_prim and articulation_root_prim.IsValid():
        UsdPhysics.ArticulationRootAPI.Apply(articulation_root_prim)
        PhysxSchema.PhysxArticulationAPI.Apply(articulation_root_prim)
        stage.GetRootLayer().Save()
        print(f"[articulation] Applied ArticulationRootAPI to "
              f"<{articulation_root_prim.GetPath()}>")
    else:
        print("[warn] Could not find a prim to apply ArticulationRootAPI — "
              "apply it manually in USD Composer if needed.")

    print(f"[done]    -> {actual_usd_path}\n")


def batch_convert(directory: str) -> None:
    dir_cfg = load_dir_config(directory)
    stl_files = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".stl")
    )
    if not stl_files:
        print(f"[warn] No .stl files found in: {directory}")
        return
    print(f"[batch] {len(stl_files)} STL file(s) in {directory}\n")
    for fname in stl_files:
        stl = os.path.join(directory, fname)
        usd = os.path.splitext(stl)[0] + ".usd"
        asset_cfg = resolve_asset_cfg(fname, dir_cfg)
        convert_one(stl, usd, asset_cfg)


def main() -> None:
    if args_cli.dir is not None:
        batch_convert(args_cli.dir)
    elif args_cli.input is not None:
        usd_out = (
            args_cli.output
            if args_cli.output is not None
            else os.path.splitext(args_cli.input)[0] + ".usd"
        )
        fname = os.path.basename(args_cli.input)
        directory = os.path.dirname(os.path.abspath(args_cli.input))
        dir_cfg = load_dir_config(directory)
        asset_cfg = resolve_asset_cfg(fname, dir_cfg)
        convert_one(args_cli.input, usd_out, asset_cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    simulation_app.close()
