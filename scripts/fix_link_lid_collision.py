"""
Run this in Isaac Sim Script Editor after importing kuka_blue URDF.
Replaces link_lid's STL collision mesh with Lid_Yellow.usd reference.
"""

import omni.usd
from pxr import Usd

LID_USD_PATH = (
    "/home/blacktower27/code/python_code/CAM/IsaacLab/source/isaaclab_assets"
    "/isaaclab_assets/custom_assets/box/middle/Lid_Yellow.usd"
)

stage = omni.usd.get_context().get_stage()

# ── Step 1: find link_lid ─────────────────────────────────────────────────
link_lid_prim = None
for prim in stage.Traverse():
    if prim.GetName() == "link_lid":
        link_lid_prim = prim
        break

if link_lid_prim is None:
    print("ERROR: link_lid not found.")
else:
    print(f"[OK] link_lid at: {link_lid_prim.GetPath()}")

    # ── Step 2: print structure ───────────────────────────────────────────
    print("\n--- link_lid subtree ---")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "link_lid" in path:
            instance_flag = " [INSTANCEABLE]" if prim.IsInstance() else ""
            proxy_flag = " [PROXY]" if prim.IsInstanceProxy() else ""
            print(f"  {path}  [{prim.GetTypeName()}]{instance_flag}{proxy_flag}")

    # ── Step 3: disable instancing on link_lid so we can edit inside it ──
    if link_lid_prim.IsInstance():
        link_lid_prim.SetInstanceable(False)
        print("\n[OK] Disabled instancing on link_lid")
    else:
        # Walk up to find the instanceable ancestor
        ancestor = link_lid_prim.GetParent()
        while ancestor.IsValid():
            if ancestor.IsInstance():
                ancestor.SetInstanceable(False)
                print(f"\n[OK] Disabled instancing on ancestor: {ancestor.GetPath()}")
                break
            ancestor = ancestor.GetParent()

    # ── Step 4: find collisions scope ────────────────────────────────────
    collisions_path = link_lid_prim.GetPath().AppendChild("collisions")
    collisions_prim = stage.GetPrimAtPath(collisions_path)

    if not collisions_prim.IsValid():
        print(f"\nERROR: collisions not found at {collisions_path}")
        print("Check the printed structure above for the correct path.")
    else:
        print(f"\n[OK] collisions at: {collisions_path}")

        # ── Step 5: remove existing STL collision children ────────────────
        for child in list(collisions_prim.GetChildren()):
            print(f"  Removing: {child.GetPath()}")
            stage.RemovePrim(child.GetPath())

        # ── Step 6: add Lid_Yellow.usd as reference ───────────────────────
        lid_col_path = collisions_path.AppendChild("lid_collision")
        lid_col_prim = stage.DefinePrim(lid_col_path)
        lid_col_prim.GetReferences().AddReference(LID_USD_PATH)
        print(f"[OK] Added Lid_Yellow.usd reference at: {lid_col_path}")

        # ── Step 7: save ──────────────────────────────────────────────────
        stage.Save()
        print("\n[DONE] Stage saved.")
