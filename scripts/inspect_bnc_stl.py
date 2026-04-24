"""Print bounds of BNC small STL meshes (stdlib + numpy). STL does not encode parametric helix L-slot.

For reference trajectories, tune ``tip_z_target`` (and P1–P2) in ``collect_bnc_trajectories.py`` (2 phases, no helix in JSON).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_BNC = _REPO / "source/isaaclab_assets/isaaclab_assets/custom_assets/bnc/small"


def _read_binary_stl_vertices(path: Path, scale: float) -> np.ndarray:
    with path.open("rb") as f:
        f.read(80)
        (n_tr,) = struct.unpack("<I", f.read(4))
        verts: list = []
        for _ in range(n_tr):
            data = f.read(50)
            if len(data) < 50:
                break
            a = struct.unpack("<3f", data[12:24])
            b = struct.unpack("<3f", data[24:36])
            c = struct.unpack("<3f", data[36:48])
            verts.extend([a, b, c])
    return np.array(verts, dtype=np.float64) * float(scale)


def _report(name: str, v: np.ndarray) -> None:
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    print(f"=== {name} ===  vertices={len(v)}  scale→metres=0.001")
    print(f"  x [{lo[0]:.6f}, {hi[0]:.6f}]  y [{lo[1]:.6f}, {hi[1]:.6f}]  z [{lo[2]:.6f}, {hi[2]:.6f}]")
    r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    print(f"  r_xy  min {r.min():.6f}  max {r.max():.6f}")


def main() -> None:
    for fname in (
        "Adi_BNC_Simulation_Small_Female.stl",
        "Adi_BNC_Simulation_Small_Male.stl",
    ):
        p = _BNC / fname
        if not p.is_file():
            print(f"Missing {p}", file=sys.stderr)
            continue
        v = _read_binary_stl_vertices(p, 0.001)
        _report(fname, v)
    print(
        f"\nHelical lock: not in STL; ``collect_bnc_trajectories`` records **approach + axial** only. "
        f"Tune ``tip_z_target`` / ``n_phase1|2`` in {_REPO / 'scripts/collect_bnc_trajectories.py'}."
    )


if __name__ == "__main__":
    main()
