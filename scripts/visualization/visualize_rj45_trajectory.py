"""
Visualize RJ45 insertion trajectories in 3D with matplotlib.

Layout
------
Each plot shows:
  - Female RJ45 socket (green, semi-transparent) at the target (world-origin)
  - Male RJ45 plug ghost at START pose  (red, semi-transparent)
  - Male RJ45 plug ghost at TARGET pose (blue, semi-transparent)
  - Phased spline trajectory: arc phase (blue line) + vertical descent (orange line)
  - Coordinate frames (X=red, Y=green, Z=blue arrows) at start, 10 intermediates, end

Usage
-----
    python scripts/visualization/visualize_rj45_trajectory.py
    python scripts/visualization/visualize_rj45_trajectory.py --output-dir /tmp --n-trajectories 40 --seed-offset 0
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.max_open_warning"] = 0
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline
from stl import mesh as stl_mesh

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved at runtime)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STL_FEMALE = os.path.join(
    _REPO_ROOT,
    "source/isaaclab_assets/isaaclab_assets/custom_assets/rj45/medium/rs_female_rj45.stl",
)
STL_MALE = os.path.join(
    _REPO_ROOT,
    "source/isaaclab_assets/isaaclab_assets/custom_assets/rj45/medium/rs_male_rj45.stl",
)

# RJ45 geometry constants (all values from forge_tasks_cfg.py / factory_utils.py)
_FEMALE_BOTTOM_Z = 0.05145   # socket USD origin world Z (= |STL Z_min|); opening face at top

# Full-insertion target from forge_tasks_cfg.py (single source of truth)
_SOCKET_TARGET_Y_LOCAL = -0.006  # socket_target_y_local: cavity centre Y offset from socket USD
_SOCKET_TARGET_Z_LOCAL =  0.014  # socket_target_z_local: held-base (tip) Z in socket-local frame
_MALE_TIP_OFFSET       = -0.003  # RJ45MaleCfg.base_height: tip 3 mm below male USD origin

# Male USD origin world Z at full insertion (visualization):
#   We show the plug 35 mm inside the socket cavity for a clear visual insertion.
#     tip_world  = _FEMALE_BOTTOM_Z - 0.035 = 0.05145 - 0.035 = 0.01645 m
#     male_origin = tip_world - _MALE_TIP_OFFSET = 0.01645 + 0.003 = 0.01945 m
#   → 35 mm of tip inside socket body → unmistakably inserted.
_MALE_USD_TARGET_Z = _FEMALE_BOTTOM_Z - 0.035 + abs(_MALE_TIP_OFFSET)
# = 0.05145 - 0.035 + 0.003 = 0.01945 m  (tip at 0.01645 m = 35 mm inside socket)


# ---------------------------------------------------------------------------
# 1. Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PoseParams:
    """A 4-DOF pose: position (m) + yaw (rad). Pitch and roll are always 0."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


@dataclass
class TrajectoryConfig:
    """Parameters controlling trajectory shape and sampling."""
    # Contact-init: tip aligned with socket opening face.
    #   Z_start ≈ _FEMALE_BOTTOM_Z + |_MALE_TIP_OFFSET| = 0.05145 + 0.003 = 0.054 m
    # Male target Z = _MALE_USD_TARGET_Z ≈ 0.019 m  (tip 35 mm inside socket)
    # approach_z_offset chosen so approach.z ≈ Z_start:
    #   approach.z = 0.019 + 0.035 = 0.054 m  ← matches contact-init Z
    # Phase-1 (arc, blue):  XY/yaw alignment at constant Z ≈ socket-opening level
    # Phase-2 (descent, orange): straight 35 mm downward into socket cavity
    approach_z_offset: float = 0.035  # m above male target z before vertical descent
    n_intermediate_frames: int = 10   # coord frames at intermediate waypoints
    n_phase1_pts: int = 60            # spline resolution – arc phase
    n_phase2_pts: int = 25            # resolution – vertical descent phase


@dataclass
class RenderConfig:
    """Visual parameters for the 3D plot."""
    stl_scale: float = 0.001          # STL files are in mm → convert to metres
    alpha_stl: float = 0.30           # mesh transparency
    alpha_stl_start: float = 0.20     # slightly more transparent for start ghost
    view_elev: float = 20.0           # 3D camera elevation (degrees)
    view_azim: float = -90.0          # look along X axis → Y-Z plane shows Y insertion
    frame_len: float = 0.012          # coordinate axis arrow length (m)
    frame_alpha: float = 0.9
    fig_dpi: int = 120
    fig_size: Tuple[float, float] = field(default_factory=lambda: (6.0, 5.0))
    traj_lw: float = 1.6              # trajectory line width


# [OLD] Generic random-offset start sampling — replaced by contact-init geometry below.
# @dataclass
# class StartRangeConfig:
#     """Random start-pose tolerance around the target."""
#     dx_range: float = 0.060           # ±m in X
#     dy_range: float = 0.060           # ±m in Y
#     dz_min: float = 0.101             # m above target z (minimum)
#     dz_max: float = 0.181             # m above target z (maximum)
#     dyaw_range: float = 0.524         # ±rad (≈ ±30°)


@dataclass
class ContactInitConfig:
    """Contact-init parameters mirroring forge_tasks_cfg.py → RJ45Insert.

    The contact-init algorithm places the male plug so that a point on its
    *bottom patch* (in male-local frame) coincides with a point on the female
    socket's *rear edge* (in female-local frame).

    Female-local frame origin = socket USD origin = socket opening face.
    Male-local frame origin   = male USD origin   = connector mating face.
    """
    # Female socket rear-edge contact region (female-local frame, metres)
    female_rear_edge_x_range_local: Tuple[float, float] = (-0.020, 0.020)
    female_rear_edge_y_local: float = -0.0186   # rear wall Y inside socket
    female_rear_edge_z_local: float = 0.0       # at socket opening face

    # Male plug bottom-patch contact region (male-local frame, metres)
    # Use physical TIP offset (= RJ45MaleCfg.base_height = -0.003) so the contact-init
    # position shows the connector tip aligned with the socket opening face.
    # (Sim uses -0.060 as a virtual approach-distance control point; for visualization
    # the tip is the natural physical contact reference.)
    male_bottom_patch_x_range_local: Tuple[float, float] = (-0.018, 0.018)
    male_bottom_patch_y_range_local: Tuple[float, float] = (-0.006, 0.009)
    male_bottom_patch_z_local: float = _MALE_TIP_OFFSET  # -0.003: physical connector tip

    # Orientation noise around the female socket's yaw (degrees)
    contact_init_yaw_range_deg: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class BatchConfig:
    """Batch generation and PDF layout parameters."""
    n_trajectories: int = 40
    pdf_cols: int = 2
    images_per_page: int = 6
    output_dir: str = os.path.join(_REPO_ROOT, "scripts/visualization")
    pdf_name: str = "rj45_trajectories.pdf"


# ---------------------------------------------------------------------------
# 2. STL loading
# ---------------------------------------------------------------------------

def load_stl_vertices(path: str, scale: float) -> np.ndarray:
    """Load an STL file and return scaled face vertices as (N, 3, 3) array.

    Args:
        path:  Absolute path to the .stl file.
        scale: Multiplicative scale applied to all coordinates (e.g. 0.001 for mm→m).

    Returns:
        ndarray of shape (N_faces, 3, 3) where last dim is XYZ.
    """
    m = stl_mesh.Mesh.from_file(path)
    # m.vectors has shape (N, 3, 3): N triangles, 3 vertices, XYZ
    return m.vectors.copy() * scale


# ---------------------------------------------------------------------------
# 3. Mesh transformation
# ---------------------------------------------------------------------------

def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
    """3×3 rotation matrix for a pure yaw (rotation around Z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def apply_pose_to_mesh(vertices: np.ndarray, pose: PoseParams) -> np.ndarray:
    """Rotate (yaw only) and translate mesh vertices.

    Args:
        vertices: (N, 3, 3) array of face vertices in local frame.
        pose:     Target pose in world frame.

    Returns:
        (N, 3, 3) transformed vertices.
    """
    R = _yaw_rotation_matrix(pose.yaw)
    # Reshape to (N*3, 3), rotate, reshape back
    flat = vertices.reshape(-1, 3)
    rotated = (R @ flat.T).T
    translated = rotated + np.array([pose.x, pose.y, pose.z])
    return translated.reshape(vertices.shape)


# ---------------------------------------------------------------------------
# 4. Start-pose sampling
# ---------------------------------------------------------------------------

# [OLD] Generic random-offset sampling — replaced by sample_contact_init_pose.
# def sample_start_pose(
#     target: PoseParams,
#     ranges: StartRangeConfig,
#     rng: np.random.Generator,
# ) -> PoseParams:
#     dx = rng.uniform(-ranges.dx_range, ranges.dx_range)
#     dy = rng.uniform(-ranges.dy_range, ranges.dy_range)
#     dz = rng.uniform(ranges.dz_min, ranges.dz_max)
#     dyaw = rng.uniform(-ranges.dyaw_range, ranges.dyaw_range)
#     return PoseParams(
#         x=target.x + dx,
#         y=target.y + dy,
#         z=target.z + dz,
#         yaw=target.yaw + dyaw,
#     )


def sample_contact_init_pose(
    target: PoseParams,
    cfg: ContactInitConfig,
    rng: np.random.Generator,
) -> PoseParams:
    """Sample a male-plug start pose using the same contact-init geometry as the sim.

    Replicates forge_env.py's contact-init logic (lines ~1296-1340):

        female_point_world = R(target.yaw) * female_rear_edge_local + target.xyz
        held_contact_pos   = female_point_world - R(yaw) * male_bottom_patch_local

    where *target* is the female socket world pose (USD origin = socket opening face).

    Args:
        target: Female socket world pose — also the male plug pose at full insertion.
        cfg:    ContactInitConfig with ranges matching forge_tasks_cfg.py.
        rng:    Seeded RNG for reproducibility.

    Returns:
        PoseParams for the male plug USD origin at contact-init.
    """
    # 1. Sample relative yaw of held plug w.r.t. female socket
    yaw = rng.uniform(*np.deg2rad(cfg.contact_init_yaw_range_deg))

    # 2. Female rear-edge contact point in female-local frame → world frame
    fx = rng.uniform(*cfg.female_rear_edge_x_range_local)
    fy = cfg.female_rear_edge_y_local
    fz = cfg.female_rear_edge_z_local
    R_fem = _yaw_rotation_matrix(target.yaw)   # female socket orientation
    female_world = R_fem @ np.array([fx, fy, fz]) + np.array([target.x, target.y, target.z])

    # 3. Male bottom-patch contact point in male-local frame
    mx = rng.uniform(*cfg.male_bottom_patch_x_range_local)
    my = rng.uniform(*cfg.male_bottom_patch_y_range_local)
    mz = cfg.male_bottom_patch_z_local          # = -0.060 m
    male_local = np.array([mx, my, mz])

    # 4. Male USD origin = female contact point minus rotated male contact offset
    #    held_contact_pos = female_world - R(yaw) * male_local
    R_held = _yaw_rotation_matrix(yaw)
    held_pos = female_world - R_held @ male_local

    return PoseParams(x=float(held_pos[0]), y=float(held_pos[1]),
                      z=float(held_pos[2]), yaw=float(yaw))


# ---------------------------------------------------------------------------
# 5. Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectory(
    start: PoseParams,
    target: PoseParams,
    cfg: TrajectoryConfig,
) -> List[PoseParams]:
    """Generate a two-phase insertion trajectory.

    Phase 1 – Cubic-spline arc from *start* to the *approach* waypoint
              (directly above target, at target.z + approach_z_offset).
              Yaw interpolates linearly; pitch/roll fixed at 0.

    Phase 2 – Straight vertical descent from approach to target.
              X, Y, yaw are constant; only Z decreases.

    Args:
        start:  Start pose.
        target: Final insertion pose.
        cfg:    Trajectory configuration.

    Returns:
        List of PoseParams (ordered start→target), length n_phase1_pts + n_phase2_pts.
    """
    approach = PoseParams(
        x=target.x,
        y=target.y,
        z=target.z + cfg.approach_z_offset,
        yaw=target.yaw,
    )

    # ---- Phase 1: cubic spline through [start, midpoint, approach] ----------
    # We add one heuristic midpoint to give the spline a gentle curve.
    mid = PoseParams(
        x=(start.x + approach.x) / 2.0,
        y=(start.y + approach.y) / 2.0,
        z=max(start.z, approach.z) + 0.01,  # slight upward bow
        yaw=start.yaw + (approach.yaw - start.yaw) * 0.5,
    )

    # Parameterise by cumulative chord length for a natural spline
    pts_xyz = np.array([[start.x, start.y, start.z],
                        [mid.x,   mid.y,   mid.z],
                        [approach.x, approach.y, approach.z]])
    diffs = np.diff(pts_xyz, axis=0)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    t_norm = chord / chord[-1]  # [0, t_mid, 1]

    cs_x = CubicSpline(t_norm, pts_xyz[:, 0])
    cs_y = CubicSpline(t_norm, pts_xyz[:, 1])
    cs_z = CubicSpline(t_norm, pts_xyz[:, 2])

    t_eval = np.linspace(0.0, 1.0, cfg.n_phase1_pts)
    yaw_phase1 = np.linspace(start.yaw, approach.yaw, cfg.n_phase1_pts)

    phase1 = [
        PoseParams(x=float(cs_x(t)), y=float(cs_y(t)), z=float(cs_z(t)), yaw=float(yaw))
        for t, yaw in zip(t_eval, yaw_phase1)
    ]

    # ---- Phase 2: linear vertical descent ------------------------------------
    z_vals = np.linspace(approach.z, target.z, cfg.n_phase2_pts)
    phase2 = [
        PoseParams(x=target.x, y=target.y, z=float(z), yaw=target.yaw)
        for z in z_vals
    ]

    # Skip the duplicated approach point at the start of phase2
    return phase1 + phase2[1:]


# ---------------------------------------------------------------------------
# 6. Frame-waypoint selection
# ---------------------------------------------------------------------------

def select_frame_waypoints(traj: List[PoseParams], n_intermediate: int) -> List[PoseParams]:
    """Return start + n_intermediate evenly-spaced intermediate + end waypoints.

    Args:
        traj:          Full trajectory list (start→end).
        n_intermediate: Number of intermediate coord frames.

    Returns:
        List of length n_intermediate + 2.
    """
    indices = np.linspace(0, len(traj) - 1, n_intermediate + 2, dtype=int)
    return [traj[i] for i in indices]


# ---------------------------------------------------------------------------
# 7. Coordinate-frame drawing
# ---------------------------------------------------------------------------

def draw_coord_frame(
    ax,
    pose: PoseParams,
    length: float,
    alpha: float = 0.9,
    linewidth: float = 1.2,
) -> None:
    """Draw X (red), Y (green), Z (blue) arrows at *pose*.

    Applies yaw rotation; pitch/roll are 0.
    """
    R = _yaw_rotation_matrix(pose.yaw)
    origin = np.array([pose.x, pose.y, pose.z])

    axes_local = np.eye(3)  # columns are X, Y, Z unit vectors
    colors = ["red", "limegreen", "dodgerblue"]
    for i, color in enumerate(colors):
        direction = R @ axes_local[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=color, alpha=alpha, linewidth=linewidth,
            arrow_length_ratio=0.25,
        )


# ---------------------------------------------------------------------------
# 8. Single-trajectory plot
# ---------------------------------------------------------------------------

def plot_single_trajectory(
    seed: int,
    female_pose: PoseParams,
    male_target: PoseParams,
    contact_init_cfg: ContactInitConfig,
    traj_cfg: TrajectoryConfig,
    render_cfg: RenderConfig,
    male_verts: np.ndarray,
    female_verts: np.ndarray,
) -> plt.Figure:
    """Create one 3D matplotlib figure for a single seeded trajectory.

    Args:
        seed:             Integer seed (determines start pose deterministically).
        female_pose:      Female socket world pose (USD origin = socket opening face).
                          Used for socket rendering and contact-init sampling reference.
        male_target:      Male plug world pose at full insertion (USD origin).
                          Derived from factory_utils.get_target_held_base_pose.
        contact_init_cfg: Contact-init geometry config (mirrors forge_tasks_cfg.py).
        traj_cfg:         Trajectory generation config.
        render_cfg:       Rendering/visual config.
        male_verts:       (N,3,3) raw vertices of the male plug (local frame, metres).
        female_verts:     (N,3,3) raw vertices of the female socket (local frame, metres).

    Returns:
        matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    start = sample_contact_init_pose(female_pose, contact_init_cfg, rng)
    traj = generate_trajectory(start, male_target, traj_cfg)
    frame_poses = select_frame_waypoints(traj, traj_cfg.n_intermediate_frames)

    fig = plt.figure(figsize=render_cfg.fig_size, dpi=render_cfg.fig_dpi)
    ax = fig.add_subplot(111, projection="3d")

    # ---- Trajectory line (two colours for two phases) -----------------------
    n1 = traj_cfg.n_phase1_pts
    xs = [p.x for p in traj]
    ys = [p.y for p in traj]
    zs = [p.z for p in traj]
    ax.plot(xs[:n1], ys[:n1], zs[:n1],
            color="royalblue", lw=render_cfg.traj_lw, label="Arc phase", zorder=3)
    ax.plot(xs[n1 - 1:], ys[n1 - 1:], zs[n1 - 1:],
            color="darkorange", lw=render_cfg.traj_lw, label="Descent phase", zorder=3)

    # ---- Coordinate frames --------------------------------------------------
    for i, pose in enumerate(frame_poses):
        # Make start and end frames slightly larger for emphasis
        length = render_cfg.frame_len * (1.5 if i in (0, len(frame_poses) - 1) else 1.0)
        draw_coord_frame(ax, pose, length, alpha=render_cfg.frame_alpha)

    # ---- STL meshes ---------------------------------------------------------
    # Female socket: placed at female_pose (USD origin = socket opening face).
    # Socket body Z ∈ [0, _FEMALE_BOTTOM_Z] (bottom flush with table).
    fv = apply_pose_to_mesh(female_verts, female_pose)
    poly_female = Poly3DCollection(
        fv, alpha=render_cfg.alpha_stl,
        facecolor="mediumseagreen", edgecolor="none",
    )
    ax.add_collection3d(poly_female)

    # Male plug ghost at START / contact-init pose (red/salmon)
    mv_start = apply_pose_to_mesh(male_verts, start)
    poly_male_start = Poly3DCollection(
        mv_start, alpha=render_cfg.alpha_stl_start,
        facecolor="salmon", edgecolor="none",
    )
    ax.add_collection3d(poly_male_start)

    # Male plug ghost at TARGET / full insertion (steel-blue)
    # male_target = USD origin at (_SOCKET_TARGET_Y_LOCAL, _MALE_USD_TARGET_Z)
    # Connector tip (3 mm below USD origin) is at the sim success position.
    mv_target = apply_pose_to_mesh(male_verts, male_target)
    poly_male_target = Poly3DCollection(
        mv_target, alpha=render_cfg.alpha_stl,
        facecolor="steelblue", edgecolor="none",
    )
    ax.add_collection3d(poly_male_target)

    # ---- Axis limits (equal aspect, z always starts at 0 = table level) -----
    all_pts = np.vstack([
        mv_start.reshape(-1, 3),
        mv_target.reshape(-1, 3),
        fv.reshape(-1, 3),
        np.array([[p.x, p.y, p.z] for p in traj]),
    ])
    centre = all_pts.mean(axis=0)
    half_range = np.max(np.abs(all_pts - centre)) * 1.15
    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(-0.010, centre[2] + half_range)  # -10 mm shows floor below socket

    # ---- Table surface (z=0 reference plane) --------------------------------
    _r = half_range * 0.8
    _cx, _cy = centre[0], centre[1]
    _tx, _ty = np.meshgrid(
        [_cx - _r, _cx + _r], [_cy - _r, _cy + _r]
    )
    # Floor at Z = -0.002 (2 mm below socket bottom) so socket visually rests on table.
    ax.plot_surface(_tx, _ty, np.full_like(_tx, -0.002),
                    alpha=0.08, color="saddlebrown", linewidth=0, zorder=0)

    # ---- Labels and view ----------------------------------------------------
    ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
    ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
    ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_title(f"Seed {seed}", fontsize=8, pad=4)
    ax.view_init(elev=render_cfg.view_elev, azim=render_cfg.view_azim)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.5)

    fig.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# 9. Batch figure generation
# ---------------------------------------------------------------------------

def generate_all_figures(
    female_pose: PoseParams,
    male_target: PoseParams,
    contact_init_cfg: ContactInitConfig,
    traj_cfg: TrajectoryConfig,
    render_cfg: RenderConfig,
    batch_cfg: BatchConfig,
    male_verts: np.ndarray,
    female_verts: np.ndarray,
    seed_offset: int = 0,
) -> List[plt.Figure]:
    """Generate one figure per trajectory seed.

    Args:
        female_pose:  Female socket world pose (USD origin).
        male_target:  Male plug world pose at full insertion (USD origin).
        seed_offset:  Added to each seed index (useful for generating new batches).

    Returns:
        List of matplotlib Figures, length = batch_cfg.n_trajectories.
    """
    figures = []
    for i in range(batch_cfg.n_trajectories):
        seed = seed_offset + i
        fig = plot_single_trajectory(
            seed, female_pose, male_target, contact_init_cfg,
            traj_cfg, render_cfg, male_verts, female_verts,
        )
        figures.append(fig)
        print(f"  Rendered seed {seed} ({i + 1}/{batch_cfg.n_trajectories})", flush=True)
    return figures


# ---------------------------------------------------------------------------
# 10. PDF collage
# ---------------------------------------------------------------------------

def save_pdf_collage(figures: List[plt.Figure], batch_cfg: BatchConfig) -> str:
    """Pack all figures into a multi-page PDF.

    Layout: *pdf_cols* columns, *images_per_page* images per page.

    Args:
        figures:    List of matplotlib Figures (already rendered).
        batch_cfg:  Batch / layout config.

    Returns:
        Absolute path to the saved PDF.
    """
    os.makedirs(batch_cfg.output_dir, exist_ok=True)
    pdf_path = os.path.join(batch_cfg.output_dir, batch_cfg.pdf_name)

    cols = batch_cfg.pdf_cols
    ipp = batch_cfg.images_per_page
    rows = math.ceil(ipp / cols)

    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(figures), ipp):
            page_figs = figures[page_start: page_start + ipp]
            n = len(page_figs)
            actual_rows = math.ceil(n / cols)

            fig_page, axes = plt.subplots(
                actual_rows, cols,
                figsize=(cols * 6.5, actual_rows * 5.5),
                squeeze=False,
            )

            for ax_row in axes:
                for ax in ax_row:
                    ax.set_visible(False)

            for idx, src_fig in enumerate(page_figs):
                r, c = divmod(idx, cols)
                ax_dest = axes[r][c]
                ax_dest.set_visible(True)
                ax_dest.set_axis_off()

                # Rasterise the source figure into the destination axes
                src_fig.canvas.draw()
                buf = src_fig.canvas.buffer_rgba()
                img = np.asarray(buf)          # (H, W, 4) RGBA uint8
                ax_dest.imshow(img)

            fig_page.tight_layout(pad=0.4)
            pdf.savefig(fig_page, bbox_inches="tight")
            plt.close(fig_page)

    return pdf_path


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RJ45 insertion trajectory visualizations as a PDF."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the PDF (default: scripts/visualization/)",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=40,
        help="Number of trajectories / images to generate (default: 40)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Integer added to each seed index (default: 0)",
    )
    args = parser.parse_args()

    # --- Config ---------------------------------------------------------------
    # female_pose: female socket USD origin in world frame.
    #   Socket body Z ∈ [0, _FEMALE_BOTTOM_Z] (bottom flush with table).
    female_pose = PoseParams(x=0.0, y=0.0, z=_FEMALE_BOTTOM_Z, yaw=0.0)

    # male_target: male plug USD origin at full insertion (visualization).
    #   tip_world  = _FEMALE_BOTTOM_Z - 0.035 = 0.01645 m (35 mm inside socket cavity)
    #   USD_origin = tip_world + 0.003 = 0.01945 m
    male_target = PoseParams(
        x=0.0,
        y=_SOCKET_TARGET_Y_LOCAL,      # = -0.006 m
        z=_MALE_USD_TARGET_Z,          # = 0.01945 m (tip 35 mm inside socket)
        yaw=0.0,
    )

    contact_init_cfg = ContactInitConfig()   # matches forge_tasks_cfg.py → RJ45Insert
    traj_cfg = TrajectoryConfig()
    render_cfg = RenderConfig()
    batch_cfg = BatchConfig(n_trajectories=args.n_trajectories)
    if args.output_dir is not None:
        batch_cfg.output_dir = os.path.abspath(args.output_dir)

    # --- Load STL files -------------------------------------------------------
    print("Loading STL files …")
    for path in (STL_FEMALE, STL_MALE):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"STL file not found: {path}")

    female_verts = load_stl_vertices(STL_FEMALE, render_cfg.stl_scale)
    male_verts = load_stl_vertices(STL_MALE, render_cfg.stl_scale)
    print(f"  Female mesh: {len(female_verts)} faces")
    print(f"  Male mesh:   {len(male_verts)} faces")

    # --- Generate figures -----------------------------------------------------
    print(f"Generating {batch_cfg.n_trajectories} trajectory figures …")
    figures = generate_all_figures(
        female_pose, male_target, contact_init_cfg, traj_cfg, render_cfg, batch_cfg,
        male_verts, female_verts,
        seed_offset=args.seed_offset,
    )

    # --- Save PDF -------------------------------------------------------------
    print("Assembling PDF collage …")
    pdf_path = save_pdf_collage(figures, batch_cfg)

    for fig in figures:
        plt.close(fig)

    print(f"\nDone. PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
