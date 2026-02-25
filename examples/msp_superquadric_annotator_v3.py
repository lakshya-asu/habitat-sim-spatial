#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import magnum as mn

# --- Habitat-Sim examples viewer import ---
VIEWER_DIR = "/home/artemis/project/habitat-sim/examples"
if VIEWER_DIR not in sys.path:
    sys.path.insert(0, VIEWER_DIR)

# IMPORTANT: Use hover viewer base so we get HUD hover labels
from viewer_hover_v6 import HabitatSimInteractiveViewerHoverV5
from habitat_sim.utils.settings import default_sim_settings
from habitat_sim.logging import logger

DEFAULT_PARAM_FILE = "/tmp/hab_superquadric_params.json"


# =========================
# Superquadric math helpers
# =========================
def _sgn_pow(x: np.ndarray, e: float) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** e)


def superquadric_surface_points(m: float, n: float, res_eta: int = 22, res_omega: int = 44) -> np.ndarray:
    """
    Return surface points array shaped [W, E, 3] in local (-1..1) space.
    Superellipsoid parameterization.
    """
    m = max(2.0, float(m))
    n = max(2.0, float(n))
    eps1 = 2.0 / m
    eps2 = 2.0 / n

    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, res_eta)
    omega = np.linspace(-np.pi, np.pi, res_omega)
    ETA, OMEGA = np.meshgrid(eta, omega)

    ceta = np.cos(ETA)
    seta = np.sin(ETA)
    comg = np.cos(OMEGA)
    somg = np.sin(OMEGA)

    x = _sgn_pow(ceta, eps1) * _sgn_pow(comg, eps2)
    y = _sgn_pow(ceta, eps1) * _sgn_pow(somg, eps2)
    z = _sgn_pow(seta, eps1)

    return np.stack([x, y, z], axis=-1)  # [W,E,3]


def superquadric_inside_mask(points_local: np.ndarray, m: float, n: float) -> np.ndarray:
    """
    points_local: [...,3] in local coordinates where the superquadric is unit scale.
    For a superellipsoid:
      ( (|x|^(2/eps2) + |y|^(2/eps2))^(eps2/eps1) + |z|^(2/eps1) ) <= 1
    where eps1=2/m, eps2=2/n
    """
    m = max(2.0, float(m))
    n = max(2.0, float(n))
    eps1 = 2.0 / m
    eps2 = 2.0 / n

    x = np.abs(points_local[..., 0])
    y = np.abs(points_local[..., 1])
    z = np.abs(points_local[..., 2])

    a = np.power(x, 2.0 / eps2) + np.power(y, 2.0 / eps2)
    b = np.power(a, eps2 / eps1) + np.power(z, 2.0 / eps1)
    return b <= 1.0

def _vec3_components(v) -> Tuple[float, float, float]:
    """
    Try to extract (x, y, z) from:
      - Magnum Vector3-like objects with .x/.y/.z
      - Numpy / sequence of length ≥ 3
      - Callables that return one of the above (e.g. aabb.center() / aabb.min())
    """
    if v is None:
        raise ValueError("Vector is None")

    # If it's a method / callable (e.g. aabb.center), call it first
    if callable(v):
        v = v()

    # Magnum-style
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return float(v.x), float(v.y), float(v.z)

    # Sequence / numpy-style
    return float(v[0]), float(v[1]), float(v[2])

def approx_aabb_and_volume_world(
    pos: mn.Vector3,
    rot: mn.Matrix3x3,
    scale: mn.Vector3,
    m: float,
    n: float,
    aabb_res_eta: int = 26,
    aabb_res_omega: int = 52,
    mc_samples: int = 150000,
    rng_seed: int = 7,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """
    Compute:
      - approximate world AABB extents (min_xyz, max_xyz) by sampling surface points
      - approximate volume by Monte Carlo inside the local AABB [-sx,sx]x[-sy,sy]x[-sz,sz]
        then transforming is irrelevant for volume (rotation doesn't change volume).

    Returns:
      (min_xyz, max_xyz, volume)
    """
    # --- AABB from surface samples ---
    pts = superquadric_surface_points(m, n, res_eta=aabb_res_eta, res_omega=aabb_res_omega)  # [W,E,3]
    pts = pts.reshape(-1, 3)

    # scale locally
    pts_scaled = np.empty_like(pts, dtype=np.float32)
    pts_scaled[:, 0] = pts[:, 0] * float(scale.x)
    pts_scaled[:, 1] = pts[:, 1] * float(scale.y)
    pts_scaled[:, 2] = pts[:, 2] * float(scale.z)

    # transform to world: world = R * local + pos
    R = np.array([[rot[0][0], rot[0][1], rot[0][2]],
                  [rot[1][0], rot[1][1], rot[1][2]],
                  [rot[2][0], rot[2][1], rot[2][2]]], dtype=np.float32)
    p = np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float32)

    pts_world = (pts_scaled @ R.T) + p  # (N,3)

    mn_xyz = pts_world.min(axis=0)
    mx_xyz = pts_world.max(axis=0)

    min_xyz = (float(mn_xyz[0]), float(mn_xyz[1]), float(mn_xyz[2]))
    max_xyz = (float(mx_xyz[0]), float(mx_xyz[1]), float(mx_xyz[2]))

    # --- Volume via Monte Carlo in LOCAL space (rotation/translation invariant) ---
    rng = np.random.default_rng(rng_seed)

    sx, sy, sz = float(scale.x), float(scale.y), float(scale.z)
    bbox_vol = (2.0 * sx) * (2.0 * sy) * (2.0 * sz)

    xs = rng.uniform(-sx, sx, size=(mc_samples,))
    ys = rng.uniform(-sy, sy, size=(mc_samples,))
    zs = rng.uniform(-sz, sz, size=(mc_samples,))
    pts_scaled_local = np.stack([xs, ys, zs], axis=-1).astype(np.float32)

    pts_unit = np.empty_like(pts_scaled_local)
    pts_unit[:, 0] = pts_scaled_local[:, 0] / max(sx, 1e-8)
    pts_unit[:, 1] = pts_scaled_local[:, 1] / max(sy, 1e-8)
    pts_unit[:, 2] = pts_scaled_local[:, 2] / max(sz, 1e-8)

    inside = superquadric_inside_mask(pts_unit, m=m, n=n)
    frac = float(np.mean(inside))
    volume = bbox_vol * frac

    return min_xyz, max_xyz, float(volume)


# =========================
# Data structs
# =========================
@dataclass
class SuperquadricMarker:
    center: mn.Vector3
    rot: mn.Matrix3x3
    m: float
    n: float
    scale: mn.Vector3


@dataclass
class AnchorSelection:
    """
    Human-selected anchor object (via hover + keypress).
    """
    sid: int
    label: str
    center: mn.Vector3  # Habitat world frame (x,y,z)


# =========================
# Main annotator viewer (V2)
# =========================
class MSPSuperquadricAnnotatorV2(HabitatSimInteractiveViewerHoverV5):
    """
    Same MSP annotator as before, but now includes hover semantic labels in HUD
    via HabitatSimInteractiveViewerHoverV5.

    Hover labels:
      Move mouse over objects -> HUD shows label (chair/table/wall) if semantics loaded.

    NEW:
      - Press 'Y' while hovering an object to set the "anchor object"
      - Anchor sid + label + world XYZ are saved into the questions CSV.
    """

    def __init__(
        self,
        sim_settings: Dict,
        questions_csv: str,
        poses_csv: str,
        hm3d_root: str,
        split: str,
        param_file: str,
        start_index: int = 0,
    ):
        self.questions_csv = questions_csv
        self.poses_csv = poses_csv
        self.hm3d_root = hm3d_root
        self.split = split
        self.param_file = param_file

        self._last_param_mtime = 0.0
        self._params = {"m": 2.0, "n": 2.0, "scale": 0.15}

        # load data first
        self.questions: List[Dict[str, str]] = self._read_csv_dicts(self.questions_csv)
        if len(self.questions) == 0:
            raise RuntimeError(f"questions csv is empty: {self.questions_csv}")

        self.pose_map: Dict[str, Dict[str, float]] = self._read_pose_map(self.poses_csv)

        # index
        self.idx = max(0, min(int(start_index), len(self.questions) - 1))

        # resolve initial scene
        row0 = self.questions[self.idx]
        scene_id = row0["scene"].strip()
        glb0, nav0 = self._resolve_scene_paths(scene_id)

        # IMPORTANT: set sim_settings['scene'] BEFORE base viewer init
        self.sim_settings = dict(sim_settings)
        self.sim_settings["scene"] = glb0

        # init base hover viewer (creates simulator + window + hover HUD)
        super().__init__(self.sim_settings)

        # God mode state (free-fly)
        self.god_mode = False
        self.god_move_speed = 2.5
        K = self.Key
        self.pressed.update({K.Q: False, K.E: False})  # for god mode vertical

        # Superquadric state
        self.res_eta = 16
        self.res_omega = 32

        self.ghost_enabled = True
        self.ghost: Optional[SuperquadricMarker] = None
        self.placed: List[SuperquadricMarker] = []

        self.current_annotation: Optional[SuperquadricMarker] = None
        self.current_anchor: Optional[AnchorSelection] = None  # NEW

        # load params, load navmesh, set initial question/pose
        self._reload_params(force=True)
        self._maybe_load_navmesh(nav0)
        self._load_question_into_scene(self.idx, first_time=True)

        logger.info("[Sim] MSP Superquadric Annotator V2 loaded (with hover HUD). Press 'H' for help.")
        self.print_help_text()

    # -------------------------
    # CSV I/O
    # -------------------------
    def _read_csv_dicts(self, path: str) -> List[Dict[str, str]]:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
        return rows

    def _write_csv_dicts(self, path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        os.replace(tmp, path)

    def _read_pose_map(self, path: str) -> Dict[str, Dict[str, float]]:
        mp: Dict[str, Dict[str, float]] = {}
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                k = r["scene_floor"].strip()
                mp[k] = {
                    "x": float(r["init_x"]),
                    "y": float(r["init_y"]),
                    "z": float(r["init_z"]),
                    "angle": float(r["init_angle"]),
                }
        return mp

    # -------------------------
    # HM3D path resolver
    # -------------------------
    def _resolve_scene_paths(self, scene_id: str) -> Tuple[str, str]:
        scene_id = scene_id.strip()
        base_name = scene_id.split("-", 1)[1] if "-" in scene_id else scene_id
        base_dir = os.path.join(self.hm3d_root, self.split, scene_id)

        glb = os.path.join(base_dir, f"{base_name}.basis.glb")
        nav = os.path.join(base_dir, f"{base_name}.basis.navmesh")
        return glb, nav

    def _maybe_load_navmesh(self, nav_path: str) -> None:
        try:
            if nav_path and os.path.exists(nav_path):
                ok = self.sim.pathfinder.load_nav_mesh(nav_path)
                self.sim.navmesh_visualization = True
                logger.info(f"[viewer] Loaded NavMesh: {nav_path} (ok={ok})")
            else:
                logger.info(f"[viewer] NavMesh not found: {nav_path} (continuing)")
        except Exception as e:
            logger.warning(f"[viewer] NavMesh load failed (continuing): {e}")

    # -------------------------
    # Param polling
    # -------------------------
    def _reload_params(self, force: bool = False):
        try:
            st = os.stat(self.param_file)
            if (not force) and st.st_mtime <= self._last_param_mtime:
                return
            self._last_param_mtime = st.st_mtime
            with open(self.param_file, "r") as f:
                data = json.load(f)

            m = float(data.get("m", self._params["m"]))
            n = float(data.get("n", self._params["n"]))
            s = float(data.get("scale", self._params["scale"]))

            self._params["m"] = max(2.0, min(20.0, m))
            self._params["n"] = max(2.0, min(20.0, n))
            self._params["scale"] = max(0.05, min(2.0, s))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"[superquadric] Failed to read params: {e}")

    # -------------------------
    # God mode movement (no actions)
    # -------------------------
    def move_and_look(self, repetitions: int) -> None:
        if repetitions == 0:
            return

        if self.god_mode:
            self._god_move(repetitions)
            if self.mouse_grabber is not None:
                self.update_grab_position(self.previous_mouse_point)
            return

        super().move_and_look(repetitions)

    def _god_move(self, repetitions: int) -> None:
        node = self.default_agent.scene_node
        dt = 1.0 / self.fps
        speed = self.god_move_speed

        forward = node.transformation.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
        right = node.transformation.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
        up = mn.Vector3(0.0, 1.0, 0.0)

        if forward.length() > 1e-6:
            forward = forward.normalized()
        if right.length() > 1e-6:
            right = right.normalized()

        move = mn.Vector3(0.0, 0.0, 0.0)
        K = self.Key

        if self.pressed.get(K.W, False):
            move += forward
        if self.pressed.get(K.S, False):
            move -= forward
        if self.pressed.get(K.D, False):
            move += right
        if self.pressed.get(K.A, False):
            move -= right

        if self.pressed.get(K.E, False) or self.pressed.get(K.Z, False):
            move += up
        if self.pressed.get(K.Q, False) or self.pressed.get(K.X, False):
            move -= up

        if move.length() > 1e-6:
            move = move.normalized() * (speed * dt)

        for _ in range(int(repetitions)):
            node.translate_local(move)

    # -------------------------
    # Camera basis helpers
    # -------------------------
    def _camera_basis(self):
        cam_node = self.render_camera.render_camera.node
        T = cam_node.absolute_transformation()
        cam_pos = cam_node.absolute_translation
        forward = T.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
        right = T.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
        up = mn.Vector3(0.0, 1.0, 0.0)

        if forward.length() > 1e-6:
            forward = forward.normalized()
        if right.length() > 1e-6:
            right = right.normalized()
        return cam_pos, forward, right, up, T

    def _snap_ghost_in_front(self, dist: float = 1.5, up_offset: float = 0.1) -> None:
        cam_pos, forward, _right, _up, T = self._camera_basis()
        center = cam_pos + forward * dist + mn.Vector3(0.0, up_offset, 0.0)
        rot = T.rotation()

        self._reload_params(force=True)
        s = float(self._params["scale"])
        self.ghost = SuperquadricMarker(
            center=center,
            rot=rot,
            m=float(self._params["m"]),
            n=float(self._params["n"]),
            scale=mn.Vector3(s, s, s),
        )

    def _nudge_ghost(self, delta_world: mn.Vector3) -> None:
        if self.ghost is None:
            return
        self.ghost.center = self.ghost.center + delta_world

    # -------------------------
    # Draw helpers
    # -------------------------
    def _draw_sq_wireframe(
        self,
        debug_line_render,
        marker: SuperquadricMarker,
        color: mn.Color4,
        res_eta: int,
        res_omega: int,
    ) -> None:
        pts_grid = superquadric_surface_points(marker.m, marker.n, res_eta=res_eta, res_omega=res_omega)
        W, E, _ = pts_grid.shape

        M = mn.Matrix4.from_(marker.rot, marker.center)
        sx, sy, sz = float(marker.scale.x), float(marker.scale.y), float(marker.scale.z)

        def to_world(v: np.ndarray) -> mn.Vector3:
            local = mn.Vector3(float(v[0] * sx), float(v[1] * sy), float(v[2] * sz))
            return M.transform_point(local)

        debug_line_render.set_line_width(2.5)

        # latitude lines (eta)
        for j in range(E):
            prev = to_world(pts_grid[0, j])
            for i in range(1, W):
                cur = to_world(pts_grid[i, j])
                debug_line_render.draw_transformed_line(prev, cur, color)
                prev = cur

        # longitude lines (omega)
        for i in range(W):
            prev = to_world(pts_grid[i, 0])
            for j in range(1, E):
                cur = to_world(pts_grid[i, j])
                debug_line_render.draw_transformed_line(prev, cur, color)
                prev = cur

    def debug_draw(self):
        super().debug_draw()

        self._reload_params()
        if self.ghost is not None:
            self.ghost.m = float(self._params["m"])
            self.ghost.n = float(self._params["n"])
            s = float(self._params["scale"])
            self.ghost.scale = mn.Vector3(s, s, s)

        dlr = self.sim.get_debug_line_render()

        for mk in self.placed:
            self._draw_sq_wireframe(dlr, mk, mn.Color4(0.2, 1.0, 0.2, 1.0), res_eta=14, res_omega=28)

        if self.ghost_enabled and self.ghost is not None:
            self._draw_sq_wireframe(dlr, self.ghost, mn.Color4(0.0, 1.0, 1.0, 1.0), res_eta=self.res_eta, res_omega=self.res_omega)

    # -------------------------
    # Scene/question loading + advancing
    # -------------------------
    def _set_window_title_best_effort(self, title: str) -> None:
        try:
            if hasattr(self, "set_window_title"):
                self.set_window_title(title)
                return
        except Exception:
            pass
        try:
            if hasattr(self, "window") and self.window is not None and hasattr(self.window, "set_title"):
                self.window.set_title(title)
                return
        except Exception:
            pass

    def _load_question_into_scene(self, idx: int, first_time: bool = False) -> None:
        idx = max(0, min(idx, len(self.questions) - 1))
        row = self.questions[idx]

        scene_id = row["scene"].strip()
        floor = int(float(row.get("floor", 0)))
        question = row.get("msp_question", "").strip()

        glb, nav = self._resolve_scene_paths(scene_id)

        if not os.path.exists(glb):
            raise FileNotFoundError(f"Scene GLB not found: {glb}")

        if not first_time:
            try:
                self.sim_settings["scene"] = glb
                self.sim_settings["enable_physics"] = self.sim_settings.get("enable_physics", False)
                self.sim_settings["default_agent_navmesh"] = False
                self.reconfigure_sim()  # hover base rebuilds semantic map here too
            except Exception as e:
                raise RuntimeError(f"Failed to reconfigure sim to {glb}: {e}")

        self._maybe_load_navmesh(nav)

        key = f"{scene_id}_{floor}"
        pose = self.pose_map.get(key, None)
        if pose is not None:
            self._set_agent_pose(pose["x"], pose["y"], pose["z"], pose["angle"])
        else:
            logger.warning(f"[pose] missing init pose for {key} (keeping current)")
        agent_pos = self._agent_world_pos()
        if agent_pos:
            logger.info(f"[Scene Load] Agent spawned at: {agent_pos}")

        self.placed.clear()
        self.current_annotation = None
        self.current_anchor = None  # reset anchor on scene change
        self._snap_ghost_in_front(dist=1.5, up_offset=0.1)

        header = f"[Q {idx+1}/{len(self.questions)}] scene={scene_id} floor={floor} :: {question}"
        logger.info(header)
        self._set_window_title_best_effort(header)

    def _set_agent_pose(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        try:
            agent = self.sim.get_agent(0)
            st = agent.get_state()
            st.position = np.array([x, y, z], dtype=np.float32)

            # yaw about +Y axis
            half = 0.5 * float(yaw_rad)
            sin_h = math.sin(half)
            cos_h = math.cos(half)

            # quaternion for rotation around Y: (x, y, z, w) = (0, sin(h), 0, cos(h))
            st.rotation = [0.0, sin_h, 0.0, cos_h]

            agent.set_state(st)
            logger.info(f"[pose] set agent pose pos=({x:.3f},{y:.3f},{z:.3f}) yaw={yaw_rad:.3f} rad")

            # sanity check: read back
            st2 = agent.get_state()
            logger.info(f"[pose] rotation after set_state: {st2.rotation}")

        except Exception as e:
            logger.warning(f"[pose] failed to set agent pose: {e}")

    def _advance(self, save_current: bool) -> None:
        if save_current:
            ok = self._submit_current_annotation()
            if not ok:
                logger.info("[submit] Not saved (no annotation). Use P to place, then ENTER to submit.")
                return

        self.idx += 1
        if self.idx >= len(self.questions):
            logger.info("[done] Reached end of questions. Staying on last item.")
            self.idx = len(self.questions) - 1
            return

        self._load_question_into_scene(self.idx, first_time=False)
        self.redraw()

    # -------------------------
    # Anchor capture helpers
    # -------------------------
    def _capture_anchor_from_hover(self) -> None:
        """
        Capture the current hovered semantic object as the anchor.

        - Uses the semantic id currently under the mouse (_hover_sid)
        - Resolves the HM3D semantic object using either:
            * self._sem_obj_by_id[sid], or
            * self.sim.semantic_scene.objects[sid]
        - Extracts an approximate world-space center from the object's AABB
        - Stores the result in self.current_anchor (sid, label, center)
        """
        # 1) Get current sid under cursor
        try:
            if hasattr(self, "get_current_hover_sid"):
                sid = self.get_current_hover_sid()
            else:
                sid = getattr(self, "_hover_sid", -1)
        except Exception:
            sid = -1

        if sid is None or sid <= 0:
            logger.info("[anchor] No valid semantic id under cursor. Move mouse over an object first.")
            return

        # 2) Resolve semantic object either from _sem_obj_by_id or from semantic_scene.objects
        sem_map = getattr(self, "_sem_obj_by_id", None)
        scene = getattr(self.sim, "semantic_scene", None)
        objects = getattr(scene, "objects", None) if scene is not None else None

        obj = None
        # Prefer map if available
        if sem_map:
            obj = sem_map.get(int(sid))

        # Fallback: treat sid as index into objects
        if obj is None and objects and 0 <= sid < len(objects):
            obj = objects[sid]

        if obj is None:
            logger.warning(
                f"[anchor] Could not resolve semantic object for sid={sid}. "
                f"semantic_map_size={0 if not sem_map else len(sem_map)}, "
                f"num_objects={0 if not objects else len(objects)}"
            )
            return

        # 3) Label from category
        cat = getattr(obj, "category", None)
        if cat is not None:
            name = getattr(cat, "name", None)
            if callable(name):
                try:
                    label = str(name())
                except Exception:
                    label = "object"
            elif isinstance(name, str):
                label = name
            else:
                label = str(cat)
        else:
            label = "object"

        # 4) World-space center from AABB (robust to callables / Magnum / numpy)
        center = None
        aabb = getattr(obj, "aabb", None)

        if aabb is not None:
            try:
                # Try direct center attribute first (center or center())
                c_attr = getattr(aabb, "center", None)
                if c_attr is not None:
                    cx, cy, cz = _vec3_components(c_attr)
                    center = mn.Vector3(cx, cy, cz)
                else:
                    # Fallback: compute center from min and max
                    mn_min_attr = getattr(aabb, "min", None)
                    mn_max_attr = getattr(aabb, "max", None)
                    if mn_min_attr is not None and mn_max_attr is not None:
                        minx, miny, minz = _vec3_components(mn_min_attr)
                        maxx, maxy, maxz = _vec3_components(mn_max_attr)
                        center = mn.Vector3(
                            0.5 * (minx + maxx),
                            0.5 * (miny + maxy),
                            0.5 * (minz + maxz),
                        )
            except Exception as e:
                logger.warning(f"[anchor] AABB center extraction failed for sid={sid}, label='{label}': {e}")
                center = None

        if center is None:
            logger.warning(
                f"[anchor] Could not determine AABB center for sid={sid}, label='{label}'. "
                "Using (0,0,0) as a placeholder."
            )
            center = mn.Vector3(0.0, 0.0, 0.0)

        # 5) Store anchor
        self.current_anchor = AnchorSelection(
            sid=int(sid),
            label=str(label),
            center=center,
        )

        logger.info(
            f"[anchor] Selected anchor sid={sid}, label='{label}', "
            f"center=({center.x:.3f},{center.y:.3f},{center.z:.3f})"
        )
    # -------------------------
    # Save annotation to CSV
    # -------------------------
    def _submit_current_annotation(self) -> bool:
        if self.current_annotation is None:
            return False

        mk = self.current_annotation

        min_xyz, max_xyz, vol = approx_aabb_and_volume_world(
            pos=mk.center,
            rot=mk.rot,
            scale=mk.scale,
            m=mk.m,
            n=mk.n,
            mc_samples=150000,
        )

        base_fields = list(self.questions[0].keys())
        extra_cols = [
            "ann_ok", "ann_ts",
            "ann_pos_x", "ann_pos_y", "ann_pos_z",
            "ann_yaw_rad",
            "ann_m", "ann_n",
            "ann_scale_x", "ann_scale_y", "ann_scale_z",
            "ann_aabb_min_x", "ann_aabb_min_y", "ann_aabb_min_z",
            "ann_aabb_max_x", "ann_aabb_max_y", "ann_aabb_max_z",
            "ann_volume",
            # NEW: anchor information
            "anchor_sid", "anchor_label",
            "anchor_center_x", "anchor_center_y", "anchor_center_z",
        ]

        fields = base_fields[:]
        for c in extra_cols:
            if c not in fields:
                fields.append(c)

        r00 = float(mk.rot[0][0])
        r02 = float(mk.rot[0][2])
        yaw = math.atan2(-r02, r00)

        row = self.questions[self.idx]
        row["ann_ok"] = "1"
        row["ann_ts"] = f"{time.time():.6f}"
        row["ann_pos_x"] = f"{float(mk.center.x):.6f}"
        row["ann_pos_y"] = f"{float(mk.center.y):.6f}"
        row["ann_pos_z"] = f"{float(mk.center.z):.6f}"
        row["ann_yaw_rad"] = f"{yaw:.6f}"
        row["ann_m"] = f"{float(mk.m):.6f}"
        row["ann_n"] = f"{float(mk.n):.6f}"
        row["ann_scale_x"] = f"{float(mk.scale.x):.6f}"
        row["ann_scale_y"] = f"{float(mk.scale.y):.6f}"
        row["ann_scale_z"] = f"{float(mk.scale.z):.6f}"

        row["ann_aabb_min_x"] = f"{min_xyz[0]:.6f}"
        row["ann_aabb_min_y"] = f"{min_xyz[1]:.6f}"
        row["ann_aabb_min_z"] = f"{min_xyz[2]:.6f}"
        row["ann_aabb_max_x"] = f"{max_xyz[0]:.6f}"
        row["ann_aabb_max_y"] = f"{max_xyz[1]:.6f}"
        row["ann_aabb_max_z"] = f"{max_xyz[2]:.6f}"
        row["ann_volume"] = f"{vol:.8f}"

        # ---- NEW: write anchor info if available ----
        if self.current_anchor is not None:
            anch = self.current_anchor
            row["anchor_sid"] = str(anch.sid)
            row["anchor_label"] = anch.label
            row["anchor_center_x"] = f"{float(anch.center.x):.6f}"
            row["anchor_center_y"] = f"{float(anch.center.y):.6f}"
            row["anchor_center_z"] = f"{float(anch.center.z):.6f}"
        else:
            # ensure fields exist but empty
            row.setdefault("anchor_sid", "")
            row.setdefault("anchor_label", "")
            row.setdefault("anchor_center_x", "")
            row.setdefault("anchor_center_y", "")
            row.setdefault("anchor_center_z", "")

        self._write_csv_dicts(self.questions_csv, self.questions, fieldnames=fields)

        logger.info(
            f"[submit] saved idx={self.idx} :: pos=({row['ann_pos_x']},{row['ann_pos_y']},{row['ann_pos_z']}) "
            f"m={row['ann_m']} n={row['ann_n']} vol={row['ann_volume']}"
        )
        if row.get("anchor_sid", "") != "":
            logger.info(
                f"[submit] anchor sid={row['anchor_sid']} label='{row['anchor_label']}' "
                f"center=({row['anchor_center_x']},{row['anchor_center_y']},{row['anchor_center_z']})"
            )
        return True

    # -------------------------
    # Key bindings
    # -------------------------
    def key_press_event(self, event):
        key = event.key
        K = self.Key
        mods = event.modifiers
        shift = bool(mods & self.Modifier.SHIFT)
        alt = bool(mods & self.Modifier.ALT)

        if key == K.G:
            self.god_mode = not self.god_mode
            logger.info(f"[Sim] Command: god mode set to {self.god_mode}")
            event.accepted = True
            self.redraw()
            return

        if key == K.C:
            logger.warning("[viewer] Disabled 'c' (discrete collision detection can abort on some HM3D stages).")
            event.accepted = True
            self.redraw()
            return

        if (key == K.Z or key == K.X) and (not self.god_mode):
            logger.info("[viewer] Z/X ignored (use GOD MODE for vertical fly).")
            event.accepted = True
            self.redraw()
            return

        if key == K.B:
            self.ghost_enabled = not self.ghost_enabled
            logger.info(f"[superquadric] ghost_enabled={self.ghost_enabled}")
            event.accepted = True
            self.redraw()
            return

        if key == K.O and (not shift):
            self._snap_ghost_in_front(dist=1.5, up_offset=0.1)
            logger.info("[superquadric] snapped ghost in front of camera")
            event.accepted = True
            self.redraw()
            return

        if key == K.P and shift:
            pose = self._agent_world_pos()
            if pose:
                x, y, z, yaw = pose
                logger.info(f"[Pose Check] Agent XYZ: ({x:.3f}, {y:.3f}, {z:.3f}) | Yaw: {yaw:.3f} rad")
            event.accepted = True
            return

        # NEW: capture anchor object under cursor
        if key == K.Y:
            self._capture_anchor_from_hover()
            event.accepted = True
            self.redraw()
            return

        if shift and self.ghost is not None:
            _cam_pos, forward, right, up, _T = self._camera_basis()
            step = 0.06 if not alt else 0.015

            if key == K.I:
                self._nudge_ghost(forward * step)
            elif key == K.K:
                self._nudge_ghost(forward * (-step))
            elif key == K.J:
                self._nudge_ghost(right * (-step))
            elif key == K.L:
                self._nudge_ghost(right * step)
            elif key == K.U:
                self._nudge_ghost(up * step)
            elif key == K.O:
                self._nudge_ghost(up * (-step))
            else:
                super().key_press_event(event)
                return

            logger.info(f"[superquadric] ghost pos={self.ghost.center}")
            event.accepted = True
            self.redraw()
            return

        if key == K.P:
            if self.ghost is None:
                self._snap_ghost_in_front(dist=1.5, up_offset=0.1)

            if self.ghost is not None:
                mk = SuperquadricMarker(
                    center=self.ghost.center,
                    rot=self.ghost.rot,
                    m=self.ghost.m,
                    n=self.ghost.n,
                    scale=self.ghost.scale,
                )
                self.placed.append(mk)
                self.current_annotation = mk
                logger.info(
                    f"[superquadric] placed #{len(self.placed)} (annotation=last) at {mk.center} "
                    f"| m={mk.m:.2f}, n={mk.n:.2f}, scale={mk.scale}"
                )

            event.accepted = True
            self.redraw()
            return

        if hasattr(K, "ENTER") and key == K.ENTER:
            event.accepted = True
            self._advance(save_current=True)
            return
        if hasattr(K, "RETURN") and key == K.RETURN:
            event.accepted = True
            self._advance(save_current=True)
            return

        if key == K.N:
            event.accepted = True
            logger.info("[skip] advancing without save")
            self._advance(save_current=False)
            return

        super().key_press_event(event)

    def print_help_text(self) -> None:
        super().print_help_text()
        logger.info(
            """
================ MSP Superquadric Annotator V2 ================
Hover labels (HUD):
  Move mouse over objects -> shows semantic label if semantics loaded.

Controls:
    G : GOD MODE free-fly (WASD + QE, also ZX)
    B : Toggle GHOST superquadric preview
    O : Snap ghost in front of camera
    SHIFT+I/J/K/L/U/O : Nudge ghost forward/left/back/right/up/down
        (+ALT) fine step

    P : Place/commit superquadric (green wireframe). Last placed = annotation
    ENTER/RETURN : Submit (compute AABB+volume, write to CSV) and advance
    N : Skip (no save) and advance

    Y : Set "anchor object" to the currently hovered semantic object
        (saves sid + label + world XYZ into CSV)

    C : Disabled (prevents Bullet abort on some HM3D stages)
==============================================================
"""
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-csv", required=True, help="questions_msp_sample_1.csv")
    parser.add_argument("--poses-csv", required=True, help="scene_init_poses_semantic_only.csv")
    parser.add_argument("--hm3d-root", required=True, help="HM3D root, e.g. /home/.../datasets/hm3d")
    parser.add_argument("--split", default="train", help="train/val/minival etc")

    # IMPORTANT: dataset config is required for HM3D semantics/hover
    parser.add_argument("--dataset", required=True, help="Path to hm3d_annotated_*_basis.scene_dataset_config.json")

    parser.add_argument("--param-file", default=DEFAULT_PARAM_FILE, help="JSON param file written by sliders UI")
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--disable-physics", action="store_true")
    parser.add_argument("--use-default-lighting", action="store_true")
    args = parser.parse_args()

    sim_settings = dict(default_sim_settings)
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["use_default_lighting"] = args.use_default_lighting
    sim_settings["enable_batch_renderer"] = False
    sim_settings["num_environments"] = 1
    sim_settings["composite_files"] = None
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_hbao"] = False

    viewer = MSPSuperquadricAnnotatorV2(
        sim_settings=sim_settings,
        questions_csv=args.questions_csv,
        poses_csv=args.poses_csv,
        hm3d_root=args.hm3d_root,
        split=args.split,
        param_file=args.param_file,
        start_index=args.start_index,
    )

    viewer.exec()


if __name__ == "__main__":
    main()