#!/usr/bin/env python3
"""
viewer_hover_v5.py

Adds hover-to-label HUD using the semantic sensor.

Goal:
- read semantic id under mouse cursor
- map it to a label like: chair / table / wall
- show it on screen continuously

Mapping strategy (robust):
1) Try semantic_scene.objects where obj.id == sid (HM3D instance id)
2) Fallback: treat sid as an index into semantic_scene.objects (some datasets do this)
3) Fallback: treat sid as a category index into semantic_scene.categories (rare but possible)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import math
from habitat_sim.agent import agent
import quaternion  # pip package "numpy-quaternion"
from habitat_sim.utils.common import quat_rotate_vector
import magnum as mn
import numpy as np
from magnum.platform.glfw import Application

from habitat_sim.logging import logger
from viewer import HabitatSimInteractiveViewer


class HabitatSimInteractiveViewerHoverV5(HabitatSimInteractiveViewer):
    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        sim_settings = dict(sim_settings)

        # Ensure semantic sensor exists
        # Habitat-sim viewer uses sim_settings -> make_cfg() which enables sensors.
        # Setting this flag is harmless even if your config already enables it.
        sim_settings["semantic_sensor"] = True

        super().__init__(sim_settings)

        self.hover_enabled: bool = True
        self._hover_label: str = "Hover: (move mouse)"
        self._hover_sid: int = -1

        self._sem_obj_by_id: Dict[int, Any] = {}
        self._build_semantic_object_map()

        logger.info("[hover_v5] Hover labels enabled (HUD).")

    def get_current_hover_sid(self) -> int:
        """
        Small helper for other tools (like the MSP annotator) to query
        the last semantic id under the mouse.
        """
        return int(getattr(self, "_hover_sid", -1))
    
    # -----------------------------
    # Rebuild map on scene reload
    # -----------------------------
    def reconfigure_sim(self) -> None:
        super().reconfigure_sim()
        self._build_semantic_object_map()

    def _build_semantic_object_map(self) -> None:
        self._sem_obj_by_id = {}
        if self.sim is None:
            return

        scene = getattr(self.sim, "semantic_scene", None)
        objects = getattr(scene, "objects", None) if scene is not None else None
        if not objects:
            logger.warning("[hover_v5] semantic_scene.objects empty (no semantic annotations loaded).")
            return

        for obj in objects:
            oid = getattr(obj, "id", None)
            if oid is None:
                continue
            try:
                self._sem_obj_by_id[int(oid)] = obj
            except Exception:
                pass

        logger.info(f"[hover_v5] Built semantic id→object map: {len(self._sem_obj_by_id)} objects")

    # -----------------------------
    # Public helpers for hover info
    # -----------------------------
    def get_current_hover_sid(self) -> int:
        """
        Return the last semantic id under the mouse cursor (or -1 if none).
        """
        return int(self._hover_sid) if self._hover_sid is not None else -1

    def get_current_hover_label(self) -> str:
        """
        Return the last human-readable hover label string.
        """
        return str(self._hover_label)

    # -----------------------------
    # Read semantic frame
    # -----------------------------
    def _get_semantic_frame(self) -> Optional[np.ndarray]:
        if self.sim is None:
            return None
        obs = self.sim.get_sensor_observations()
        if not isinstance(obs, dict):
            return None

        # common keys
        if "semantic_sensor" in obs:
            return obs["semantic_sensor"]
        if "semantic" in obs:
            return obs["semantic"]
        if "sem" in obs:
            return obs["sem"]
        return None

    def _semantic_at_pixel(self, px: mn.Vector2i) -> Optional[int]:
        sem = self._get_semantic_frame()
        if sem is None:
            return None

        h, w = sem.shape[0], sem.shape[1]
        x, y = int(px.x), int(px.y)
        if x < 0 or y < 0 or x >= w or y >= h:
            return None

        v = sem[y, x]
        if isinstance(v, np.ndarray):
            v = v.item() if v.size == 1 else v.flat[0]
        try:
            return int(v)
        except Exception:
            return None

    # -----------------------------
    # Convert semantic id → label
    # -----------------------------
    def _cat_name(self, cat: Any) -> str:
        if cat is None:
            return ""
        n = getattr(cat, "name", None)
        if callable(n):
            try:
                return str(n())
            except Exception:
                return ""
        if isinstance(n, str):
            return n
        return str(cat)

    def _sid_to_label(self, sid: int) -> str:
        if sid is None or sid <= 0:
            return "None"

        # Strategy 1: HM3D instance id stored in semantic buffer
        obj = self._sem_obj_by_id.get(int(sid), None)
        if obj is not None:
            cat = getattr(obj, "category", None)
            cat_name = self._cat_name(cat) or "object"
            return f"{cat_name}  (sid={sid})"

        scene = getattr(self.sim, "semantic_scene", None)
        objects = getattr(scene, "objects", None) if scene is not None else None

        # Strategy 2: sid is an index into objects list
        if objects and 0 <= sid < len(objects):
            obj2 = objects[sid]
            cat = getattr(obj2, "category", None)
            cat_name = self._cat_name(cat) or "object"
            oid = getattr(obj2, "id", sid)
            return f"{cat_name}  (obj.id={oid}, sid={sid})"

        # Strategy 3: sid is a category index (rare, but handle)
        cats = getattr(scene, "categories", None) if scene is not None else None
        if cats and 0 <= sid < len(cats):
            return f"{self._cat_name(cats[sid])}  (category_id={sid})"

        return f"(unmapped sid={sid})"

    # -----------------------------
    # HUD text: add hover line
    # -----------------------------
    def _agent_world_pos(self):
        try:
            node = self.default_agent.scene_node
            p = node.absolute_translation
            x, y, z = float(p.x), float(p.y), float(p.z)
        except Exception:
            return None

        try:
            # Get agent state rotation (preferred, stable)
            agent = self.sim.get_agent(0)
            st = agent.get_state()  # AgentState.rotation exists
            r = st.rotation

            # Normalize to a numpy-quaternion (w, x, y, z)
            if isinstance(r, quaternion.quaternion):
                q = r
            else:
                # Habitat-Sim uses [x, y, z, w] for list/np arrays
                rx, ry, rz, rw = [float(v) for v in r]
                q = quaternion.quaternion(rw, rx, ry, rz)

            # Rotate the local forward vector (-Z) into world frame
            fwd = quat_rotate_vector(q, np.array([0.0, 0.0, -1.0], dtype=np.float32))

            # Yaw around +Y (Habitat is typically Y-up)
            yaw_rad = math.atan2(float(fwd[0]), -float(fwd[2]))

        except Exception as e:
            logger.exception(f"[hover_v5] yaw failed: {e}")
            yaw_rad = None

        return x, y, z, yaw_rad

    def draw_text(self, sensor_spec) -> None:
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )

        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]

        # Get our 4 values
        pos_data = self._agent_world_pos()

        if pos_data:
            x, y, z, yaw = pos_data
            if yaw is None:
                pos_str = f"Agent XYZ: ({x:.2f}, {y:.2f}, {z:.2f}) | Yaw: n/a"
            else:
                # Convert to your CSV/init_angle convention
                yaw_init_convention = -yaw          # <-- your previous conversion
                yaw_deg = math.degrees(yaw_init_convention)

                pos_str = (
                    f"Agent XYZ: ({x:.2f}, {y:.2f}, {z:.2f}) | "
                    f"Yaw(init): {yaw_init_convention:.3f} rad ({yaw_deg:+.1f}°)"
                )
        else:
            pos_str = "Agent XYZ: (n/a)"

        self.window_text.render(
            f"{self.fps} FPS\n"
            f"Mouse Interaction: {'LOOK' if self.mouse_interaction.value == 0 else 'GRAB'}\n"
            f"{pos_str}\n"
            f"{self._hover_label}"
        )
        self.shader.draw(self.window_text.mesh)
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)

    # -----------------------------
    # Update hover on mouse move
    # -----------------------------
    def pointer_move_event(self, event: Application.PointerMoveEvent) -> None:
        super().pointer_move_event(event)

        if not self.hover_enabled:
            return

        px = self.get_mouse_position(event.position)
        sid = self._semantic_at_pixel(px)

        if sid is None:
            self._hover_label = "Hover: (no semantic frame)"
            self._hover_sid = -1
            return

        if sid != self._hover_sid:
            self._hover_sid = sid
            self._hover_label = f"Hover: {self._sid_to_label(sid)}"