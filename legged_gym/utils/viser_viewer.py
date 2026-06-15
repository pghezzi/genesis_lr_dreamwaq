"""Web-based robot visualization using viser.

Parses URDF files, runs forward kinematics, and renders the robot
in a browser-based 3D viewer synchronized with the physics simulator.
"""

from __future__ import annotations

import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material

from legged_gym import LEGGED_GYM_ROOT_DIR

try:
    import viser
    import viser.transforms as vtf
    HAS_VISER = True
except ImportError:
    HAS_VISER = False


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from Euler angles (extrinsic XYZ = intrinsic ZYX)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ],
    ])


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues formula)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    if abs(angle) < 1e-12:
        return np.eye(3)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from xyzw to wxyz."""
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from wxyz to xyzw."""
    return np.array([q[1], q[2], q[3], q[0]])


@dataclass
class UrdfJoint:
    """Parsed URDF joint."""
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    lower: float = -math.pi
    upper: float = math.pi
    dof_index: int = -1


@dataclass
class UrdfLinkVisual:
    """Visual geometry for a link."""
    mesh_path: Optional[str] = None
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0]))
    primitive_type: Optional[str] = None
    primitive_size: Optional[np.ndarray] = None


@dataclass
class UrdfLink:
    """Parsed URDF link."""
    name: str
    visuals: List[UrdfLinkVisual] = field(default_factory=list)


class UrdfKinematicModel:
    """Parses a URDF file and provides forward kinematics.

    Handles revolute, prismatic, and fixed joints. Meshes are loaded
    via trimesh for visualization.
    """

    def __init__(self, urdf_path: str, dof_names: Optional[List[str]] = None):
        """
        Args:
            urdf_path: Path to the URDF file.
            dof_names: Ordered list of joint names that correspond to the
                simulator's DOF vector. If None, all revolute joints are
                used in URDF declaration order.
        """
        self.urdf_path = urdf_path
        self.urdf_dir = os.path.dirname(urdf_path)

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        self.links: Dict[str, UrdfLink] = {}
        self.joints: Dict[str, UrdfJoint] = {}
        self.joint_order: List[str] = []

        self._parse_links(root)
        self._parse_joints(root)
        self._assign_dof_indices(dof_names)

        self.children: Dict[str, List[str]] = {}
        self.parent: Dict[str, str] = {}
        for jnt in self.joints.values():
            self.parent[jnt.child_link] = jnt.name
            self.children.setdefault(jnt.parent_link, []).append(jnt.child_link)

        self._topo_order = self._build_topo_order()

        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}

    def _parse_links(self, root: ET.Element) -> None:
        for link_elem in root.findall('link'):
            name = link_elem.get('name')
            link = UrdfLink(name=name)

            for visual_elem in link_elem.findall('visual'):
                vis = UrdfLinkVisual()

                origin = visual_elem.find('origin')
                if origin is not None:
                    vis.origin_xyz = self._parse_xyz(origin.get('xyz', '0 0 0'))
                    vis.origin_rpy = self._parse_rpy(origin.get('rpy', '0 0 0'))

                geom = visual_elem.find('geometry')
                if geom is not None:
                    mesh = geom.find('mesh')
                    if mesh is not None:
                        vis.mesh_path = mesh.get('filename')
                    else:
                        box = geom.find('box')
                        cyl = geom.find('cylinder')
                        sph = geom.find('sphere')
                        if box is not None:
                            vis.primitive_type = 'box'
                            vis.primitive_size = self._parse_xyz(box.get('size', '1 1 1'))
                        elif cyl is not None:
                            vis.primitive_type = 'cylinder'
                            vis.primitive_size = np.array([
                                float(cyl.get('radius', '0.5')),
                                float(cyl.get('length', '1.0')),
                            ])
                        elif sph is not None:
                            vis.primitive_type = 'sphere'
                            vis.primitive_size = np.array([float(sph.get('radius', '0.5'))])

                mat = visual_elem.find('material')
                if mat is not None:
                    color_elem = mat.find('color')
                    if color_elem is not None:
                        rgba_str = color_elem.get('rgba', '0.5 0.5 0.5 1.0')
                        vis.rgba = np.array([float(x) for x in rgba_str.split()])

                link.visuals.append(vis)

            self.links[name] = link

    def _parse_joints(self, root: ET.Element) -> None:
        for joint_elem in root.findall('joint'):
            name = joint_elem.get('name')
            jtype = joint_elem.get('type')

            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            if parent_elem is None or child_elem is None:
                continue

            jnt = UrdfJoint(
                name=name,
                joint_type=jtype,
                parent_link=parent_elem.get('link'),
                child_link=child_elem.get('link'),
            )

            origin = joint_elem.find('origin')
            if origin is not None:
                jnt.origin_xyz = self._parse_xyz(origin.get('xyz', '0 0 0'))
                jnt.origin_rpy = self._parse_rpy(origin.get('rpy', '0 0 0'))

            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                jnt.axis = self._parse_xyz(axis_elem.get('xyz', '0 0 1'))

            limit = joint_elem.find('limit')
            if limit is not None:
                jnt.lower = float(limit.get('lower', '-3.14159'))
                jnt.upper = float(limit.get('upper', '3.14159'))

            self.joints[name] = jnt

    def _assign_dof_indices(self, dof_names: Optional[List[str]]) -> None:
        """Assign DOF indices to revolute/prismatic joints."""
        if dof_names is not None:
            for i, name in enumerate(dof_names):
                if name in self.joints and self.joints[name].joint_type in ('revolute', 'prismatic'):
                    self.joints[name].dof_index = i
                    self.joint_order.append(name)
        else:
            idx = 0
            for name, jnt in self.joints.items():
                if jnt.joint_type in ('revolute', 'prismatic'):
                    jnt.dof_index = idx
                    self.joint_order.append(name)
                    idx += 1

    def _build_topo_order(self) -> List[str]:
        """BFS from root link to get topological order."""
        child_links = {j.child_link for j in self.joints.values()}
        root_links = [l for l in self.links if l not in child_links]
        if not root_links:
            root_links = [list(self.links.keys())[0]]

        order = []
        queue = list(root_links)
        visited = set(queue)
        while queue:
            link = queue.pop(0)
            order.append(link)
            for child_link in self.children.get(link, []):
                if child_link not in visited:
                    visited.add(child_link)
                    queue.append(child_link)
        return order

    def _parse_xyz(self, s: str) -> np.ndarray:
        if s is None:
            return np.zeros(3)
        return np.array([float(x) for x in s.split()])

    def _parse_rpy(self, s: str) -> np.ndarray:
        if s is None:
            return np.zeros(3)
        return np.array([float(x) for x in s.split()])

    def get_joint_for_link(self, link_name: str) -> Optional[UrdfJoint]:
        """Get the joint that connects parent to this link."""
        jnt_name = self.parent.get(link_name)
        if jnt_name:
            return self.joints[jnt_name]
        return None

    def forward_kinematics(
        self,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute world-frame transforms for all links.

        Args:
            base_pos: Base position (3,).
            base_quat_wxyz: Base orientation as wxyz quaternion (4,).
            dof_pos: Joint positions vector, ordered by dof_names.

        Returns:
            Dict mapping link_name → (pos(3,), quat_wxyz(4,)).
        """
        world_pos: Dict[str, np.ndarray] = {}
        world_rot: Dict[str, np.ndarray] = {}

        # Root transform
        root = self._topo_order[0] if self._topo_order else None
        if root is None:
            return {}

        world_pos[root] = base_pos.copy()
        world_rot[root] = vtf.SO3(wxyz=base_quat_wxyz).as_matrix()

        for link_name in self._topo_order[1:]:
            jnt = self.get_joint_for_link(link_name)
            if jnt is None:
                parent_name = None
                for p, children in self.children.items():
                    if link_name in children:
                        parent_name = p
                        break
                if parent_name and parent_name in world_pos:
                    world_pos[link_name] = world_pos[parent_name].copy()
                    world_rot[link_name] = world_rot[parent_name].copy()
                continue

            parent_name = jnt.parent_link
            if parent_name not in world_pos:
                continue

            T_joint = np.eye(4)
            T_joint[:3, :3] = _rpy_to_matrix(*jnt.origin_rpy)
            T_joint[:3, 3] = jnt.origin_xyz

            if jnt.joint_type == 'revolute' and jnt.dof_index >= 0:
                angle = float(dof_pos[jnt.dof_index])
                R_joint = _axis_angle_to_matrix(jnt.axis, angle)
                T_joint[:3, :3] = T_joint[:3, :3] @ R_joint
            elif jnt.joint_type == 'prismatic' and jnt.dof_index >= 0:
                displacement = float(dof_pos[jnt.dof_index])
                T_joint[:3, 3] += jnt.axis * displacement

            T_parent = np.eye(4)
            T_parent[:3, :3] = world_rot[parent_name]
            T_parent[:3, 3] = world_pos[parent_name]

            T_world = T_parent @ T_joint

            world_pos[link_name] = T_world[:3, 3]
            world_rot[link_name] = T_world[:3, :3]

        return {name: (world_pos[name], world_rot[name])
                for name in world_pos}

    def load_link_meshes(self) -> Dict[str, trimesh.Trimesh]:
        """Load and cache all link visual meshes.

        Returns:
            Dict mapping link_name → merged trimesh for all visuals.
        """
        result = {}
        for link_name, link in self.links.items():
            meshes = []
            for vis in link.visuals:
                mesh = self._load_visual_mesh(vis)
                if mesh is not None:
                    T = np.eye(4)
                    T[:3, :3] = _rpy_to_matrix(*vis.origin_rpy)
                    T[:3, 3] = vis.origin_xyz
                    mesh.apply_transform(T)
                    meshes.append(mesh)

            if meshes:
                merged = trimesh.util.concatenate(meshes)
                result[link_name] = merged
        return result

    def _load_visual_mesh(self, vis: UrdfLinkVisual) -> Optional[trimesh.Trimesh]:
        """Load a single visual mesh or create a primitive."""
        if vis.mesh_path is not None:
            return self._load_stl(vis.mesh_path)
        elif vis.primitive_type is not None:
            return self._create_primitive(vis)
        return None

    def _load_stl(self, rel_path: str) -> Optional[trimesh.Trimesh]:
        """Load an STL mesh file relative to the URDF directory."""
        if rel_path in self._mesh_cache:
            return self._mesh_cache[rel_path]

        full_path = os.path.join(self.urdf_dir, rel_path)
        if not os.path.exists(full_path):
            return None

        try:
            mesh = trimesh.load(full_path, force='mesh')
            self._mesh_cache[rel_path] = mesh
            return mesh
        except Exception:
            return None

    def _create_primitive(self, vis: UrdfLinkVisual) -> Optional[trimesh.Trimesh]:
        """Create a trimesh primitive from URDF visual spec."""
        size = vis.primitive_size
        if size is None:
            return None

        if vis.primitive_type == 'box':
            mesh = trimesh.creation.box(extents=size)
        elif vis.primitive_type == 'cylinder':
            radius, length = size[0], size[1]
            mesh = trimesh.creation.cylinder(radius=radius, height=length)
        elif vis.primitive_type == 'sphere':
            mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
        else:
            return None

        rgba = (np.clip(vis.rgba, 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba, (len(mesh.vertices), 1))
        )
        return mesh


class ViserViewer:
    """Manages a viser server for robot visualization.

    Usage::

        viewer = ViserViewer(urdf_path, dof_names, num_envs=1)
        viewer.set_terrain_mesh(terrain_trimesh)

        # In the interaction loop:
        viewer.update(base_pos_np, base_quat_wxyz_np, dof_pos_np)
    """

    def __init__(
        self,
        urdf_path: str,
        dof_names: Optional[List[str]] = None,
        num_envs: int = 1,
        server: Optional[object] = None,
        port: int = 8080,
    ):
        if not HAS_VISER:
            raise ImportError(
                "viser is required for web visualization. "
                "Install with: pip install viser"
            )

        self.num_envs = num_envs
        self.urdf_path = urdf_path

        self.kin_model = UrdfKinematicModel(urdf_path, dof_names)

        self.link_meshes = self.kin_model.load_link_meshes()

        if server is not None:
            self.server = server
        else:
            self.server = viser.ViserServer(port=port)

        self._body_handles: Dict[str, object] = {}
        self._env_frames: List[object] = []

        self._build_scene()

    def _build_scene(self) -> None:
        """Create viser scene nodes for the robot and terrain."""
        for env_idx in range(self.num_envs):
            prefix = f"/env_{env_idx}" if self.num_envs > 1 else ""
            frame = self.server.scene.add_frame(
                f"{prefix}/robot",
                show_axes=False,
            )
            self._env_frames.append(frame)

            for link_name, mesh in self.link_meshes.items():
                path = f"{prefix}/robot/{link_name}"
                handle = self.server.scene.add_mesh_trimesh(
                    path,
                    mesh,
                    cast_shadow=True,
                    receive_shadow=True,
                )
                self._body_handles[(env_idx, link_name)] = handle

        self.server.scene.add_grid(
            "/ground",
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=0.2,
            plane_opacity=0.4,
        )

        self._setup_camera()

    def _setup_camera(self) -> None:
        """Configure initial camera position."""
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = np.array([2.0, 2.0, 1.5])
            client.camera.look_at = np.array([0.0, 0.0, 0.3])
            client.camera.fov = np.radians(60.0)

    def set_terrain_mesh(self, terrain_mesh: Optional[trimesh.Trimesh]) -> None:
        """Add a terrain mesh to the scene.

        Args:
            terrain_mesh: Trimesh object representing the terrain, or None
                to remove terrain.
        """
        if terrain_mesh is None:
            return

        if hasattr(self, '_terrain_handle') and self._terrain_handle is not None:
            self._terrain_handle.remove()

        self._terrain_handle = self.server.scene.add_mesh_trimesh(
            "/terrain",
            terrain_mesh,
            cast_shadow=True,
            receive_shadow=True,
        )

    def set_terrain_from_heightfield(
        self,
        height_samples: np.ndarray,
        horizontal_scale: float = 0.1,
        vertical_scale: float = 0.005,
    ) -> None:
        """Create terrain mesh from a heightfield array.

        Args:
            height_samples: 2D array of height values (rows, cols).
            horizontal_scale: Distance between samples in meters.
            vertical_scale: Height scale factor.
        """
        hfield = height_samples.astype(np.float64) * vertical_scale
        nrow, ncol = hfield.shape

        x = np.arange(ncol) * horizontal_scale
        y = np.arange(nrow) * horizontal_scale
        xx, yy = np.meshgrid(x, y)
        zz = hfield

        vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        ri, ci = np.mgrid[:nrow-1, :ncol-1]
        i0 = (ri * ncol + ci).ravel()
        faces = np.column_stack([
            i0, i0 + 1, i0 + ncol + 1,
            i0, i0 + ncol + 1, i0 + ncol,
        ]).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        heights = vertices[:, 2]
        h_min, h_max = heights.min(), heights.max()
        if h_max - h_min > 1e-6:
            normalized = (heights - h_min) / (h_max - h_min)
        else:
            normalized = np.zeros_like(heights)
        colors = np.zeros((len(heights), 4), dtype=np.uint8)
        colors[:, 0] = (50 + 100 * normalized).astype(np.uint8)
        colors[:, 1] = (150 + 100 * (1 - normalized)).astype(np.uint8)
        colors[:, 2] = (100 + 50 * normalized).astype(np.uint8)
        colors[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)

        self.set_terrain_mesh(mesh)

    def update(
        self,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
        env_idx: int = 0,
    ) -> None:
        """Update robot visualization from simulator state.

        Args:
            base_pos: Base position in world frame (3,).
            base_quat_wxyz: Base orientation as wxyz quaternion (4,).
            dof_pos: Joint positions vector matching dof_names order.
            env_idx: Environment index to update.
        """
        fk_results = self.kin_model.forward_kinematics(
            base_pos, base_quat_wxyz, dof_pos
        )

        with self.server.atomic():
            for link_name, (pos, rot) in fk_results.items():
                handle = self._body_handles.get((env_idx, link_name))
                if handle is not None:
                    quat_wxyz = vtf.SO3.from_matrix(rot).wxyz
                    handle.position = pos
                    handle.wxyz = quat_wxyz

        self.server.flush()

    def update_batch(
        self,
        base_positions: np.ndarray,
        base_quats_wxyz: np.ndarray,
        dof_positions: np.ndarray,
    ) -> None:
        """Update all environments at once.

        Args:
            base_positions: (num_envs, 3) base positions.
            base_quats_wxyz: (num_envs, 4) base quaternions (wxyz).
            dof_positions: (num_envs, num_dof) joint positions.
        """
        num = min(base_positions.shape[0], self.num_envs)
        for env_idx in range(num):
            self.update(
                base_positions[env_idx],
                base_quats_wxyz[env_idx],
                dof_positions[env_idx],
                env_idx,
            )

    def update_from_simulator(self, env, robot_index: int = 0) -> None:
        """Update visualization directly from a LeggedRobot environment.

        Reads base_pos, base_quat, and dof_pos from the simulator
        and pushes to the viser scene.

        Args:
            env: LeggedRobot environment instance.
            robot_index: Which robot to visualize (default: 0).
        """
        base_pos = env.simulator.base_pos[robot_index].cpu().numpy()
        base_quat_xyzw = env.simulator.base_quat[robot_index].cpu().numpy()
        base_quat_wxyz = _xyzw_to_wxyz(base_quat_xyzw)
        dof_pos = env.simulator.dof_pos[robot_index].cpu().numpy()

        self.update(base_pos, base_quat_wxyz, dof_pos, env_idx=0)

    def create_gui(self) -> None:
        """Add visualization controls to the viser GUI."""
        with self.server.gui.add_folder("Viewer"):
            pause_cb = self.server.gui.add_checkbox(
                "Pause",
                initial_value=False,
            )
            speed_slider = self.server.gui.add_slider(
                "Speed",
                min=0.1,
                max=3.0,
                step=0.1,
                initial_value=1.0,
            )
            track_cb = self.server.gui.add_checkbox(
                "Track camera",
                initial_value=True,
            )

        return {
            'pause': pause_cb,
            'speed': speed_slider,
            'track_camera': track_cb,
        }

    def stop(self) -> None:
        """Stop the viser server."""
        if hasattr(self, 'server'):
            self.server.stop()


def create_viser_viewer(
    env,
    port: int = 8080,
    robot_index: int = 0,
) -> ViserViewer:
    """Create a ViserViewer from a running environment.

    Args:
        env: LeggedRobot environment instance.
        port: Port for the viser web server.
        robot_index: Which robot to visualize.

    Returns:
        Configured ViserViewer instance.
    """
    urdf_path = env.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    dof_names = env.cfg.asset.dof_names

    viewer = ViserViewer(
        urdf_path=urdf_path,
        dof_names=dof_names,
        num_envs=1,
        port=port,
    )

    _attach_terrain(viewer, env)

    return viewer


def _attach_terrain(viewer: ViserViewer, env) -> None:
    """Attempt to load terrain from the environment into the viewer."""
    try:
        terrain = getattr(env.simulator, '_terrain', None)
        if terrain is None:
            return

        mesh_type = env.cfg.terrain.mesh_type

        if mesh_type == 'plane':
            pass
        elif mesh_type == 'trimesh' and hasattr(terrain, 'terrain_mesh'):
            viewer.set_terrain_mesh(terrain.terrain_mesh)
        elif mesh_type == 'heightfield' and hasattr(terrain, 'heightsamples'):
            viewer.set_terrain_from_heightfield(
                terrain.heightsamples,
                horizontal_scale=env.cfg.terrain.horizontal_scale,
                vertical_scale=env.cfg.terrain.vertical_scale,
            )
    except Exception as e:
        print(f"[viser_viewer] Warning: Could not load terrain: {e}")
