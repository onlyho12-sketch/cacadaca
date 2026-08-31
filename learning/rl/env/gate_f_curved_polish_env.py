"""Isolated Gate F environment for curved Workpiece and local-normal force.

This subclass keeps the established planar/factory environments untouched.  F1
ports the geometry, trimesh, path-height, and ``abs(F dot n)`` wiring only; the
first Isaac/PhysX execution belongs to Gate F4.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from learning.rl.gate_f_curved_geometry import (
    SurfaceGeometrySpec,
    biased_normal_command,
    build_surface_mesh,
    collected_link6_target_quaternion_xyzw,
    surface_height_normal,
    vertical_pad_target_z_m,
)
from scripts.polishing_v5_modules.common import (
    POLISHING_DISK_HEIGHT,
    create_polishing_contact_disk_for_robot,
)

from .factory_planar_roi_polish_env import FactoryPlanarRoiPolishEnv
from .gate_f_curved_polish_env_cfg import GateFCurvedPolishEnvCfg


class GateFCurvedPolishEnv(FactoryPlanarRoiPolishEnv):
    """Factory-profile environment with an isolated macro-curved Workpiece."""

    cfg: GateFCurvedPolishEnvCfg

    def __init__(self, cfg: GateFCurvedPolishEnvCfg,
                 render_mode: str | None = None, **kwargs):
        self._geometry_spec = SurfaceGeometrySpec(
            kind=str(cfg.surface_kind),
            patch_size_m=tuple(cfg.patch_size_m),
            curvature_radius_m=float(cfg.curvature_radius_m),
            freeform_seed=int(cfg.freeform_seed),
        )
        self._geometry_spec.validate()
        super().__init__(cfg, render_mode, **kwargs)

    def _height_normal(self, u_m: float, v_m: float) -> tuple[float, np.ndarray]:
        return surface_height_normal(
            self._geometry_spec.kind,
            self._geometry_spec.curvature_radius_m,
            self._geometry_spec.patch_size_m,
            u_m,
            v_m,
            freeform_seed=self._geometry_spec.freeform_seed,
        )

    def _setup_scene(self):
        """Reproduce the protected robot setup but replace only the Workpiece."""
        self.robot = Articulation(self.cfg.robot_cfg)

        import omni.usd
        from pxr import PhysxSchema, UsdGeom

        stage = omni.usd.get_context().get_stage()
        robot_container_path = "/World/envs/env_0/Robot"
        sander_candidates = [
            str(prim.GetPath()) for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(robot_container_path)
            and prim.GetName() == "sander_pad"
        ]
        if len(sander_candidates) != 1:
            descendants = [
                str(prim.GetPath()) for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(robot_container_path)
                and prim.GetName() in {"m0609", "link_6", "sander_pad"}
            ]
            raise RuntimeError(
                f"collected M0609 sander_pad resolution failed: {sander_candidates}; "
                f"matching descendants={descendants}"
            )
        robot_path = sander_candidates[0].rsplit("/", 1)[0]
        self._robot_asset_root_suffix = robot_path.removeprefix("/World/envs/env_0")
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(robot_container_path) and prim.GetName() in {
                "tn__114555_", "tn__104327_"
            }:
                UsdGeom.Imageable(prim).MakeInvisible()
        pad_path = create_polishing_contact_disk_for_robot(
            stage, robot_path, robot_path + "/sander_pad", None
        )
        pad_prim = stage.GetPrimAtPath(pad_path)
        PhysxSchema.PhysxContactReportAPI.Apply(pad_prim).CreateThresholdAttr().Set(0.0)

        if self.cfg.enable_pad_physical_contact:
            from pxr import UsdPhysics
            from isaaclab.sim.spawners.materials.physics_materials import (
                spawn_rigid_body_material,
            )
            from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import (
                PhysxRigidBodyMaterialCfg,
            )
            from scripts.polishing_v5_modules.common import set_collision_enabled_recursive

            set_collision_enabled_recursive(stage, robot_container_path, False)
            UsdPhysics.CollisionAPI.Apply(pad_prim).CreateCollisionEnabledAttr().Set(True)
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(pad_prim)
            physx_col.CreateContactOffsetAttr().Set(float(self.cfg.pad_contact_offset_m))
            physx_col.CreateRestOffsetAttr().Set(float(self.cfg.pad_rest_offset_m))
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(pad_prim)
            physx_rb.CreateMaxDepenetrationVelocityAttr().Set(
                float(self.cfg.pad_max_depenetration_vel_m_s)
            )
            material_path = "/World/PhysicsMaterials/gate_f_pad_compliant"
            spawn_rigid_body_material(material_path, PhysxRigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
                compliant_contact_stiffness=float(self.cfg.pad_compliant_stiffness_n_m),
                compliant_contact_damping=float(self.cfg.pad_compliant_damping_n_s_m),
            ))
            sim_utils.bind_physics_material(pad_path, material_path)

        pedestal = sim_utils.CuboidCfg(
            size=(self.cfg.patch_size_m[0] + 0.06, self.cfg.patch_size_m[1] + 0.06,
                  self.cfg.work_top_m - 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        )
        pedestal.func(
            "/World/envs/env_0/Pedestal", pedestal,
            translation=(self.cfg.patch_center_xy_m[0], self.cfg.patch_center_xy_m[1],
                         (self.cfg.work_top_m - 0.05) / 2),
        )

        if self._geometry_spec.kind == "flat":
            self._spawn_flat_workpiece()
        else:
            self._spawn_curved_workpiece()

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        sensor_cfg = ContactSensorCfg(
            prim_path=("/World/envs/env_.*/" + self._robot_asset_root_suffix.lstrip("/")
                       + "/polishing_contact_pad"),
            update_period=0.0,
            history_length=1,
            track_pose=True,
        )
        if self.cfg.enable_pad_physical_contact:
            sensor_cfg.filter_prim_paths_expr = ["/World/envs/env_.*/Workpiece"]
        self.pad_force_sensor = ContactSensor(sensor_cfg)
        self.scene.sensors["pad_force"] = self.pad_force_sensor

        light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.87, 0.9))
        light.func("/World/Light", light)

    def _spawn_flat_workpiece(self) -> None:
        plate = sim_utils.CuboidCfg(
            size=(self.cfg.patch_size_m[0] + 0.04, self.cfg.patch_size_m[1] + 0.04, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.12, 0.35), roughness=0.15, metallic=0.6
            ),
        )
        if self.cfg.enable_pad_physical_contact:
            plate.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            plate.mass_props = sim_utils.MassPropertiesCfg(mass=10.0)
        plate.func(
            "/World/envs/env_0/Workpiece", plate,
            translation=(self.cfg.patch_center_xy_m[0], self.cfg.patch_center_xy_m[1],
                         self.cfg.work_top_m - 0.025),
        )

    def _spawn_curved_workpiece(self) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        mesh_data = build_surface_mesh(
            self._geometry_spec,
            grid_size=int(self.cfg.curved_mesh_grid_size),
            margin_m=float(self.cfg.curved_mesh_margin_m),
        )
        cx, cy = self.cfg.patch_center_xy_m
        offset = np.asarray((cx, cy, self.cfg.work_top_m), dtype=np.float64)
        points = [Gf.Vec3f(*(vertex + offset)) for vertex in mesh_data.vertices_m]
        indices = mesh_data.triangles.reshape(-1).tolist()

        stage = omni.usd.get_context().get_stage()
        mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Workpiece")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * len(mesh_data.triangles))
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.12, 0.35)])
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr("none")
        if self.cfg.enable_pad_physical_contact:
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(10.0)
        self._gate_f_mesh_vertex_count = int(len(mesh_data.vertices_m))
        self._gate_f_mesh_triangle_count = int(len(mesh_data.triangles))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, device=self.device).long()
        for env_id in ids.cpu().tolist():
            state = self._surfaces[env_id]
            u = state.nominal_surface_xyz_m[..., 0]
            v = state.nominal_surface_xyz_m[..., 1]
            height, normal = surface_height_normal(
                self._geometry_spec.kind,
                self._geometry_spec.curvature_radius_m,
                self._geometry_spec.patch_size_m,
                u,
                v,
                freeform_seed=self._geometry_spec.freeform_seed,
            )
            state.nominal_surface_xyz_m[..., 2] = height
            state.normal_xyz[...] = normal

    def _update_measured_pad_state(self):
        """Measure curved gap and project ContactSensor force onto local normal."""
        pad = self.robot.data.body_pose_w.torch[:, self._pad_body_id]
        face_local = torch.zeros((self.num_envs, 3), device=self.device)
        face_local[:, 1] = -0.5 * float(POLISHING_DISK_HEIGHT)
        face_w = pad[:, :3] + quat_apply(pad[:, 3:7], face_local)
        self._pad_face_w = face_w
        local = face_w - self.scene.env_origins
        lower_x = self.cfg.patch_center_xy_m[0] - self.cfg.patch_size_m[0] / 2
        lower_y = self.cfg.patch_center_xy_m[1] - self.cfg.patch_size_m[1] / 2
        self._pad_uv_actual[:, 0] = local[:, 0] - lower_x
        self._pad_uv_actual[:, 1] = local[:, 1] - lower_y

        uv = self._pad_uv_actual.detach().cpu().numpy()
        heights = np.empty(self.num_envs, dtype=np.float64)
        normals = np.empty((self.num_envs, 3), dtype=np.float64)
        for i in range(self.num_envs):
            heights[i], normals[i] = self._height_normal(float(uv[i, 0]), float(uv[i, 1]))
        height_t = torch.as_tensor(heights, dtype=local.dtype, device=self.device)
        normal_t = torch.as_tensor(normals, dtype=local.dtype, device=self.device)
        self._pad_surface_normal_w = normal_t
        link6 = self.robot.data.body_pose_w.torch[:, self._ee_body_id]
        self._link6_quat_w = link6[:, 3:7].clone()
        self._pad_quat_w = pad[:, 3:7].clone()
        link6_axis_local = torch.zeros((self.num_envs, 3), device=self.device)
        link6_axis_local[:, 2] = 1.0
        self._link6_axis_w = quat_apply(link6[:, 3:7], link6_axis_local)
        # Measure the actual contact disk frame.  The contact face point is pad
        # local -Y (face_local above), so its outward workpiece-facing normal is
        # pad local +Y.  A link6-axis proxy is invalid once both tilt axes move.
        pad_outward_local = torch.zeros((self.num_envs, 3), device=self.device)
        pad_outward_local[:, 1] = 1.0
        self._pad_outward_axis_w = quat_apply(pad[:, 3:7], pad_outward_local)
        alignment_dot = torch.sum(
            self._pad_outward_axis_w * normal_t, dim=-1).clamp(-1.0, 1.0)
        self._normal_alignment_error_deg = torch.rad2deg(torch.acos(alignment_dot))
        self._pad_gap_m = local[:, 2] - (float(self.cfg.work_top_m) + height_t)

        margin = self.cfg.pad_outside_margin_m
        self._pad_in_patch = (
            (self._pad_uv_actual[:, 0] >= -margin)
            & (self._pad_uv_actual[:, 0] <= self.cfg.patch_size_m[0] + margin)
            & (self._pad_uv_actual[:, 1] >= -margin)
            & (self._pad_uv_actual[:, 1] <= self.cfg.patch_size_m[1] + margin)
        )

        try:
            net = self.pad_force_sensor.data.net_forces_w.torch
            if net.ndim == 3:
                net = net[:, 0, :]
            fault = ~torch.isfinite(net).all(dim=-1)
            finite_net = torch.where(fault.unsqueeze(-1), torch.zeros_like(net), net)
            raw = (finite_net * normal_t).sum(dim=-1).abs()
        except Exception:
            fault = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            raw = torch.zeros(self.num_envs, device=self.device)
        self._sensor_fault = fault
        self._force_sensor_n = raw
        alpha = float(self.cfg.sensor_filter_alpha)
        self._force_sensor_filt_n = alpha * raw + (1.0 - alpha) * self._force_sensor_filt_n

        try:
            matrix = self.pad_force_sensor.data.force_matrix_w
            if matrix is not None and matrix.torch is not None and matrix.torch.numel() > 0:
                pair_force = matrix.torch.reshape(self.num_envs, -1, 3).sum(dim=1)
                self._sensor_matrix_n = (pair_force * normal_t).sum(dim=-1).abs()
        except Exception:
            pass

    def _apply_action(self) -> None:
        """Preserve established force logic while adding the curved target Z."""
        self._update_measured_pad_state()
        measured_gap = torch.where(
            self._pad_in_patch, self._pad_gap_m, torch.full_like(self._pad_gap_m, 0.20)
        )
        physical = self.cfg.enable_pad_physical_contact
        if physical:
            self._force_hard_violated |= self._force_sensor_n > self.cfg.force_hard_limit_n
        sensor_feedback = None
        if physical:
            sensor_feedback = torch.where(
                self._sensor_fault, self.contact.filtered, self._force_sensor_filt_n
            )
        model_force = self.contact.step(
            self._force_cmd,
            self._is_side,
            measured_clearance=measured_gap,
            control_force_override=sensor_feedback,
        )
        if physical:
            used = torch.where(self._sensor_fault, model_force, self._force_sensor_filt_n)
            self._fallback_steps += self._sensor_fault.long()
        elif self.cfg.use_physical_force_when_valid:
            sensor_valid = self._force_sensor_n > self.cfg.sensor_valid_min_n
            used = torch.where(sensor_valid, self._force_sensor_n, model_force)
        else:
            used = model_force
        used = torch.where(self._pad_in_patch, used, torch.zeros_like(used))
        self._force_model_n = model_force
        self._force_used_n = used
        if self._repolish_mode:
            self._pass_force_max = torch.maximum(self._pass_force_max, used)
        self._force_accum += used
        self._force_sq_accum += used * used
        self._substep_n += 1

        self._arc += self._feed_cmd * self.cfg.sim.dt
        self._sim_time += self.cfg.sim.dt
        targets = torch.zeros((self.num_envs, 7), device=self.device)
        arcs = self._arc.detach().cpu().numpy()
        max_step = self.cfg.line_transition_speed_m_s * self.cfg.sim.dt
        for i in range(self.num_envs):
            uv_raw = np.asarray(self._pos_at_arc(float(arcs[i])), dtype=np.float64)
            previous = self._prev_uv[i]
            delta = uv_raw - previous
            distance = float(np.hypot(delta[0], delta[1]))
            uv = previous + delta / distance * max_step if distance > max_step else uv_raw
            self._prev_uv[i] = uv
            targets[i, 0] = (
                float(uv[0]) - self.cfg.patch_size_m[0] / 2 + self.cfg.patch_center_xy_m[0]
            )
            targets[i, 1] = (
                float(uv[1]) - self.cfg.patch_size_m[1] / 2 + self.cfg.patch_center_xy_m[1]
            )
            height, _ = self._height_normal(float(uv[0]), float(uv[1]))
            targets[i, 2] = (
                vertical_pad_target_z_m(
                    self.cfg.work_top_m,
                    height,
                    float(self.contact.command_clearance[i].clamp(-0.003, 0.08)),
                    self.cfg.vertical_tracking_compensation_m,
                )
            )

        desired_face_w = targets[:, :3] + self.scene.env_origins
        ee_now_w = self.robot.data.body_pose_w.torch[:, self._ee_body_id, :3]
        ee_now_quat_w = self.robot.data.body_pose_w.torch[:, self._ee_body_id, 3:7]
        target_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        target_quat_w[:, 0] = 1.0
        if bool(self.cfg.align_pad_to_surface_normal):
            target_normals = np.empty((self.num_envs, 3), dtype=np.float64)
            for i in range(self.num_envs):
                _, target_normals[i] = self._height_normal(
                    float(self._prev_uv[i, 0]), float(self._prev_uv[i, 1])
                )
            if self._geometry_spec.kind != "flat":
                target_normals = biased_normal_command(
                    target_normals,
                    tuple(self.cfg.normal_alignment_command_bias_xy_deg),
                )
            target_quat_w = torch.as_tensor(
                collected_link6_target_quaternion_xyzw(target_normals),
                dtype=targets.dtype,
                device=self.device,
            )
        # Couple translation to the commanded orientation.  Reusing the
        # current world-frame face offset while tilting link6 makes the face
        # center miss its target and can drive a disk edge into curved mesh.
        # The link6-local face-center offset is invariant to the commanded
        # orientation (and to rotation of the axial pad joint).
        face_offset_link6 = quat_apply_inverse(
            ee_now_quat_w, self._pad_face_w - ee_now_w
        )
        target_face_offset_w = quat_apply(target_quat_w, face_offset_link6)
        target_link6_w = desired_face_w - target_face_offset_w
        self._target_link6_quat_w = target_quat_w
        root = self.robot.data.root_pose_w.torch
        target_b_pos, target_b_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], target_link6_w, target_quat_w
        )
        self._ik.set_command(torch.cat((target_b_pos, target_b_quat), dim=1))

        ee_w = self.robot.data.body_pose_w.torch[:, self._ee_body_id]
        ee_b_pos, ee_b_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], ee_w[:, :3], ee_w[:, 3:7]
        )
        jacobian = self.robot.data.body_link_jacobian_w.torch[
            :, self._ee_jacobi_idx, :, :
        ][:, :, self._arm_joint_ids]
        joint_pos = self.robot.data.joint_pos.torch[:, self._arm_joint_ids]
        desired_joint_pos = self._ik.compute(ee_b_pos, ee_b_quat, jacobian, joint_pos)
        self.robot.set_joint_position_target_index(
            target=desired_joint_pos, joint_ids=self._arm_joint_ids
        )
        self.robot.set_joint_position_target_index(
            target=torch.zeros((self.num_envs, 1), device=self.device),
            joint_ids=[self._pad_joint_id],
        )
