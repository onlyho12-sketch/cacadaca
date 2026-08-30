"""Robot-coupled polishing environment.

The M0609 and polishing disk are the same collected USD and runtime disk builder
used by ``polishing_v5``.  The quality/thermal model consumes the measured pad
contact position and the force computed from the measured pad-to-surface gap.
"""
from __future__ import annotations

from collections.abc import Sequence
import json

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from scripts.polishing_v5_modules.common import (
    POLISHING_DISK_HEIGHT,
    POLISHING_DISK_RADIUS,
    create_polishing_contact_disk_for_robot,
)

from .polish_env import PolishEnv
from .robot_polish_env_cfg import RobotPolishEnvCfg


class RobotPolishEnv(PolishEnv):
    cfg: RobotPolishEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Build the exact v5 runtime pad on the source environment before cloning.
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
            # ── PhysX 실접촉 활성화 (인수인계서 17.1) ─────────────────────
            # v5는 USE_PHYSICAL_CONTACT_SENSOR=False 로 패드 collider를 끈 채 생성한다
            # (common.py:880). v5 코드는 읽기 전용이므로 여기서(호출 이후) 속성만 덮어쓴다.
            from pxr import UsdPhysics
            from isaaclab.sim.spawners.materials.physics_materials import (
                spawn_rigid_body_material,
            )
            from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import (
                PhysxRigidBodyMaterialCfg,
            )
            from scripts.polishing_v5_modules.common import set_collision_enabled_recursive

            # v5 PAD_ONLY_ROBOT_COLLISIONS 원칙: 팔·샌더 본체 collider는 전부 끄고
            # 폴리싱 패드 하나만 작업면과 실충돌시킨다 (self-collision·허위 접촉 차단).
            set_collision_enabled_recursive(stage, robot_container_path, False)
            UsdPhysics.CollisionAPI.Apply(pad_prim).CreateCollisionEnabledAttr().Set(True)
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(pad_prim)
            physx_col.CreateContactOffsetAttr().Set(float(self.cfg.pad_contact_offset_m))
            physx_col.CreateRestOffsetAttr().Set(float(self.cfg.pad_rest_offset_m))
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(pad_prim)
            physx_rb.CreateMaxDepenetrationVelocityAttr().Set(
                float(self.cfg.pad_max_depenetration_vel_m_s)
            )
            # compliant contact 재질 — 마찰·반발 0은 v5 상수(POLISHING_*_FRICTION=0)와 동일.
            pad_mat_path = "/World/PhysicsMaterials/pad_compliant"
            spawn_rigid_body_material(pad_mat_path, PhysxRigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
                compliant_contact_stiffness=float(self.cfg.pad_compliant_stiffness_n_m),
                compliant_contact_damping=float(self.cfg.pad_compliant_damping_n_s_m),
            ))
            sim_utils.bind_physics_material(pad_path, pad_mat_path)

        # Workpiece dimensions and appearance match the existing robot demo.
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
        plate = sim_utils.CuboidCfg(
            size=(self.cfg.patch_size_m[0] + 0.04, self.cfg.patch_size_m[1] + 0.04, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.12, 0.35), roughness=0.15, metallic=0.6
            ),
        )
        if self.cfg.enable_pad_physical_contact:
            # GPU contact filter 는 static collider 를 지원하지 않아 (Workpiece not
            # supported 경고) 패드↔작업면 분리힘(force_matrix_w)이 0 이 된다.
            # kinematic rigid body 로 스폰하면 정지 상태 그대로 필터가 지원된다.
            plate.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            plate.mass_props = sim_utils.MassPropertiesCfg(mass=10.0)
        plate.func(
            "/World/envs/env_0/Workpiece", plate,
            translation=(self.cfg.patch_center_xy_m[0], self.cfg.patch_center_xy_m[1],
                         self.cfg.work_top_m - 0.025),
        )

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        pad_sensor_cfg = ContactSensorCfg(
            prim_path=("/World/envs/env_.*/" + self._robot_asset_root_suffix.lstrip("/")
                       + "/polishing_contact_pad"),
            update_period=0.0,
            history_length=1,
            track_pose=True,
        )
        if self.cfg.enable_pad_physical_contact:
            # 패드↔작업면 pair 만 분리 측정(force_matrix_w) — net force 에 다른 물체와의
            # 허위 접촉이 섞였는지 검증 스크립트가 대조하는 용도.
            pad_sensor_cfg.filter_prim_paths_expr = ["/World/envs/env_.*/Workpiece"]
        self.pad_force_sensor = ContactSensor(pad_sensor_cfg)
        self.scene.sensors["pad_force"] = self.pad_force_sensor

        light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.87, 0.9))
        light.func("/World/Light", light)

    def __init__(self, cfg: RobotPolishEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.enable_pad_physical_contact:
            # 물리 접촉 모드: 접촉 안정화를 위해 physics dt를 올리고(1/60→1/120),
            # 제어 주기는 dt×decimation = 1/20 s 로 유지한다. VirtualPadContact 는
            # PolishEnv 가 cfg.sim.dt 로 생성하므로 반드시 super().__init__ 전에 바꾼다.
            cfg.sim.dt = cfg.physical_sim_dt
            cfg.decimation = cfg.physical_decimation
            cfg.sim.render_interval = cfg.physical_decimation
            cfg.robot_cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=cfg.solver_position_iterations,
                solver_velocity_iteration_count=cfg.solver_velocity_iterations,
            )
        super().__init__(cfg, render_mode, **kwargs)

        # 자동차 클리어코트용 이송속도는 공용 BO recipe를 바꾸지 않고 로봇 환경에서만
        # 덮어쓴다. 경로 형상/길이는 feed와 무관하므로 recipe와 초기 feed 텐서만 바꾸면
        # 이후 _pre_physics_step 및 _reset_idx도 동일 값을 일관되게 사용한다.
        robot_feed = float(cfg.robot_feed_speed_mm_s)
        if not np.isfinite(robot_feed) or robot_feed <= 0.0:
            raise ValueError(f"robot_feed_speed_mm_s must be finite and > 0, got {robot_feed}")
        recipe_feed = float(self.recipe.feed_speed_mm_s)
        self.recipe.feed_speed_mm_s = robot_feed
        self._feed_cmd.fill_(robot_feed / 1000.0)
        print(
            f"[RobotPolishEnv] robot feed override {recipe_feed:.3f} -> "
            f"{robot_feed:.3f} mm/s (policy range "
            f"{robot_feed * (1.0 - self.cfg.feed_ratio_limit):.3f}~"
            f"{robot_feed * (1.0 + self.cfg.feed_ratio_limit):.3f} mm/s)"
        )

        if cfg.enable_pad_physical_contact:
            # 폐루프(센서 피드백) 안정화 — cfg 주석의 Phase B 실측 근거 참고.
            self.contact.admittance_damping = float(cfg.physical_admittance_damping)
            self.contact.admittance_max_vel = float(cfg.physical_admittance_max_vel_m_s)

        self._arm_joint_ids = [self.robot.joint_names.index(f"joint_{i}") for i in range(1, 7)]
        self._pad_joint_id = self.robot.joint_names.index("pad_joint")
        self._ee_body_id = self.robot.body_names.index("link_6")
        self._pad_body_id = self.robot.body_names.index("polishing_contact_pad")
        self._ee_jacobi_idx = self._ee_body_id - 1 if self.robot.is_fixed_base else self._ee_body_id
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=self.device,
        )

        # 줄바꿈 램프 리미터 상태 — _apply_action 참고. 첫 웨이포인트로 초기화해
        # 첫 스텝에 가짜 대점프가 안 생기게 한다.
        first_uv = np.asarray(self._pos_at_arc(0.0), dtype=np.float64)
        self._prev_uv = np.tile(first_uv, (self.num_envs, 1))

        self._pad_offset_link6 = self._calibrate_pad_offset_link6()
        self._pad_uv_actual = torch.zeros((self.num_envs, 2), device=self.device)
        self._pad_gap_m = torch.full((self.num_envs,), 0.20, device=self.device)
        self._pad_in_patch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._force_sensor_n = torch.zeros(self.num_envs, device=self.device)   # raw normal force
        self._force_sensor_filt_n = torch.zeros(self.num_envs, device=self.device)
        self._force_model_n = torch.zeros(self.num_envs, device=self.device)
        self._force_used_n = torch.zeros(self.num_envs, device=self.device)
        self._sensor_matrix_n = torch.zeros(self.num_envs, device=self.device)  # 패드↔작업면 분리힘(진단)
        self._sensor_fault = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fallback_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 물리 보상 항용: control step 내 substep 힘 제곱합 (std 계산) + 오류 계수기
        self._force_sq_accum = torch.zeros(self.num_envs, device=self.device)
        self._no_contact_removal_errors = 0

        # ── 재폴리싱 상태기계 (인수인계서 19장) — repolish_mode=False 면 미사용 ──
        self._repolish_mode = False
        self._pass_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._unstable_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._unstable_hard_violated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._repolish_prev_metrics: dict[int, dict] = {}
        # _reset_idx 가 지우지 않는 종료 로그 — 외부 루프가 env.step() 리턴 직후 읽는다
        # (last_episode_results 와 동일 패턴, PolishEnv._get_rewards 참고).
        self._repolish_log: dict[int, dict] = {}
        # 미달분·안전예산 기반 다음 pass 목표힘 (env별). 기본값은 BO 기준 레시피 힘 —
        # _pre_physics_step 이 self.recipe.target_contact_force_n 대신 이 텐서를 쓴다.
        self._pass_base_force = torch.full(
            (self.num_envs,), self.recipe.target_contact_force_n, device=self.device)
        # 이번 pass 동안 실측 평균힘 — "힘당 제거율" 추정에 쓴다 (_quality_update 에서 누적).
        self._pass_force_accum = torch.zeros(self.num_envs, device=self.device)
        self._pass_force_n = torch.zeros(self.num_envs, device=self.device)
        # 재폴리싱 진단 전용 누적값. 제어·품질·종료 판정에는 사용하지 않고 pass 종료
        # 시점에 CSV로 내보낼 근거만 모은다.
        self._pass_force_cmd_accum = torch.zeros(self.num_envs, device=self.device)
        self._pass_feed_accum = torch.zeros(self.num_envs, device=self.device)
        self._pass_action_accum = torch.zeros((self.num_envs, 2), device=self.device)
        self._pass_force_max = torch.zeros(self.num_envs, device=self.device)
        self._repolish_pass_history: dict[int, list[dict]] = {}

        self._update_measured_pad_state()
        print(
            "[RobotPolishEnv] v5 M0609+pad coupled | "
            f"link_6 contact offset={self._pad_offset_link6.detach().cpu().numpy().round(4)} m"
        )

    def _calibrate_pad_offset_link6(self) -> torch.Tensor:
        """Apply the same v5 contact-face calibration, returning link_6-local XYZ."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        root = "/World/envs/env_0" + self._robot_asset_root_suffix
        link = stage.GetPrimAtPath(root + "/link_6")
        pad = stage.GetPrimAtPath(root + "/polishing_contact_pad")
        if not link.IsValid() or not pad.IsValid():
            raise RuntimeError("M0609 link_6 or v5 polishing_contact_pad missing")
        cache = UsdGeom.XformCache()
        link_to_world = cache.GetLocalToWorldTransform(link)
        pad_to_world = cache.GetLocalToWorldTransform(pad)
        face = pad_to_world.Transform(Gf.Vec3d(0.0, -0.5 * float(POLISHING_DISK_HEIGHT), 0.0))
        local = link_to_world.GetInverse().Transform(face)
        return torch.tensor([float(local[0]), float(local[1]), float(local[2])], device=self.device)

    def _update_measured_pad_state(self):
        # Read the fixed v5 disk body itself.  Its local -Y face is the contact face
        # used by polishing_v5._pad_contact_world_pos(), avoiding any visual proxy.
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
        self._pad_gap_m = local[:, 2] - self.cfg.work_top_m
        margin = self.cfg.pad_outside_margin_m
        self._pad_in_patch = (
            (self._pad_uv_actual[:, 0] >= -margin)
            & (self._pad_uv_actual[:, 0] <= self.cfg.patch_size_m[0] + margin)
            & (self._pad_uv_actual[:, 1] >= -margin)
            & (self._pad_uv_actual[:, 1] <= self.cfg.patch_size_m[1] + margin)
        )

        # ── PhysX ContactSensor → normal force (인수인계서 17.1-8~10) ──────
        # F_normal = |F_contact · n_surface|. 작업면이 수평 평판이므로 n = +Z(월드).
        # 곡면 셀 도입 시 이 투영만 셀 법선으로 교체하면 된다.
        # NaN/Inf 는 0 으로 씻어내지 않고 _sensor_fault 로 분리 기록한다 — 조용한
        # fallback 성공 처리 금지 원칙 (인수인계서 17.1-12).
        try:
            net = self.pad_force_sensor.data.net_forces_w.torch
            if net.ndim == 3:
                net = net[:, 0, :]                       # (E,1,3) → (E,3): 패드 body 1개
            fault = ~torch.isfinite(net).all(dim=-1)
            net = torch.where(fault.unsqueeze(-1), torch.zeros_like(net), net)
            raw = net[:, 2].abs()
        except Exception:
            fault = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            raw = torch.zeros(self.num_envs, device=self.device)
        self._sensor_fault = fault
        self._force_sensor_n = raw
        alpha = float(self.cfg.sensor_filter_alpha)
        self._force_sensor_filt_n = alpha * raw + (1.0 - alpha) * self._force_sensor_filt_n
        # 진단: 패드↔작업면 pair 분리힘. net 과 큰 차이가 나면 다른 물체와의 허위 접촉.
        try:
            fm = self.pad_force_sensor.data.force_matrix_w
            if fm is not None:
                fm_t = fm.torch
                if fm_t is not None and fm_t.numel() > 0:
                    self._sensor_matrix_n = (
                        fm_t.reshape(self.num_envs, -1, 3).sum(dim=1)[:, 2].abs()
                    )
        except Exception:
            pass

    def _quality_uv(self, i: int, arc_m: float) -> tuple[float, float]:
        return (float(self._pad_uv_actual[i, 0]), float(self._pad_uv_actual[i, 1]))

    def _apply_action(self) -> None:
        # Measure the real attached pad first; this measured pose gates force and removal.
        self._update_measured_pad_state()
        measured_gap = torch.where(
            self._pad_in_patch, self._pad_gap_m, torch.full_like(self._pad_gap_m, 0.20)
        )
        physical = self.cfg.enable_pad_physical_contact
        if physical:
            # 물리 안전 게이트는 control-step 평균뿐 아니라 센서 raw 순간값도 본다.
            # 평균화로 14 N 초과 피크가 숨는 것을 막는다.
            self._force_hard_violated |= (
                self._force_sensor_n > self.cfg.force_hard_limit_n)
        sensor_feedback = None
        if physical:
            # 폐루프 force tracking: 어드미턴스 피드백을 PhysX 센서 필터힘으로 교체 —
            # 센서힘이 목표에 못 미치면 z_offset 이 내려가 패드가 표면을 더 누른다.
            # fault(NaN) env 만 직전 모델 필터힘으로 대체하고 fallback 으로 계수한다.
            sensor_feedback = torch.where(
                self._sensor_fault, self.contact.filtered, self._force_sensor_filt_n
            )
        model_force = self.contact.step(
            self._force_cmd, self._is_side, measured_clearance=measured_gap,
            control_force_override=sensor_feedback,
        )
        if physical:
            # 물리 모드의 사용힘 = 센서 필터힘 (자유공간 0 N 은 fallback 이 아니라 사실).
            # 센서 fault 시에만 모델힘으로 대체하며, 그 사용량은 로그·결과에 명시된다.
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

        # Advance the reference path; quality uses measured pad coordinates, not this reference directly.
        self._arc += self._feed_cmd * self.cfg.sim.dt
        self._sim_time += self.cfg.sim.dt

        targets = torch.zeros((self.num_envs, 7), device=self.device)
        arcs = self._arc.detach().cpu().numpy()
        # 줄바꿈(레스터 line 전환·n_passes 경계) 램프 리미터 (진단 결과 — 인수인계서 17
        # 이후 발견): _pos_at_arc 는 줄 끝→다음 줄 시작을 순간이동으로 반환한다. 이걸
        # IK 목표에 그대로 먹이면 한 control step 안에 ~20mm(스텝오버) 또는 그 이상
        # (n_passes 경계에서 첫 줄로 복귀) 점프를 요구하게 되고, 실측으로 확인한 바
        # 이게 접촉력을 5.77N→18N 이상으로 튀게 만든다(경로 중간 라인전환 지점,
        # action=0 에서도 재현됨 — 정책·재폴리싱 로직과 무관한 경로추종 버그).
        # 고정: 목표 (u,v) 가 한 번에 max_step 이상 못 움직이게 램프한다. 평소 이송
        # 중엔 한 step 이동량이 max_step 보다 훨씬 작아 원래 동작과 동일하다.
        max_step = self.cfg.line_transition_speed_m_s * self.cfg.sim.dt
        for i in range(self.num_envs):
            uv_raw = np.asarray(self._pos_at_arc(float(arcs[i])), dtype=np.float64)
            prev = self._prev_uv[i]
            delta = uv_raw - prev
            dist = float(np.hypot(delta[0], delta[1]))
            if dist > max_step:
                uv = prev + delta / dist * max_step
            else:
                uv = uv_raw
            self._prev_uv[i] = uv
            targets[i, 0] = float(uv[0]) - self.cfg.patch_size_m[0] / 2 + self.cfg.patch_center_xy_m[0]
            targets[i, 1] = float(uv[1]) - self.cfg.patch_size_m[1] / 2 + self.cfg.patch_center_xy_m[1]
            targets[i, 2] = self.cfg.work_top_m + float(self.contact.command_clearance[i].clamp(-0.003, 0.08))

        # Convert the desired contact-face pose to a link_6 target using measured
        # face feedback.  This remains correct even if the imported fixed-joint
        # body origin differs from its authored mesh origin.
        desired_face_w = targets[:, :3] + self.scene.env_origins
        ee_now_w = self.robot.data.body_pose_w.torch[:, self._ee_body_id, :3]
        target_link6_w = ee_now_w + (desired_face_w - self._pad_face_w)
        target_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        target_quat_w[:, 0] = 1.0
        root = self.robot.data.root_pose_w.torch
        target_b_pos, target_b_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], target_link6_w, target_quat_w
        )
        self._ik.set_command(torch.cat((target_b_pos, target_b_quat), dim=1))

        ee_w = self.robot.data.body_pose_w.torch[:, self._ee_body_id]
        ee_b_pos, ee_b_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], ee_w[:, :3], ee_w[:, 3:7]
        )
        jac = self.robot.data.body_link_jacobian_w.torch[
            :, self._ee_jacobi_idx, :, :
        ][:, :, self._arm_joint_ids]
        q = self.robot.data.joint_pos.torch[:, self._arm_joint_ids]
        q_des = self._ik.compute(ee_b_pos, ee_b_quat, jac, q)
        self.robot.set_joint_position_target_index(target=q_des, joint_ids=self._arm_joint_ids)
        self.robot.set_joint_position_target_index(
            target=torch.zeros((self.num_envs, 1), device=self.device),
            joint_ids=[self._pad_joint_id],
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 부모(PolishEnv)와 동일하지만, 목표힘 baseline 을 전 env 공용 스칼라
        # (self.recipe.target_contact_force_n) 대신 env별 self._pass_base_force 로 쓴다.
        # repolish_mode 가 아니면 _pass_base_force 는 생성 시 값(=recipe 스칼라)에서
        # 절대 안 바뀌므로 부모와 완전히 동일하게 동작한다 (엄격한 상위호환).
        a = torch.clamp(actions, -1.0, 1.0)
        self._action_rate = (a - self._prev_action).square().mean(dim=1)
        self._prev_action = a.clone()
        self._force_cmd = self._pass_base_force * (1.0 + a[:, 0] * self.cfg.force_ratio_limit)
        self._feed_cmd = (self.recipe.feed_speed_mm_s / 1000.0) * (1.0 + a[:, 1] * self.cfg.feed_ratio_limit)
        self._force_accum = torch.zeros(self.num_envs, device=self.device)
        self._substep_n = 0
        self._force_sq_accum = torch.zeros(self.num_envs, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        r = super()._get_rewards()
        cfg = self.cfg
        if cfg.enable_pad_physical_contact and self._substep_n > 0:
            # 힘 overshoot (비대칭) — 명령 초과분만 벌한다 (인수인계서 18.3).
            overshoot = (self._force_mean - self._force_cmd - cfg.force_overshoot_tol_n).clamp(min=0.0)
            # 불안정 접촉 — control step 내 substep 힘 표준편차의 허용 초과분.
            mean = self._force_accum / self._substep_n
            var = (self._force_sq_accum / self._substep_n - mean * mean).clamp(min=0.0)
            std_excess = (var.sqrt() - cfg.unstable_std_tol_n).clamp(min=0.0)
            r = (r - cfg.w_force_overshoot * overshoot.clamp(max=2.0)
                 - cfg.w_unstable_contact * std_excess.clamp(max=2.0))
        return r

    def _quality_update(self):
        super()._quality_update()
        # 비접촉 가공 오류 가드 (인수인계서 18.3): 힘이 사실상 0인데 제거가 발생하면
        # 모델 배선 오류다 — 조용히 넘기지 않고 계수·경고한다 (구조상 0이어야 정상).
        if self._substep_n > 0:
            # 모델 계약(polishing_model.step): force<=0 이면 제거 0. 0<force<0.05 의
            # 스치는 접촉(접근 전환 스텝)은 미세 제거가 정상이므로 오류가 아니다.
            bad = (self._force_mean <= 0.0) & (
                (self._defect_removal + self._healthy_over) > 1e-9)
            n_bad = int(bad.sum())
            if n_bad:
                self._no_contact_removal_errors += n_bad
                print(f"[RobotPolishEnv] ⚠ 비접촉 가공 오류 {n_bad}건 "
                      f"(누적 {self._no_contact_removal_errors}) — 힘 배선 점검 필요")
            self.extras.setdefault("log", {})["Errors/no_contact_removal"] = float(
                self._no_contact_removal_errors)
        if self._repolish_mode and self.cfg.enable_pad_physical_contact and self._substep_n > 0:
            # 접촉 불안정 하드컷 (인수인계서 19-9 "접촉력 불안정") — 순간 힘 진동이
            # 여러 control step 연속되면 실패 처리. w_unstable_contact 보상 페널티와
            # 별개의 안전 게이트다.
            mean = self._force_accum / self._substep_n
            var = (self._force_sq_accum / self._substep_n - mean * mean).clamp(min=0.0)
            unstable_now = var.sqrt() > self.cfg.repolish_unstable_std_hard_n
            self._unstable_streak = torch.where(
                unstable_now, self._unstable_streak + 1, torch.zeros_like(self._unstable_streak))
            self._unstable_hard_violated |= (
                self._unstable_streak >= self.cfg.repolish_unstable_streak_limit)
        if self._repolish_mode and self._substep_n > 0:
            # pass 전체 평균힘 누적 — 미달분·안전예산 기반 다음 pass 힘 산정에 쓴다.
            self._pass_force_accum += self._force_accum / self._substep_n
            self._pass_force_cmd_accum += self._force_cmd
            self._pass_feed_accum += self._feed_cmd
            self._pass_action_accum += self._prev_action
            self._pass_force_n += 1.0
        if self.log_raw_steps and self.step_log:
            self.step_log[-1].update({
                "force_cmd_n": float(self._force_cmd[0]),
                "force_sensor_n": float(self._force_sensor_n[0]),          # = raw (하위호환)
                "force_sensor_raw_n": float(self._force_sensor_n[0]),
                "force_sensor_filtered_n": float(self._force_sensor_filt_n[0]),
                "force_model_n": float(self._force_model_n[0]),
                "force_used_n": float(self._force_used_n[0]),
                "sensor_matrix_n": float(self._sensor_matrix_n[0]),
                "sensor_fault": bool(self._sensor_fault[0]),
                "fallback_steps": int(self._fallback_steps[0]),
                "contact_mode": ("physical" if self.cfg.enable_pad_physical_contact
                                 else "model"),
                "pad_u_m": float(self._pad_uv_actual[0, 0]),
                "pad_v_m": float(self._pad_uv_actual[0, 1]),
                "pad_gap_m": float(self._pad_gap_m[0]),
            })

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # repolish_mode=False 면 PolishEnv 와 완전히 동일 — 기존 학습/평가 무변경.
        if not self._repolish_mode:
            return super()._get_dones()

        self._quality_update()
        done_path = self._arc >= self._path_len
        hard_violation = (self._force_hard_violated | self._thermal_hard_violated
                          | self._unstable_hard_violated)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminate = torch.zeros_like(done_path)

        # 1) 경로 완주 — 완주가 최우선이다. 도중에 하드위반이 있었어도 여기서 한 번에
        #    판정한다(_repolish_decide 의 safety_ok 가 그 위반들을 반영한다). 위반과
        #    완주가 같은 tick 에 겹칠 때 "정상 완주"를 "즉시 실패"로 잘못 분류하던
        #    버그를 여기서 고쳤다 — 순서가 반대였다.
        done_ids = done_path.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        continue_ids = []
        for i in done_ids:
            if self._repolish_decide(i):
                terminate[i] = True
            else:
                continue_ids.append(i)
        if continue_ids:
            self._soft_reset_path(torch.as_tensor(continue_ids, device=self.device).long())

        # 2) 경로 중간(완주 전)에 난 하드위반 — 완주를 기다리지 않고 즉시 중단.
        mid_violation = hard_violation & ~done_path
        for i in mid_violation.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
            terminate[i] = True
            reason = ("fail_overheat" if bool(self._thermal_hard_violated[i])
                      else "fail_unstable_contact" if bool(self._unstable_hard_violated[i])
                      else "fail_force_overload")
            fin = self._evaluate_quality(i)
            self._repolish_log[i] = {
                "outcome": reason, "passes": int(self._pass_count[i]),
                "before": self._before_metrics.get(i, fin), "final": fin,
                "quality_ok": False, "safety_ok": False,
            }

        # 3) 경로 도중 타임아웃 (드묾) — 위 두 경로에서 못 잡았으면 여기서 기록.
        for i in time_out.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
            if i not in self._repolish_log:
                fin = self._evaluate_quality(i)
                self._repolish_log[i] = {
                    "outcome": "fail_timeout", "passes": int(self._pass_count[i]),
                    "before": self._before_metrics.get(i, fin), "final": fin,
                    "quality_ok": False, "safety_ok": False,
                }
        return terminate, time_out

    def _reset_robot_pose(self, ids: torch.Tensor) -> None:
        """로봇 자세 + IK + 센서 필터를 기본 상태로. 표면(surface)은 건드리지 않는다 —
        _reset_idx(새 표면)와 _soft_reset_path(같은 표면, 다음 pass)가 공유한다."""
        root_state = self.robot.data.default_root_state.torch[ids].clone()
        root_state[:, :3] += self.scene.env_origins[ids]
        joint_pos = self.robot.data.default_joint_pos.torch[ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[ids].clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7], ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)
        self._ik.reset(ids)
        # reset 직후 잔여 접촉력/필터 상태로 인한 허위 spike 방지 (인수인계서 17.2 안전시험)
        self._force_sensor_filt_n[ids] = 0.0
        self._sensor_fault[ids] = False
        self._fallback_steps[ids] = 0
        self._unstable_streak[ids] = 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, device=self.device).long()
        super()._reset_idx(ids)
        self._reset_robot_pose(ids)
        # 새 표면(다음 재폴리싱 시퀀스) 시작 — 작업 카운터 초기화. _repolish_log 는
        # 외부 루프가 아직 못 읽었을 수 있으니 지우지 않는다.
        self._pass_count[ids] = 0
        self._unstable_hard_violated[ids] = False
        self._pass_base_force[ids] = self.recipe.target_contact_force_n
        self._pass_force_accum[ids] = 0.0
        self._pass_force_n[ids] = 0.0
        self._pass_force_cmd_accum[ids] = 0.0
        self._pass_feed_accum[ids] = 0.0
        self._pass_action_accum[ids] = 0.0
        self._pass_force_max[ids] = 0.0
        first_uv = np.asarray(self._pos_at_arc(0.0), dtype=np.float64)
        self._prev_uv[ids.cpu().numpy()] = first_uv
        for i in ids.cpu().tolist():
            self._repolish_prev_metrics.pop(i, None)
            self._repolish_pass_history.pop(i, None)

    def _soft_reset_path(self, ids: torch.Tensor) -> None:
        """재폴리싱: 같은 표면(surface)에서 다음 pass를 위해 경로·접촉·로봇만 리셋.
        표면 상태(clearcoat/scratch/온도)와 _before_metrics(시퀀스 최초 기준)는 유지한다."""
        self._reset_robot_pose(ids)
        self.contact.reset(ids, is_side=self._is_side)
        self._arc[ids] = 0.0
        self._sim_time[ids] = 0.0
        first_uv = np.asarray(self._pos_at_arc(0.0), dtype=np.float64)
        self._prev_uv[ids.cpu().numpy()] = first_uv
        self._prev_action[ids] = 0.0
        self._prev_force[ids] = 0.0
        self._force_hard_violated[ids] = False
        self._force_mean[ids] = 0.0
        self._force_accum[ids] = 0.0
        self._force_sq_accum[ids] = 0.0
        self._defect_removal[ids] = 0.0
        self._healthy_over[ids] = 0.0
        self._thermal_damage_delta[ids] = 0.0
        self._action_rate[ids] = 0.0
        self._pass_force_accum[ids] = 0.0
        self._pass_force_n[ids] = 0.0
        self._pass_force_cmd_accum[ids] = 0.0
        self._pass_feed_accum[ids] = 0.0
        self._pass_action_accum[ids] = 0.0
        self._pass_force_max[ids] = 0.0
        # _pass_base_force 는 여기서 건드리지 않는다 — _repolish_decide 가 다음 pass
        # 목표힘을 이미 정해서 넣어뒀다(정보 없으면 recipe 기준값 그대로).
        # thermal_hard_violated 는 일부러 유지한다 — 표면의 peak_temperature_c 는
        # pass 를 넘어 누적되는 값이라, 이전 pass 에서 이미 과열이었다면 계속 과열이다.

    def _apply_cooldown(self, i: int, cooldown_s: float) -> None:
        """무가공 냉각 — LiteraturePolishingModel 의 비접촉 냉각 경로를 quality_dt
        간격으로 반복 호출한다 (인수인계서 19-7/19-8, 실시간 시뮬 없이 상태만 전진)."""
        from learning.polytwin.polishing_model import ContactState
        n_ticks = max(1, int(round(cooldown_s / self.quality_dt)))
        cool = ContactState(pad_center_uv_m=(0.0, 0.0), contact_force_n=0.0,
                            rpm=0.0, feed_speed_m_s=0.0, in_contact=False)
        t = float(self._sim_time[i])
        for _ in range(n_ticks):
            self._model.step(self._surfaces[i], cool, dt_s=self.quality_dt, sim_time_s=t)
            t += self.quality_dt
        self._sim_time[i] = t

    def _tile_quality_diagnostic(self, i: int, tiles: tuple[int, int] = (5, 5)) -> dict:
        """Pass 종료 시 타일별 품질/안전 진단. 제어에는 사용하지 않는 출력 전용 값."""
        from learning.polytwin.gloss_proxy import _tile_slices, gu_from_relative

        result = self._gloss.evaluate(self._surfaces[i], tiles=tiles)
        gu_map = result["gu_map"]
        term_maps = result["term_maps"]
        limiter_terms = ["q_ra", "q_scratch", "q_uniformity", "q_thermal"]
        if self._gloss.cfg.w_clearcoat > 0.0:
            limiter_terms.append("q_clearcoat")
        limiter_terms = tuple(limiter_terms)
        limiter_stack = np.stack([term_maps[k] for k in limiter_terms], axis=-1)
        limiter_idx = np.argmin(limiter_stack, axis=-1)
        limiter_map = np.asarray(limiter_terms, dtype=object)[limiter_idx]

        surface = self._surfaces[i]
        gloss_cfg = self._gloss.cfg
        safety_um = gloss_cfg.clearcoat_failure_limit_um
        cc_min_map = np.zeros(tiles, dtype=np.float64)
        cc_initial_min_map = np.zeros(tiles, dtype=np.float64)
        cc_initial_mean_map = np.zeros(tiles, dtype=np.float64)
        q_cc_min_baseline_map = np.zeros(tiles, dtype=np.float64)
        q_cc_cellwise_worst_map = np.zeros(tiles, dtype=np.float64)
        for sl_i, sl_j, ti, tj in _tile_slices(surface.shape, tiles):
            initial = surface.initial_clearcoat_um[sl_i, sl_j]
            remaining = surface.clearcoat_remaining_um[sl_i, sl_j]
            initial_min = float(initial.min())
            remaining_min = float(remaining.min())
            cc_min_map[ti, tj] = remaining_min
            cc_initial_min_map[ti, tj] = initial_min
            cc_initial_mean_map[ti, tj] = float(initial.mean())
            q_cc_min_baseline_map[ti, tj] = np.clip(
                (remaining_min - safety_um) / max(initial_min - safety_um, 1e-6),
                0.0, 1.0)
            local_margin_ratio = (
                (remaining - safety_um) / np.maximum(initial - safety_um, 1e-6)
            )
            q_cc_cellwise_worst_map[ti, tj] = float(
                np.clip(local_margin_ratio.min(), 0.0, 1.0))

        # 세 식의 차이만 보기 위해 다른 품질항은 동일하게 유지한다. current는 현행
        # (최종 최소)/(초기 평균), min_baseline은 (최종 최소)/(초기 최소),
        # cellwise_worst는 위치별 안전여유 잔존비 중 최솟값이다.
        geometric_weight_sum = (
            gloss_cfg.w_ra + gloss_cfg.w_scratch
            + gloss_cfg.w_uniformity + gloss_cfg.w_clearcoat
        )

        def _gu_for_q_clearcoat(q_clearcoat_map, perfect_mutable: bool):
            log_geom = gloss_cfg.w_clearcoat * np.log(
                np.maximum(q_clearcoat_map, 1e-6))
            if not perfect_mutable:
                log_geom += (
                    gloss_cfg.w_ra * np.log(np.maximum(term_maps["q_ra"], 1e-6))
                    + gloss_cfg.w_scratch
                    * np.log(np.maximum(term_maps["q_scratch"], 1e-6))
                    + gloss_cfg.w_uniformity
                    * np.log(np.maximum(term_maps["q_uniformity"], 1e-6))
                )
            log_q = (
                log_geom / geometric_weight_sum
                + gloss_cfg.w_thermal
                * np.log(np.maximum(term_maps["q_thermal"], 1e-6))
            )
            return gu_from_relative(np.exp(log_q), gloss_cfg.upper_anchor_gu)

        gu_min_baseline_map = _gu_for_q_clearcoat(q_cc_min_baseline_map, False)
        gu_cellwise_worst_map = _gu_for_q_clearcoat(q_cc_cellwise_worst_map, False)

        # Clearcoat 보전도와 누적 열손상은 추가 폴리싱으로 좋아질 수 없다. 반대로
        # Ra·scratch·균일도를 모두 완벽(q=1)으로 놓은 낙관적 GU 상한을 식별로 비교한다.
        optimistic_ceiling_gu_map = _gu_for_q_clearcoat(term_maps["q_clearcoat"], True)
        optimistic_ceiling_min_baseline_map = _gu_for_q_clearcoat(
            q_cc_min_baseline_map, True)
        optimistic_ceiling_cellwise_worst_map = _gu_for_q_clearcoat(
            q_cc_cellwise_worst_map, True)

        fail = gu_map < self.cfg.repolish_target_gu
        # 추가가공 가능은 품질 성공 예측이 아니라, Clearcoat 설계 여유가 남았다는 뜻뿐이다.
        cc_rework_ok = (
            cc_min_map > self.cfg.clearcoat_safety_limit_um
            + self.cfg.repolish_cc_safety_margin_um
        )
        rework = fail & cc_rework_ok
        stop = fail & ~cc_rework_ok
        ceiling_blocked = fail & (optimistic_ceiling_gu_map < self.cfg.repolish_target_gu)
        ceiling_allows = fail & ~ceiling_blocked
        plausible_rework = ceiling_allows & cc_rework_ok
        budget_stop = ceiling_allows & ~cc_rework_ok
        fail_indices = [f"{x}:{y}" for x, y in np.argwhere(fail)]
        rework_indices = [f"{x}:{y}" for x, y in np.argwhere(rework)]
        stop_indices = [f"{x}:{y}" for x, y in np.argwhere(stop)]
        ceiling_blocked_indices = [f"{x}:{y}" for x, y in np.argwhere(ceiling_blocked)]
        plausible_rework_indices = [f"{x}:{y}" for x, y in np.argwhere(plausible_rework)]
        budget_stop_indices = [f"{x}:{y}" for x, y in np.argwhere(budget_stop)]
        limiter_counts = {
            k: int(((limiter_map == k) & fail).sum()) for k in limiter_terms
        }
        return {
            "tile_gu_mean": float(gu_map.mean()),
            "tile_gu_min": float(gu_map.min()),
            "tile_gu_fail_count": int(fail.sum()),
            "tile_rework_candidate_count": int(rework.sum()),
            "tile_stop_count": int(stop.sum()),
            "tile_optimistic_ceiling_gu_mean": float(optimistic_ceiling_gu_map.mean()),
            "tile_optimistic_ceiling_gu_min": float(optimistic_ceiling_gu_map.min()),
            "tile_min_baseline_ceiling_blocked_count": int(
                ((gu_map < self.cfg.repolish_target_gu)
                 & (optimistic_ceiling_min_baseline_map
                    < self.cfg.repolish_target_gu)).sum()),
            "tile_cellwise_ceiling_blocked_count": int(
                ((gu_map < self.cfg.repolish_target_gu)
                 & (optimistic_ceiling_cellwise_worst_map
                    < self.cfg.repolish_target_gu)).sum()),
            "tile_ceiling_blocked_count": int(ceiling_blocked.sum()),
            "tile_plausible_rework_count": int(plausible_rework.sum()),
            "tile_budget_stop_count": int(budget_stop.sum()),
            "tile_fail_indices": ";".join(fail_indices),
            "tile_rework_indices": ";".join(rework_indices),
            "tile_stop_indices": ";".join(stop_indices),
            "tile_ceiling_blocked_indices": ";".join(ceiling_blocked_indices),
            "tile_plausible_rework_indices": ";".join(plausible_rework_indices),
            "tile_budget_stop_indices": ";".join(budget_stop_indices),
            "tile_limiter_counts_json": json.dumps(limiter_counts, sort_keys=True),
            "tile_gu_map_json": json.dumps(np.round(gu_map, 3).tolist()),
            "tile_gu_min_baseline_map_json": json.dumps(
                np.round(gu_min_baseline_map, 3).tolist()),
            "tile_gu_cellwise_worst_map_json": json.dumps(
                np.round(gu_cellwise_worst_map, 3).tolist()),
            "tile_optimistic_ceiling_gu_map_json": json.dumps(
                np.round(optimistic_ceiling_gu_map, 3).tolist()),
            "tile_optimistic_ceiling_min_baseline_map_json": json.dumps(
                np.round(optimistic_ceiling_min_baseline_map, 3).tolist()),
            "tile_optimistic_ceiling_cellwise_worst_map_json": json.dumps(
                np.round(optimistic_ceiling_cellwise_worst_map, 3).tolist()),
            "tile_q_clearcoat_current_map_json": json.dumps(
                np.round(term_maps["q_clearcoat"], 6).tolist()),
            "tile_q_clearcoat_min_baseline_map_json": json.dumps(
                np.round(q_cc_min_baseline_map, 6).tolist()),
            "tile_q_clearcoat_cellwise_worst_map_json": json.dumps(
                np.round(q_cc_cellwise_worst_map, 6).tolist()),
            "tile_limiter_map_json": json.dumps(limiter_map.tolist()),
            "tile_cc_min_map_json": json.dumps(np.round(cc_min_map, 3).tolist()),
            "tile_cc_initial_min_map_json": json.dumps(
                np.round(cc_initial_min_map, 3).tolist()),
            "tile_cc_initial_mean_map_json": json.dumps(
                np.round(cc_initial_mean_map, 3).tolist()),
        }

    def _repolish_decide(self, i: int) -> bool:
        """한 pass 종료 시점의 성공/재시도/실패 판정 (인수인계서 19장 5~9번).

        True 면 시퀀스 종료(다음 tick에 프레임워크가 새 표면으로 리셋). False 면
        같은 표면에서 다음 pass 를 계속한다 — 이 경우 냉각과 다음 pass 목표힘
        산정까지 이 함수 안에서 끝낸다.
        """
        cfg = self.cfg
        self._pass_count[i] += 1
        fin = self._evaluate_quality(i)
        prev = self._repolish_prev_metrics.get(i)
        start = prev if prev is not None else self._before_metrics.get(i, fin)
        quality_ok = (fin["gu"] >= cfg.repolish_target_gu
                      and fin["ra"] <= cfg.t_ra_pass_max_um
                      and fin["rz"] <= cfg.t_rz_pass_max_um)
        # 안전성은 이번 pass 도중 실제로 있었던 하드위반들을 전부 반영한다 — 예전엔
        # 접촉 불안정만 봤는데, 힘 초과/과열이 나도 여기 안 걸리면 "완주했으니
        # 안전하다"고 잘못 통과시키는 구멍이 있었다.
        force_ok = not bool(self._force_hard_violated[i])
        thermal_ok = fin["temperature_peak_c"] < cfg.thermal_hard_limit_c
        cc_ok = fin["cc_min"] >= cfg.clearcoat_safety_limit_um
        unstable_ok = not bool(self._unstable_hard_violated[i])
        safety_ok = force_ok and thermal_ok and cc_ok and unstable_ok

        n_steps = float(self._pass_force_n[i].clamp(min=1.0))
        pass_force_n = float(self._pass_force_accum[i] / n_steps)
        pass_removal_um = max(0.0, float(start["cc_min"]) - fin["cc_min"])
        cc_budget = fin["cc_min"] - cfg.clearcoat_safety_limit_um
        gu_gap = max(0.0, cfg.repolish_target_gu - fin["gu"]) / max(cfg.repolish_target_gu, 1e-6)
        ra_gap = max(0.0, fin["ra"] - cfg.t_ra_pass_max_um) / max(cfg.t_ra_pass_max_um, 1e-6)
        rz_gap = max(0.0, fin["rz"] - cfg.t_rz_pass_max_um) / max(cfg.t_rz_pass_max_um, 1e-6)
        shortfall_ratio = max(gu_gap, ra_gap, rz_gap)
        removal_per_n = pass_removal_um / pass_force_n if pass_force_n > 0.5 else 0.0
        pass_diag = {
            "pass": int(self._pass_count[i]),
            "decision": "",
            "base_force_n": float(self._pass_base_force[i]),
            "force_cmd_mean_n": float(self._pass_force_cmd_accum[i] / n_steps),
            "force_used_mean_n": pass_force_n,
            "force_used_max_n": float(self._pass_force_max[i]),
            "force_action_mean": float(self._pass_action_accum[i, 0] / n_steps),
            "feed_cmd_mean_mm_s": float(self._pass_feed_accum[i] / n_steps) * 1000.0,
            "feed_action_mean": float(self._pass_action_accum[i, 1] / n_steps),
            "fallback_steps": int(self._fallback_steps[i]),
            "gu_before": float(start["gu"]), "gu_after": fin["gu"],
            "ra_before_um": float(start["ra"]), "ra_after_um": fin["ra"],
            "rz_before_um": float(start["rz"]), "rz_after_um": fin["rz"],
            "scratch_before_um": float(start["scratch"]),
            "scratch_after_um": fin["scratch"],
            "clearcoat_before_um": float(start["cc_min"]),
            "clearcoat_after_um": fin["cc_min"],
            "clearcoat_removed_um": pass_removal_um,
            "clearcoat_budget_um": cc_budget,
            "temperature_peak_c": fin["temperature_peak_c"],
            "thermal_damage_peak": fin["thermal_damage_peak"],
            "shortfall_ratio": shortfall_ratio,
            "removal_per_force_um_n": removal_per_n,
            "desired_extra_force_n": None,
            "max_extra_force_n": None,
            "next_base_force_n": None,
            "quality_ok": quality_ok, "safety_ok": safety_ok,
        }
        pass_diag.update(self._tile_quality_diagnostic(i))

        def _record(decision: str, next_force: float | None = None):
            pass_diag["decision"] = decision
            pass_diag["next_base_force_n"] = next_force
            self._repolish_pass_history.setdefault(i, []).append(pass_diag.copy())

        def _finish(outcome: str):
            _record(outcome)
            b = self._before_metrics.get(i, fin)
            self._repolish_log[i] = {
                "outcome": outcome, "passes": int(self._pass_count[i]),
                "before": b, "final": fin,
                "quality_ok": quality_ok, "safety_ok": safety_ok,
                "pass_history": list(self._repolish_pass_history.get(i, [])),
            }

        if quality_ok and safety_ok:
            _finish("success")
            return True
        if not safety_ok:
            reason = ("fail_clearcoat" if not cc_ok
                      else "fail_overheat" if not thermal_ok
                      else "fail_force_overload" if not force_ok
                      else "fail_unstable_contact")
            _finish(reason)
            return True
        if int(self._pass_count[i]) >= cfg.repolish_max_passes:
            _finish("fail_max_passes")
            return True
        if prev is not None:
            # Scratch 하나만 계속 줄어도 GU/Ra/Rz 가 악화되는 과가공을 "개선"으로
            # 통과시키지 않는다. 최종 판정 지표 중 하나라도 허용오차 이상 퇴행하면
            # 즉시 중단하고, 퇴행 없이 품질 지표가 하나 이상 좋아져야 계속한다.
            quality_regressed = (
                (prev["gu"] - fin["gu"]) > cfg.repolish_gu_improve_eps
                or (fin["ra"] - prev["ra"]) > cfg.repolish_ra_improve_eps_um
                or (fin["rz"] - prev["rz"]) > cfg.repolish_rz_improve_eps_um
            )
            quality_improved = (
                (fin["gu"] - prev["gu"]) > cfg.repolish_gu_improve_eps
                or (prev["ra"] - fin["ra"]) > cfg.repolish_ra_improve_eps_um
                or (prev["rz"] - fin["rz"]) > cfg.repolish_rz_improve_eps_um
            )
            if quality_regressed:
                _finish("fail_quality_regression")
                return True
            if not quality_improved:
                _finish("fail_no_improvement")
                return True

        # 선택적 근접 마무리 게이트. 전체 레스터 3차 pass는 기존 100환경에서
        # 진입 GU<68인 12/12가 개선되지 않았으므로, finish_rule 평가에서만
        # 2차 종료 후 목표 근접 + Ra/Rz 통과 표면에 한정한다. 기본값 0은 비활성.
        gate_pass = int(cfg.repolish_finish_gate_after_pass)
        if gate_pass > 0 and int(self._pass_count[i]) >= gate_pass:
            finish_eligible = (
                fin["gu"] >= cfg.repolish_finish_min_gu
                and fin["ra"] <= cfg.t_ra_pass_max_um
                and fin["rz"] <= cfg.t_rz_pass_max_um
            )
            if not finish_eligible:
                _finish("fail_finish_not_eligible")
                return True

        # ── 미달이지만 안전 — 다음 pass 목표힘을 "미달분·안전예산" 기반으로 재산정 ──
        # "정해진 스텝만큼 무조건 올리기"는 clearcoat을 안전선 아래로 뚫을 수 있어
        # 채택하지 않는다. 대신 방금 pass 에서 실측한 (힘당 clearcoat 감소율)로
        # 다음 힘을 역산하고, 남은 안전예산을 넘지 않는 선에서만 올린다.
        if cc_budget <= cfg.repolish_cc_safety_margin_um:
            # 더 깎을 안전 여유가 사실상 없다 — 억지로 재시도하지 않는다.
            _finish("fail_clearcoat_budget")
            return True

        base_force = self.recipe.target_contact_force_n
        # _pass_base_force 뒤에 정책의 +force_ratio_limit residual 이 곱해지므로,
        # 기본힘 자체를 0.85*hard_limit 로 두면 최종 명령은 hard limit 를 넘을 수 있다.
        # 최대 양의 residual 이후에도 지정 비율 안에 남도록 역산한다.
        hard_cap = (cfg.repolish_force_cap_ratio * cfg.force_hard_limit_n
                    / (1.0 + cfg.force_ratio_limit))

        next_force = base_force
        if pass_force_n > 0.5 and pass_removal_um > 1e-6:
            desired_extra_force = (shortfall_ratio * cfg.repolish_force_gain_um) / removal_per_n
            max_extra_force = max(0.0, cc_budget - cfg.repolish_cc_safety_margin_um) / removal_per_n
            pass_diag["desired_extra_force_n"] = desired_extra_force
            pass_diag["max_extra_force_n"] = max_extra_force
            if desired_extra_force > max_extra_force and shortfall_ratio > cfg.repolish_infeasible_shortfall:
                # 안전예산 안에서 낼 수 있는 최대 힘으로도 남은 미달을 못 채울 것으로
                # 추정된다 — 억지로 pass 를 반복하는 대신 정직하게 실패 처리한다.
                _finish("fail_infeasible")
                return True
            extra_force = min(desired_extra_force, max_extra_force)
            next_force = max(base_force, min(base_force + extra_force, hard_cap))

        self._pass_base_force[i] = next_force
        self._repolish_prev_metrics[i] = fin
        _record("continue", next_force)
        # 냉각 후 같은 표면에서 다음 pass (19-7/19-8)
        self._apply_cooldown(i, cfg.repolish_cooldown_s)
        return False
