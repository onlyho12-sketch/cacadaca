"""로봇 팔 시연 — 2대 정책 비교 또는 16대 병렬 정책 재생.

    ~/isaacsim/python.sh learning/rl/demo_arm.py            # GUI
    ~/isaacsim/python.sh learning/rl/demo_arm.py --num_envs 16 --all_policy \
        --checkpoint learning/rl/thermal/logs/2026-08-29_02-51-36/model_250.pt
    python learning/rl/demo_arm.py --headless # 검증 실행

기본 구성 (env 2개, 좌=baseline / 우=정책):
  · M0609 + 폴리셔 (usd/env/Collected_m0609_with_polisher — 상대경로 수집본. 원본 wrapper 는
    /home/rokey/... 절대참조가 깨져 있어 사용 불가)
  · 표면 patch 는 로봇 앞 바닥의 회색 판. 스크래치는 얇은 막대 마커 —
    잔여 깊이에 따라 빨강(깊음)→노랑→초록(제거됨)으로 변한다.
  · 팔은 DifferentialIK 로 패드 목표(경로점 + 어드미턴스 z_offset)를 추종한다.
    접촉력·품질은 검증된 해석 모델(contact.py + LiteraturePolishingModel)이 계산한다 —
    원 시뮬도 강체 충돌 OFF 였으므로(A안) 팔은 시각·기구 검증 역할이다.
"""
import argparse
import sys

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default=None,
                    help="기본: <repo>/learning/rl/champion/model_bc.pt")
parser.add_argument("--surface_seed", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=2,
                    help="시각화할 로봇/환경 수. 16이면 Isaac Lab 기본 격자로 4x4 배치")
parser.add_argument("--all_policy", action="store_true",
                    help="모든 환경에 checkpoint 정책 적용. 미지정 시 env0=baseline, 나머지=정책")
parser.add_argument("--same_surface", action="store_true",
                    help="모든 환경에 같은 surface_seed 사용 (정책 비교용)")
parser.add_argument("--render_every", type=int, default=3,
                    help="몇 physics substep마다 렌더할지 (16대 권장값 6)")
parser.add_argument("--max_seconds", type=float, default=None,
                    help="시뮬 시간 상한 (기본: 에피소드 완주까지)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint is None:
    args.checkpoint = _os.path.join(_REPO_ROOT, "learning", "rl", "champion", "model_bc.pt")
app = AppLauncher(args).app

from importlib import metadata  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from learning.polytwin import config as PC  # noqa: E402
from learning.polytwin.gloss_proxy import LiteratureGlossProxyModel  # noqa: E402
from learning.polytwin.path_executor import load_calibrated_config, raster_waypoints  # noqa: E402
from learning.polytwin.polishing_model import ContactState, LiteraturePolishingModel  # noqa: E402
from learning.polytwin.surface_state import make_flat_patch  # noqa: E402
from learning.rl.env.contact import VirtualPadContact  # noqa: E402
from learning.rl.env.polish_env import _load_recipe  # noqa: E402
from learning.rl.env.polish_env_cfg import PolishEnvCfg  # noqa: E402

ROBOT_USD = _os.path.join(_REPO_ROOT, "usd", "env", "Collected_m0609_with_polisher",
                          "m0609_with_polisher.usd")
PATCH_SIZE = (0.20, 0.20)   # 시연용 확대 — 학습/판정은 0.12 (정책은 국소 관측이라 무관)
PATCH_CENTER = (0.45, 0.0)          # 로봇 베이스 기준 앞쪽 45cm (M0609 도달반경 내)
E = args.num_envs
if E < 1:
    parser.error("--num_envs must be >= 1")
if args.render_every < 1:
    parser.error("--render_every must be >= 1")


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"joint_1": 0.0, "joint_2": -1.05, "joint_3": 1.45,
                       "joint_4": 0.0, "joint_5": 1.15, "joint_6": 0.0, "pad_joint": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint_[1-6]"],
                                       stiffness=10000.0, damping=500.0),
            # ⚠ pad_joint 는 위치제어로 고정한다. 강성 0(자유 관절)이면 tool0 의 회전을
            #   관절이 그대로 흡수해 원판 자세가 따라오지 않는다 — 실측: IK 는 명령을 0.3° 로
            #   달성하는데 패드면 법선은 홈값 그대로였다 (필요 회전축과 pad_joint 축이 일치).
            "pad": ImplicitActuatorCfg(joint_names_expr=["pad_joint"],
                                       stiffness=2000.0, damping=100.0),
        },
    )


def main():
    # 16대 GUI에서도 초기화가 빠른 Fabric을 사용한다. 뷰포트는 아래에서 별도 USD 카메라에
    # 직접 연결하므로 기본 Perspective 카메라 동기화 문제에 의존하지 않는다.
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, use_fabric=True))
    scene_cfg = DemoSceneCfg(num_envs=E, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=5000.0, color=(0.9, 0.93, 1.0))
    light.func("/World/Light", light)

    # 16개 환경 전체에 확실한 명암을 주는 대각선 상부 보조광.
    origins_np = scene.env_origins.detach().cpu().numpy()
    grid_center = origins_np.mean(axis=0)
    grid_span = max(float(np.ptp(origins_np[:, 0])), float(np.ptp(origins_np[:, 1])), 2.5)
    key_light = sim_utils.SphereLightCfg(
        radius=max(2.0, 0.35 * grid_span), intensity=90000.0,
        color=(1.0, 0.88, 0.72),
    )
    key_light.func(
        "/World/OverviewKeyLight", key_light,
        translation=(float(grid_center[0] - 0.35 * grid_span),
                     float(grid_center[1] - 0.45 * grid_span),
                     max(6.0, 0.9 * grid_span)),
    )

    # 작업대(회색) 위에 차 도장 패널(네이비) — 팔이 자연스러운 자세로 작업하는 허리 높이
    WORK_TOP = 0.40
    pedestal = sim_utils.CuboidCfg(
        size=(PATCH_SIZE[0] + 0.06, PATCH_SIZE[1] + 0.06, WORK_TOP - 0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)))
    plate = sim_utils.CuboidCfg(
        size=(PATCH_SIZE[0] + 0.04, PATCH_SIZE[1] + 0.04, 0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.12, 0.35), roughness=0.15, metallic=0.6))
    for i in range(E):
        pedestal.func(f"/World/envs/env_{i}/Pedestal", pedestal,
                      translation=(PATCH_CENTER[0], PATCH_CENTER[1], (WORK_TOP - 0.05) / 2))
        plate.func(f"/World/envs/env_{i}/Workpiece", plate,
                   translation=(PATCH_CENTER[0], PATCH_CENTER[1], WORK_TOP - 0.025))

    sim.reset()
    _force_environment_visibility(E)
    _set_overview_camera(sim, scene.env_origins)
    _install_overview_usd_camera(scene.env_origins)
    robot: Articulation = scene["robot"]

    # ★ 수동 씬에서는 기본 관절상태(홈 자세)를 명시적으로 써야 한다 — 안 쓰면 0-자세(수직)로
    #   시작해 기준 방향 캡처가 오염되고 IK 가 불가능 명령을 받는다 (ik_diag 로 실측 확인).
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.torch.clone(),
                                   robot.data.default_joint_vel.torch.clone())
    robot.reset()
    for _ in range(60):    # 1초 안정화
        robot.set_joint_position_target_index(
            target=robot.data.default_joint_pos.torch.clone(),
            joint_ids=list(range(robot.num_joints)))
        scene.write_data_to_sim(); sim.step(render=False); scene.update(1 / 60)
    _force_environment_visibility(E)
    # Kit visualizer가 시작 직후 viewport 카메라를 한 번 덮어쓸 수 있어 안정화 뒤 재설정한다.
    _set_overview_camera(sim, scene.env_origins)
    _install_overview_usd_camera(scene.env_origins)

    # ── 팔 IK 세팅 (튜토리얼 run_diff_ik 패턴) ──
    arm_joints = [robot.joint_names.index(f"joint_{k}") for k in range(1, 7)]
    pad_joint = robot.joint_names.index("pad_joint")
    pad_body = robot.body_names.index("sander_pad")   # (숨김 대상 중복본 — 로깅용 아님)
    # ★ IK 대상은 tool0 — sander_pad 는 pad_joint 로 3000 rpm 회전하는 링크라 그 자세를
    #   IK 로 고정하려 하면 스핀과 싸운다 (실측: 명령해도 패드면 법선 ·down = −0.399).
    #   원판 법선은 스핀축과 같아 회전에 불변이므로, 비회전 링크 tool0 을 제어하면 된다.
    # ★ IK 대상 = link_6. 화면에 보이는 샌더는 link_6/quick_mount/sanding_kit 에 달려 있다
    #   (자산에 샌더 메쉬가 두 벌: sander_pad 아래 복사본은 팔을 따라오지 않아 숨긴다).
    ee_body = robot.body_names.index("link_6")
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=E, device=sim.device)
    ee_jacobi_idx = ee_body - 1 if robot.is_fixed_base else ee_body

    PAD_HALF_THICKNESS = 0.0115   # 실측: sander_pad 링크프레임 extent Y = 0.023 m

    # ── 폴리싱 로직 (PolishEnv 와 동일 구성 요소) ──
    env_cfg = PolishEnvCfg()
    recipe = _load_recipe(env_cfg.recipe_json_path)
    from dataclasses import replace as _dc_replace
    recipe = _dc_replace(recipe, n_passes=1)     # 시연용 — 판정 숫자는 eval_ppo(2pass) 기준
    contact = VirtualPadContact(E, sim.device, dt=1 / 60, lag_offset=0.0)
    is_side = torch.zeros(E, dtype=torch.bool, device=sim.device)
    contact.reset(is_side=is_side)
    cal = load_calibrated_config()
    model = LiteraturePolishingModel(cal)
    gloss = LiteratureGlossProxyModel()
    # 기본 비교는 같은 표면, 병렬 정책 재생은 서로 다른 seed의 손상·clearcoat 맵을 사용한다.
    seeds = ([args.surface_seed] * E if args.same_surface or not args.all_policy
             else [args.surface_seed + 97 * i for i in range(E)])
    surfaces = [make_flat_patch(PATCH_SIZE, 0.002, seed=seed, with_scratches=True)
                for seed in seeds]
    scratch_specs = [_extract_scratch_segments(surface) for surface in surfaces]

    spacing = recipe.step_over_spacing_ratio * PC.PAD_DIAMETER_M
    lines = raster_waypoints(PATCH_SIZE, spacing)
    line_len = np.array([float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) for p0, p1, _ in lines])
    path_len = float(line_len.sum()) * recipe.n_passes

    def pos_at_arc(a):
        one = float(line_len.sum())
        a = a % one if a < path_len else one - 1e-9
        for (p0, p1, _), L in zip(lines, line_len):
            if a <= L:
                t = a / L
                return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            a -= L
        return lines[-1][1]

    # ★ 패드 자세·오프셋 — 원 프로젝트 규약 + 실측
    #   agent.py `_ee_orientation`: EE 의 패드 접촉축을 −normal(표면 안쪽)에 정렬한다.
    #   실측 (BBoxCache.ComputeRelativeBound, link_6 프레임, 최하단 메쉬 57개):
    #     패드 중심 XY = (0.0000, −0.0156),  접촉면 Z = −0.0902,  지름 ≈ 0.070 m
    #   → 접촉축은 link_6 로컬 −Z. 수평 패널이면 −Z 가 세계 −Z 를 향하면 되므로 자세는 항등.
    PAD_OFFSET_LINK6 = np.array([0.0, -0.0156, -0.0902])

    def _quat_to_R(q):                      # q = (w, x, y, z)
        w, x, y, z = [float(v) for v in q]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    quat_cmd_world = torch.zeros(E, 4, device=sim.device); quat_cmd_world[:, 0] = 1.0
    tool_offset_world = PAD_OFFSET_LINK6.copy()      # 자세가 항등이라 그대로 세계 오프셋
    _root0 = robot.data.root_pose_w.torch
    _, quat_ref_b = subtract_frame_transforms(
        _root0[:, 0:3], _root0[:, 3:7], _root0[:, 0:3], quat_cmd_world)
    quat_ref_b = quat_ref_b.clone()
    print(f"[demo] link_6 → 패드접촉면 오프셋 {np.round(tool_offset_world, 4)} m, 목표자세 항등")

    # ── 정책 로드 (legacy 11-D / thermal 14-D 자동 판별) ──
    policy = _load_policy(args.checkpoint, sim.device)

    # ── 스크래치 마커: 잔여 비율별 색 프로토타입 5단계 ──
    colors = [(0.9, 0.1, 0.1), (0.95, 0.5, 0.1), (0.95, 0.85, 0.1),
              (0.55, 0.9, 0.2), (0.1, 0.85, 0.3)]
    proto = {f"lv{k}": sim_utils.CuboidCfg(
        size=(1.0, 0.004, 0.002),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(0.3*x for x in c)))
        for k, c in enumerate(colors)}
    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/scratches", markers=proto))

    # 환경 상태등: 파랑=실행 중, 초록=전체 통과, 노랑=재폴리싱 필요, 빨강=안전 실패.
    status_proto = {
        "running": sim_utils.SphereCfg(radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.45, 1.0), emissive_color=(0.05, 0.2, 0.7))),
        "pass": sim_utils.SphereCfg(radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.9, 0.2), emissive_color=(0.05, 0.6, 0.1))),
        "retry": sim_utils.SphereCfg(radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.75, 0.05), emissive_color=(0.7, 0.4, 0.0))),
        "unsafe": sim_utils.SphereCfg(radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.08, 0.05), emissive_color=(0.7, 0.02, 0.01))),
    }
    status_markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/env_status", markers=status_proto))
    _update_status_markers(status_markers, scene.env_origins, [0] * E)

    arc = torch.zeros(E, device=sim.device)
    prev_force = torch.zeros(E, device=sim.device)
    prev_action = torch.zeros(E, 2, device=sim.device)
    sim_t = 0.0
    control_n = 0
    force_cmd = torch.full((E,), recipe.target_contact_force_n, device=sim.device)
    feed_cmd = torch.full((E,), recipe.feed_speed_mm_s / 1000.0, device=sim.device)

    policy_desc = "all envs=policy" if args.all_policy else "env0=baseline, env1..=policy"
    print(f"[demo] recipe {recipe} | path {path_len:.2f} m | {E} envs | {policy_desc}")
    print(f"[demo] surface seeds: {seeds}")
    force_accum = torch.zeros(E, device=sim.device)
    sub = 0
    while app.is_running():
        # 초기 viewport가 검게 남는 Isaac Lab 3 Kit 초기화 순서에 대비해 몇 차례 재적용.
        if sub in (0, 30, 120):
            _set_overview_camera(sim, scene.env_origins)
            _install_overview_usd_camera(scene.env_origins)
        # ── 20Hz control step (3 substep 마다) ──
        if sub % 3 == 0 and sub > 0:
            f_mean = force_accum / 3.0
            force_accum.zero_()
            # 품질 갱신 + 관측 구성 (PolishEnv 와 동일 의미)
            obs = _build_obs(surfaces, arc, f_mean, feed_cmd, prev_force, prev_action,
                             pos_at_arc, path_len, force_cmd, env_cfg, sim.device)
            for i in range(E):
                uv = pos_at_arc(float(arc[i]))
                model.step(surfaces[i], ContactState(uv, float(f_mean[i]), recipe.rpm,
                                                     float(feed_cmd[i])), 0.05, sim_t)
            prev_force = f_mean.clone()
            # 행동: 전체 정책 재생 또는 env0 baseline/나머지 정책 비교.
            a = torch.zeros(E, 2, device=sim.device)
            if policy is not None:
                start = 0 if args.all_policy else 1
                if start < E:
                    a[start:] = policy(obs[start:]).clamp(-1, 1)
            prev_action = a.clone()
            force_cmd = recipe.target_contact_force_n * (1 + a[:, 0] * env_cfg.force_ratio_limit)
            feed_cmd = (recipe.feed_speed_mm_s / 1000.0) * (1 + a[:, 1] * env_cfg.feed_ratio_limit)
            control_n += 1
            if control_n % 40 == 0:   # 2초마다 스크래치 마커 갱신
                _update_markers(markers, scratch_specs, surfaces, scene.env_origins)

        # ── 60Hz physics substep ──
        f = contact.step(force_cmd, is_side)
        force_accum += f
        arc += feed_cmd * (1 / 60)
        sim_t += 1 / 60

        # 패드 목표: 경로점 + z = clearance (팔이 어드미턴스 압입을 그대로 재현)
        targets = torch.zeros(E, 7, device=sim.device)
        for i in range(E):
            uv = pos_at_arc(float(arc[i]))
            targets[i, 0] = uv[0] - PATCH_SIZE[0] / 2 + PATCH_CENTER[0]
            targets[i, 1] = uv[1] - PATCH_SIZE[1] / 2 + PATCH_CENTER[1]
            targets[i, 2] = WORK_TOP + float(contact.actual_clearance[i].clamp(min=-0.003))
        # 접촉면 목표 → tool0 원점 목표 (자세가 고정이므로 상수 오프셋)
        targets[:, 0] -= float(tool_offset_world[0])
        targets[:, 1] -= float(tool_offset_world[1])
        targets[:, 2] -= float(tool_offset_world[2])
        # world → base frame, 자세는 홈 기준 고정
        root = robot.data.root_pose_w.torch
        ident = torch.zeros(E, 4, device=sim.device); ident[:, 0] = 1.0
        tgt_b_pos, _ = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], targets[:, 0:3] + scene.env_origins, ident)
        ik.set_command(torch.cat([tgt_b_pos, quat_ref_b], dim=1))

        jac = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, :][:, :, arm_joints]
        ee_w = robot.data.body_pose_w.torch[:, ee_body]
        ee_b_pos, ee_b_quat = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
        q = robot.data.joint_pos.torch[:, arm_joints]
        q_des = ik.compute(ee_b_pos, ee_b_quat, jac, q)
        robot.set_joint_position_target_index(target=q_des, joint_ids=arm_joints)
        # ⚠ 시연에서 패드 스핀은 끈다. 실측: pad_joint 의 회전축이 원판 법선과 **수직**이라
        #   3000 rpm 을 주면 원판이 레코드판이 아니라 **바퀴처럼 굴러** 순간순간 날이 아래를
        #   향한다 (패드면 법선이 XZ 평면에서 연속 회전, y성분 0 으로 관측).
        #   회전은 어차피 20 Hz 렌더에서 보이지 않고, 제거량·접촉력은 해석 모델(rpm 파라미터)이
        #   계산하므로 시각적으로 끄는 것이 물리 결과에 영향을 주지 않는다.
        robot.set_joint_position_target_index(
            target=torch.zeros((E, 1), device=sim.device), joint_ids=[pad_joint])
        scene.write_data_to_sim()
        sim.step(render=(sub % args.render_every == 0))
        scene.update(1 / 60)
        sub += 1
        if sub % 600 == 0:                        # 10초마다 생존 로그 + 패드 자세 검증
            _ny = _quat_to_R(robot.data.body_pose_w.torch[0, ee_body, 3:7].cpu().numpy()) @ \
                  np.array([0.0, 0.0, -1.0])
            _cmd_q = quat_cmd_world[0].cpu().numpy(); _ach_q = robot.data.body_pose_w.torch[0, ee_body, 3:7].cpu().numpy()
            _dot = abs(float(np.dot(_cmd_q, _ach_q)))
            print(f"[demo] pad quat {np.round(robot.data.body_pose_w.torch[0,pad_body,3:7].cpu().numpy(),3)} "
                  f"pad_joint각 {float(robot.data.joint_pos.torch[0, pad_joint]):.3f} rad", flush=True)
            print(f"[demo] tool0 자세 명령{np.round(_cmd_q,3)} 달성{np.round(_ach_q,3)} "
                  f"각오차={np.degrees(2*np.arccos(min(_dot,1.0))):.1f}°", flush=True)
            print(f"[demo] 패드면 법선(세계) {np.round(_ny,3)}  ·down={float(-_ny[2]):.3f} "
                  f"(1.000 이면 면접촉)", flush=True)
            print(f"[demo] t={sim_t:.0f}s  진행 {min(arc.min().item()/path_len,1)*100:.0f}%  "
                  f"F={contact.filtered.max().item():.1f}N  "
                  f"추종오차={float((robot.data.body_pose_w.torch[:, ee_body, 0:3] - (targets[:, 0:3] + scene.env_origins)).norm(dim=1).max())*100:.1f}cm", flush=True)

        done = bool((arc >= path_len).all())
        over = args.max_seconds is not None and sim_t >= args.max_seconds
        if done or over:
            break

    # ── 결과 ──
    print("\n=== 16대 병렬 시연 결과 (SYNTHETIC / GU proxy) ===")
    final_status = []
    for i in range(E):
        name = "policy" if args.all_policy or i > 0 else "baseline"
        q = model.evaluate(surfaces[i])
        g = gloss.evaluate(surfaces[i])["summary"]
        gu = float(g["gu_mean"])
        scratch = float(q["max_residual_scratch_um"])
        ra = float(q["ra_um"])
        rz = float(q["rz_um"])
        cc = float(q["clearcoat_min_um"])
        peak_c = float(q["temperature_peak_c"])
        damage = float(q["thermal_damage_peak"])
        unsafe = cc < env_cfg.clearcoat_safety_limit_um or peak_c > env_cfg.thermal_hard_limit_c
        passed = (gu >= 70.0 and ra <= env_cfg.t_ra_pass_max_um
                  and rz <= env_cfg.t_rz_pass_max_um and not unsafe)
        status = "UNSAFE" if unsafe else "PASS" if passed else "RETRY"
        final_status.append(3 if unsafe else 1 if passed else 2)
        print(f"  env{i:02d} {name:8s} [{status:6s}] | GU {gu:5.2f} | "
              f"Ra {ra:.3f} | Rz {rz:.3f} μm | scratch {scratch:.3f} μm | "
              f"CC {cc:.2f} μm | peak {peak_c:.2f}°C | damage {damage:.4f}")
    _update_status_markers(status_markers, scene.env_origins, final_status)
    print("[demo] 상태등: 초록=전체 통과, 노랑=재폴리싱 필요, 빨강=과열/clearcoat 안전 실패")
    if sim.has_gui:
        print("[demo] GUI 유지 중 — 창을 닫으면 종료됩니다.")
        while app.is_running():
            sim.step()


# ── helpers ───────────────────────────────────────────────────────────────
def _set_overview_camera(sim, env_origins):
    """환경 실제 좌표 범위를 기준으로 대각선 상공 overview 카메라를 설정한다."""
    p = env_origins.detach().cpu().numpy()
    center = p.mean(axis=0)
    span_x = float(np.ptp(p[:, 0])) if len(p) > 1 else 0.0
    span_y = float(np.ptp(p[:, 1])) if len(p) > 1 else 0.0
    span = max(span_x, span_y, 2.5)
    target = (float(center[0] + PATCH_CENTER[0]), float(center[1]), 0.42)
    if len(p) <= 2:
        eye = (target[0] + 0.75, target[1] - 0.85, 1.15)
    else:
        # +X/-Y 대각선 방향에서 전체 격자를 약 40° 아래로 내려다본다.
        eye = (target[0] + 0.92 * span,
               target[1] - 1.08 * span,
               max(6.5, 0.95 * span))
    try:
        sim.set_camera_view(eye=eye, target=target)
        print(f"[demo] overview camera eye={tuple(round(v, 2) for v in eye)} "
              f"target={tuple(round(v, 2) for v in target)} "
              f"grid=({span_x:.2f} x {span_y:.2f})m")
    except Exception as exc:
        print(f"[demo] ⚠ overview camera 설정 실패: {exc}")


def _overview_eye_target(env_origins):
    p = env_origins.detach().cpu().numpy()
    center = p.mean(axis=0)
    span_x = float(np.ptp(p[:, 0])) if len(p) > 1 else 0.0
    span_y = float(np.ptp(p[:, 1])) if len(p) > 1 else 0.0
    span = max(span_x, span_y, 2.5)
    target = (float(center[0] + PATCH_CENTER[0]), float(center[1]), 0.42)
    if len(p) <= 2:
        eye = (target[0] + 0.75, target[1] - 0.85, 1.15)
    else:
        eye = (target[0] + 0.92 * span,
               target[1] - 1.08 * span,
               max(6.5, 0.95 * span))
    return eye, target


def _install_overview_usd_camera(env_origins):
    """실제 USD 카메라를 만들고 Kit 활성 viewport에 직접 연결한다."""
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Sdf, UsdGeom

        eye, target = _overview_eye_target(env_origins)
        stage = omni.usd.get_context().get_stage()
        path = "/World/OverviewCamera"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        camera.GetFocalLengthAttr().Set(18.0)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        pose = Gf.Matrix4d(1.0).SetLookAt(
            Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
        xform.AddTransformOp().Set(pose)
        camera.GetPrim().CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(*target))

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("active Kit viewport 없음")
        viewport.set_active_camera(path)

        # Isaac Sim 6 renderer에도 같은 카메라를 명시적으로 연결한다.
        from isaacsim.core.rendering_manager import ViewportManager
        ViewportManager.set_camera_view(path, eye=list(eye), target=list(target))
        print(f"[demo] active viewport camera={viewport.get_active_camera()} ({path})")
    except Exception as exc:
        print(f"[demo] ⚠ USD overview camera 연결 실패: {exc}")


def _force_environment_visibility(num_envs):
    """Kit partial-visualization 잔여 상태와 상관없이 16개 환경을 USD에서 표시한다."""
    try:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        visible = 0
        for i in range(num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
            if prim and prim.IsValid():
                UsdGeom.Imageable(prim).MakeVisible()
                visible += 1
        print(f"[demo] USD 환경 visibility 강제 표시: {visible}/{num_envs}")
    except Exception as exc:
        print(f"[demo] ⚠ 환경 visibility 설정 실패: {exc}")


def _extract_scratch_segments(surface):
    """스크래치 맵에서 마커용 선분(중심·방향·길이) 근사 추출 — 연결성분 PCA."""
    from scipy import ndimage
    labels, n = ndimage.label(surface.initial_scratch_depth_um > 0.05)
    segs = []
    for k in range(1, n + 1):
        i_idx, j_idx = np.nonzero(labels == k)     # ⚠ 배열 축 = (x, y) 순서 — 스왑 금지
        if len(i_idx) < 5:
            continue
        pts = np.stack([i_idx, j_idx], 1).astype(float) * surface.resolution_m
        c = pts.mean(0)
        d = pts - c
        w, v = np.linalg.eigh(d.T @ d / len(d))
        axis = v[:, -1]
        length = float((d @ axis).max() - (d @ axis).min())
        cells = (labels == k)
        segs.append({"center": c, "angle": float(np.arctan2(axis[1], axis[0])),
                     "length": max(length, 0.01), "cells": cells})
    return segs


def _update_markers(markers, segs_by_env, surfaces, env_origins, WORK_TOP=0.40):
    pos, quat, scale, idx = [], [], [], []
    for e, surf in enumerate(surfaces):
        remaining_map = np.clip(surf.initial_scratch_depth_um - surf.cumulative_removal_um, 0, None)
        for s in segs_by_env[e]:
            init = surf.initial_scratch_depth_um[s["cells"]].max()
            rem = remaining_map[s["cells"]].max()
            ratio = float(rem / max(init, 1e-6))
            level = min(4, int((1.0 - ratio) * 5))         # 0=빨강(온전) → 4=초록(제거)
            ox, oy = float(env_origins[e][0]), float(env_origins[e][1])
            pos.append([s["center"][0] - PATCH_SIZE[0] / 2 + PATCH_CENTER[0] + ox,
                        s["center"][1] - PATCH_SIZE[1] / 2 + PATCH_CENTER[1] + oy, WORK_TOP + 0.003])
            half = s["angle"] / 2.0
            quat.append([np.cos(half), 0.0, 0.0, np.sin(half)])
            scale.append([s["length"], 1.0, 1.0])
            idx.append(level)
    import torch as th
    markers.visualize(translations=th.tensor(pos), orientations=th.tensor(quat),
                      scales=th.tensor(scale), marker_indices=th.tensor(idx))


def _update_status_markers(markers, env_origins, status_indices):
    """각 환경 위 상태등을 표시한다: 0 running / 1 pass / 2 retry / 3 unsafe."""
    import torch as th
    positions = env_origins.detach().cpu().clone()
    positions[:, 2] = 1.65
    orientations = th.zeros((len(status_indices), 4), dtype=th.float32)
    orientations[:, 0] = 1.0
    markers.visualize(
        translations=positions,
        orientations=orientations,
        marker_indices=th.tensor(status_indices, dtype=th.int32),
    )


def _build_obs(surfaces, arc, f_mean, feed_cmd, prev_force, prev_action,
               pos_at_arc, path_len, force_cmd, env_cfg, device):
    """PolishEnv._get_observations 와 같은 14차원 열 관측."""
    E = len(surfaces)
    stats = np.zeros((E, 7), dtype=np.float32)
    for i in range(E):
        s = surfaces[i]
        uv = pos_at_arc(float(arc[i]))
        res, R = s.resolution_m, 0.5 * PC.PAD_RADIUS_M
        i0 = max(int((uv[0] - R) / res), 0); i1 = min(int((uv[0] + R) / res) + 1, s.shape[0])
        j0 = max(int((uv[1] - R) / res), 0); j1 = min(int((uv[1] + R) / res) + 1, s.shape[1])
        sl = (slice(i0, i1), slice(j0, j1))
        remaining = np.clip(s.initial_scratch_depth_um[sl] - s.cumulative_removal_um[sl], 0, None)
        stats[i] = (
            remaining.mean(), remaining.max(), s.cumulative_removal_um[sl].mean(),
            s.clearcoat_remaining_um[sl].min() - env_cfg.clearcoat_safety_limit_um,
            s.temperature_c[sl].mean(), s.peak_temperature_c[sl].max(),
            s.thermal_damage_proxy[sl].mean(),
        )
    st = torch.as_tensor(stats, device=device)
    f = f_mean.unsqueeze(1)
    progress = (arc / path_len).clamp(0, 1).unsqueeze(1)
    return torch.cat([
        f / 10.0, (f - force_cmd.unsqueeze(1)) / 5.0, (f - prev_force.unsqueeze(1)) / 5.0,
        feed_cmd.unsqueeze(1) * 100.0, progress,
        st[:, 0:1] / 2.0, st[:, 1:2] / 2.0, st[:, 2:3] / 5.0, st[:, 3:4] / 20.0,
        prev_action,
        (st[:, 4:5] - PC.AMBIENT_TEMPERATURE_C) / 40.0,
        (st[:, 5:6] - PC.AMBIENT_TEMPERATURE_C) / 60.0,
        st[:, 6:7] / PC.THERMAL_DAMAGE_MAX,
    ], dim=1)


def _load_policy(ckpt_path, device):
    """legacy 11-D 또는 thermal 14-D actor를 checkpoint에서 직접 재구성."""
    try:
        from rsl_rl.models.mlp_model import MLPModel
        from tensordict import TensorDict

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ck["actor_state_dict"]
        obs_dim = int(state["mlp.0.weight"].shape[1])
        dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
        actor = MLPModel(dummy, {"actor": ["policy"]}, "actor", 2,
                         hidden_dims=[128, 128], activation="elu", obs_normalization=True,
                         distribution_cfg={"class_name": "GaussianDistribution",
                                           "init_std": 0.3, "std_type": "scalar"}).to(device)
        actor.load_state_dict(state)
        actor.eval()

        def _policy(obs_tensor):
            td = TensorDict({"policy": obs_tensor[:, :obs_dim]}, batch_size=[len(obs_tensor)])
            with torch.no_grad():
                return actor(td)          # deterministic mean
        print(f"[demo] policy loaded: {ckpt_path} (obs_dim={obs_dim})")
        return _policy
    except Exception as exc:
        print(f"[demo] ⚠ 정책 로드 실패 ({exc}) — env1 도 baseline 으로 동작")
        return None


if __name__ == "__main__":
    main()
    app.close()
