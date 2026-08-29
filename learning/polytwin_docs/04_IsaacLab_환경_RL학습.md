# 04. Isaac Lab 환경과 강화학습

## 1. 학습 목표

하나의 자동차 mesh를 외우지 않고 다양한 국소 표면에서 다음을 동시에 학습한다.

```text
목표 접촉력 유지
Feed Speed 유지
표면 법선에 패드 정렬
경로 추종
미작업 영역 coverage
Scratch와 Ra/Rz 개선
20° GU proxy 목표 달성
Clearcoat 과다 제거 방지
정상부 불필요 연마 방지
```

## 2. 초기 정책

### 2.1 기준 정책

초기 정책의 중심은 현재 v5 제어기다.

```text
경로 생성기
+ 표면법선 기반 기준자세
+ RMPFlow
+ 가상 스프링/힘 제어
+ 상단·측면 안전 제한
```

### 2.2 PPO 잔차 정책

PPO는 기준정책을 제거하지 않고 작은 잔차를 추가한다.

```python
baseline_command = baseline_controller(observation, recipe)
residual_action = ppo_actor(observation)
command = safety_filter(baseline_command + scale(residual_action))
```

Actor의 초기 평균출력은 0에 가깝게 만들어 첫 정책이 기준정책과 비슷하게 동작하게 한다.

```text
actor final mean layer: zero 또는 매우 작은 초기화
initial action mean    : 약 0
initial action std     : 탐색 가능하되 safety intervention이 지배하지 않는 값
```

초기 action std는 고정 추측값으로 확정하지 않고 평면 baseline rollout에서 비교한다.

## 3. Isaac Lab 이식 검증

RL을 켜기 전에 `residual_action=0`으로 기존 v5와 비교한다.

```text
TCP trajectory
Force time series/RMSE/p95/peak
Path error
Normal alignment error
Contact ratio
Completion time
Safety events
```

좌표축, asset scale, physics dt, controller dt, force filtering 및 안전제어가 맞지 않으면 RL 학습
전에 수정한다.

## 4. 절차적 학습 표면

| Surface family | 형상 |
|---|---|
| Plane | 평면과 경사 평면 |
| Convex cylinder | 한 방향 양의 곡률 |
| Concave cylinder | 한 방향 음의 곡률 |
| Convex sphere | 두 방향 양의 곡률 |
| Concave sphere | 두 방향 음의 곡률 |
| Saddle | 서로 반대 부호의 주곡률 |
| Freeform | 저주파 자유곡면 |
| Transition | 평면↔곡면, 볼록↔오목 전이 |

각 episode에서 위치, 방향, `k1/k2`, 경로방향, 초기 Scratch와 Roughness seed를 랜덤화한다.

## 5. 실제 새 물체 입력

새 자동차 또는 물체는 다음 파이프라인으로 정책에 연결한다.

```text
mesh/point cloud 로드
→ 작업 가능 patch 분할
→ 법선과 주곡률 계산
→ robot reachability와 collision 반영 경로 생성
→ 각 waypoint를 LocalSurfaceContext로 변환
→ 동일 PPO 정책 실행
```

PPO는 물체 이름이나 고정 vertex id를 보지 않는다.

## 6. Observation

```python
observation = {
    "joint_position_norm": ...,
    "joint_velocity_norm": ...,
    "tcp_position_error_surface_frame": [ex, ey, ez],
    "tcp_orientation_error_surface_frame": ...,
    "surface_normal_tcp_frame": [nx, ny, nz],
    "path_tangent_tcp_frame": [tx, ty, tz],
    "principal_curvatures_norm": [k1, k2],
    "contact_force_norm": ...,
    "force_error_norm": ...,
    "force_derivative_norm": ...,
    "feed_speed_error_norm": ...,
    "path_progress": ...,
    "local_coverage_crop": ...,
    "local_scratch_crop": ...,
    "local_removal_crop": ...,
    "local_clearcoat_crop": ...,
    "local_gu_proxy_crop": ...,
    "dwell_norm": ...,
    "pass_count_norm": ...,
    "boundary_distance_norm": ...,
    "gravity_tcp_frame": ...,
    "region_and_mount_onehot": ...,
    "process_recipe_context": ...,
    "previous_action": ...,
    "phase_onehot": ...,
}
```

Map crop을 그대로 CNN에 넣거나 encoder latent로 변환할 수 있다. scalar/vector와 map encoder의
출력을 합쳐 actor/critic에 입력한다.

## 7. Action

기존 담당자의 확정 잔차사양이 있으면 우선한다. 기하학적 잔차안:

```python
action = {
    "delta_normal_offset": [-1, 1],
    "delta_feed_scale": [-1, 1],
    "delta_tool_tilt_x": [-1, 1],
    "delta_tool_tilt_y": [-1, 1],
}
```

임피던스 잔차안:

```python
action = {
    "delta_stiffness_ratio": [-1, 1],
    "delta_damping_ratio": [-1, 1],
    "delta_force_target_ratio": [-1, 1],
    "delta_feed_ratio": [-1, 1],
}
```

두 안을 동시에 사용하면 action별 ablation을 수행한다. 물리단위 scale은 action=0 baseline의
잔여오차 p95로 결정하고 rate limit을 둔다.

## 8. Process recipe context

BO 또는 recipe sampler가 episode 시작 시 제공한다.

```python
ProcessRecipe = {
    "target_contact_force_n": ...,
    "feed_speed_mm_s": ...,
    "rpm": ...,
    "step_over_spacing_ratio": ...,
    "tool_path_type": ...,
    "target_tool_angle_rad": ...,
    "compound_profile_id": ...,
    "pad_profile_id": ...,
}
```

Dwell과 Pass Count는 independent action이 아니라 실제 trajectory에서 누적한다.

## 9. 한 control step

```text
1. Actor action 출력
2. action clipping/rate limit
3. 기준 제어기 명령 계산
4. 잔차 결합
5. 안전 필터
6. physics substep 진행
7. force/pose/velocity 집계
8. PolishingModel.step()
9. GU proxy와 품질지표 계산
10. reward/termination 계산
11. observation과 로그 반환
```

## 10. Reward

### 10.1 제어항

```python
r_force = -huber(force_error_n / force_tolerance_n)
r_speed = -huber(feed_error / feed_tolerance)
r_path = -huber(path_error_m / path_tolerance_m)
r_tilt = -huber(tilt_error_rad / tilt_tolerance_rad)
r_force_rate = -clip(abs(force_rate) / force_rate_limit, 0, 1)
r_action_rate = -mean(square(action_t - action_t_minus_1))
```

### 10.2 품질 개선항

절대 최종값을 매 step 반복 보상하지 않고 이전 step 대비 개선량을 사용한다.

```python
r_scratch = previous_scratch_cost - current_scratch_cost
r_ra = previous_ra_target_cost - current_ra_target_cost
r_rz = previous_rz_target_cost - current_rz_target_cost
r_gu = current_gu_proxy_norm - previous_gu_proxy_norm
r_coverage = max(0, coverage_t - coverage_t_minus_1)
```

### 10.3 보전·효율항

```python
p_healthy_removal = -healthy_overremoval_delta_norm
p_clearcoat = -clearcoat_risk_delta_norm
p_heat = -heat_proxy_delta_norm
p_revisit = -revisit_without_quality_gain
p_time = -step_cost
```

### 10.4 결합

```python
reward = (
    w_force * r_force
    + w_speed * r_speed
    + w_path * r_path
    + w_tilt * r_tilt
    + w_force_rate * r_force_rate
    + w_scratch * r_scratch
    + w_ra * r_ra
    + w_rz * r_rz
    + w_gu * r_gu
    + w_coverage * r_coverage
    + w_healthy * p_healthy_removal
    + w_clearcoat * p_clearcoat
    + w_heat * p_heat
    + w_revisit * p_revisit
    + w_time * p_time
)
```

모든 항은 `[0,1]` 또는 비교가능한 범위로 정규화하고 raw/normalized/weighted 값을 따로 로그에
남긴다.

## 11. Phase별 reward mask

| Phase | 활성항 |
|---|---|
| Approach | path, tilt, time |
| Contact acquisition | force ramp, force-rate, tilt |
| Polishing | control + quality + coverage + preservation |
| Line transition | safety, path re-entry, action-rate |
| Completion | success bonus 1회 |

접촉하지 않은 Approach에서 품질항을 주지 않는다.

## 12. 종료조건

### 성공

```text
GU proxy 목표와 균일도 통과
Ra/Rz 목표 통과
잔존 Scratch 통과
Coverage 통과
Clearcoat와 정상부 과다제거 통과
경로 완료
```

### 실패

```text
force hard limit
force spike
tool penetration
non-pad collision
joint limit
contact loss unrecoverable
clearcoat safety failure
heat proxy failure
NaN/Inf
```

### Truncation

```text
max steps
max simulation time
```

## 13. Curriculum

```text
Stage A: 평면 직선, 고정 recipe, 제어 reward
Stage B: 평면 Scratch, 품질모델과 GU reward 추가
Stage C: 볼록/오목 단일곡률
Stage D: 구면/Saddle 두 방향 곡률
Stage E: Raster와 Step-over, line transition
Stage F: 자유곡면과 곡률전이
Stage G: process recipe randomization
Stage H: 미학습 synthetic surface 평가
Stage I: 미학습 자동차/물체 mesh 평가
```

다음 Stage로 이동할 때 train reward가 아니라 고정 validation seed의 안전·Force·Path·품질
통과율을 사용한다.

## 14. PPO 학습

```python
env = PolyTwinPolishingEnv(
    baseline_controller=v5_adapter,
    surface_generator=procedural_generator,
    polishing_model=LiteraturePolishingModel,
    gloss_model=LiteratureGlossProxyModel,
    safety_filter=region_safety_filter,
)

assert action_zero_baseline_passes(env)

for iteration in range(max_iterations):
    rollout = collect_parallel_rollout(env, policy, split="train")
    policy.ppo_update(rollout)
    validation = evaluate_fixed_manifest(policy, split="validation")
    save_latest()
    save_best_if_constraints_pass(validation)

evaluate_once(best_policy, split="test")
```

## 15. Reward hacking 시험

- 접촉하지 않고 경로만 이동
- 같은 정상셀 반복
- 아주 느리게 움직여 Force reward만 획득
- 안전필터가 항상 행동을 수정하도록 큰 action 출력
- Scratch를 남기고 주변 평면만 연마
- Clearcoat를 과도하게 제거해 Ra/GU만 개선
- episode 완료를 회피해 step reward 누적

각 편법정책은 자동 회귀시험으로 만들고 총보상이 정상정책보다 낮아야 한다.

## 16. 일반화 평가

```text
Train              : 학습 surface manifest
Validation         : 미사용 곡률·방향·seed
Test synthetic     : 미학습 freeform/transition
Test object mesh   : 미학습 자동차 또는 물체 patch
```

정책 선택은 Validation까지만 사용하고 Test 결과로 다시 tuning하지 않는다.
