# 05. BO와 PPO 통합

## 1. 역할

```text
BO : 디지털 트윈에서 공정 레시피 탐색
PPO: 선택된 레시피를 곡면에서 매 스텝 안정적으로 실행
```

| 구분 | BO | PPO |
|---|---|---|
| 선택 시점 | episode 시작 전 | control step마다 |
| 입력 | 초기 표면상태, recipe 후보 | 현재 로봇·표면·품질 상태, recipe |
| 출력 | Force/RPM/Feed/Step-over 등 | 잔차 위치·속도·자세·제어계수 |
| 평가 | episode 최종 품질·시간·안전 | step reward와 rollout return |
| 모델 | black-box surrogate + acquisition | actor/critic policy network |

## 2. BO를 사용하는 이유

Force, RPM, Feed, Step-over를 episode 중 고정하고 최종점수 한 번으로 평가하면 순차 제어가
아니라 저차원 black-box optimization이다. PPO로 레시피까지 무작정 탐색하면 긴 episode를
대량 반복해야 하고 제어정책과 공정조건 효과가 섞인다.

BO는 다음에 적합하다.

- 평가비용이 큰 full polishing episode
- 연속 파라미터가 비교적 적은 recipe
- 품질과 안전의 다중 제약
- 적은 초기표본에서 다음 후보를 선택하는 탐색

## 3. Search space

```yaml
target_contact_force_n:
  type: continuous
  domain: intersection(literature_candidate, region_safe_domain)

feed_speed_mm_s:
  type: continuous
  min: 1.0
  max: 8.0

rpm:
  type: continuous_or_integer
  min: 3000
  max: 6000

step_over_spacing_ratio:
  type: continuous
  bounds: PT_DESIGN

tool_path_type:
  type: categorical
  choices: [raster, contour]

tool_angle:
  type: continuous
  bounds: region_safe_config
```

패드 크기는 고정이므로 search variable이 아니다. Dwell과 Pass Count는 별도 고정 recipe로
제공하거나 실제 trajectory 결과로 계산한다.

## 4. BO evaluation

```python
def evaluate_recipe(recipe, surface_manifest, ppo_checkpoint):
    episode_results = []

    for surface_id, seed in surface_manifest:
        result = run_digital_twin_episode(
            recipe=recipe,
            policy=ppo_checkpoint,
            surface_id=surface_id,
            seed=seed,
        )
        episode_results.append(result)

    return aggregate_objective_and_constraints(episode_results)
```

한 표면의 운 좋은 결과가 recipe를 지배하지 않도록 여러 곡면과 seed의 평균·하위 percentile·
worst case를 함께 사용한다.

## 5. 목적함수

최소화 비용 예시:

```python
cost = (
    w_gu * gloss_shortfall_norm
    + w_ra * ra_target_distance_norm
    + w_rz * rz_target_distance_norm
    + w_scratch * residual_scratch_norm
    + w_map * removal_nonuniformity_norm
    + w_healthy * healthy_overremoval_norm
    + w_time * process_time_norm
    + w_heat * heat_proxy_norm
)
```

```python
gloss_shortfall_norm = clip(
    (gu_target - mean_gu_proxy) / gu_target,
    0.0,
    1.0,
)
```

## 6. Hard constraints

```python
feasible = (
    force_peak_n <= force_hard_limit_n
    and clearcoat_min_um >= clearcoat_safety_limit_um
    and healthy_overremoval_um <= healthy_overremoval_limit_um
    and heat_proxy_peak <= heat_proxy_limit
    and collision_count == 0
    and completion_rate >= completion_rate_limit
)
```

`feasible=False`인 recipe는 GU가 높아도 최적후보로 선택하지 않는다.

## 7. BO 반복

```text
초기 recipe 표본 생성
→ 각 recipe를 디지털 트윈에서 반복평가
→ BO surrogate 학습
→ constrained acquisition으로 다음 recipe 제안
→ 디지털 트윈 평가
→ 데이터셋 갱신
→ 종료조건까지 반복
→ 최적 feasible recipe export
```

초기 표본은 한쪽 구간에 몰리지 않도록 space-filling design을 사용한다.

## 8. BO surrogate와 품질모델 구분

```text
LiteraturePolishingModel
  한 episode 안에서 표면상태를 매 step 갱신

BO surrogate
  recipe를 넣었을 때 episode 최종점수를 빠르게 근사
```

두 모델은 역할과 저장파일을 분리한다.

```text
polishing_model_version
bo_surrogate_version
```

## 9. PPO 학습과 BO 순서

순환 의존을 줄이기 위한 기본 순서:

```text
1. 중간 기준 recipe로 PPO 제어정책 학습
2. PPO checkpoint 고정
3. 고정 PPO로 BO recipe 탐색
4. BO recipe 분포를 context로 PPO 추가학습
5. PPO checkpoint 다시 고정
6. BO 재탐색
7. 성능변화가 작아질 때 종료
```

BO와 PPO를 매 update마다 동시에 바꾸면 성능변화 원인을 구분하기 어렵다. 각 outer iteration에
`ppo_checkpoint_id`와 `bo_dataset_version`을 고정한다.

## 10. Recipe JSON

```json
{
  "schema_version": "1.0",
  "recipe_id": "recipe_00042",
  "source": "synthetic_bo_v1",
  "polishing_model_version": "literature_polishing_v1",
  "gloss_model_version": "literature_gu_proxy_v1",
  "pad_profile_id": "fixed_pad_v1",
  "compound_profile_id": "compound_v1",
  "target_contact_force_n": 7.0,
  "feed_speed_mm_s": 4.0,
  "rpm": 4000,
  "step_over_spacing_ratio": 0.40,
  "tool_path_type": "raster",
  "tool_angle_rad": [0.0, 0.0],
  "gu_target": 70.0,
  "gu_type": "literature_proxy",
  "constraints_config_hash": "..."
}
```

## 11. PPO 실행

```python
recipe = load_recipe_json(recipe_path)
process_context = normalize_recipe(recipe)

for step in episode:
    observation = build_observation(process_context=process_context)
    residual = ppo_policy(observation)
    command = safety_filter(baseline_controller(recipe) + residual)
    state = env.step(command)
```

PPO는 recipe 원본을 변경하지 않고 곡면상태에 맞는 잔차만 출력한다.

## 12. 결과 추적

최종 한 episode는 다음 조합으로 재현한다.

```text
polishing_model_version
+ gloss_model_version
+ ppo_checkpoint_id
+ recipe_id
+ surface_id
+ seed
+ resolved_config_hash
```

## 13. BO와 PPO 비교를 위한 테스트

1. 같은 recipe에서 baseline과 PPO 비교.
2. 같은 PPO에서 manual recipe와 BO recipe 비교.
3. BO surrogate 예측값과 실제 digital-twin episode 결과 비교.
4. Train 곡면 recipe와 Test 곡면 recipe의 성능차 비교.
5. BO가 hard constraint 경계 밖 후보를 최종선택하지 않는지 확인.
6. PPO가 recipe 목표와 싸워 지속적으로 action saturation을 내지 않는지 확인.

## 14. 알고리즘 변경

PPO는 기본안이다. CHEQ 또는 다른 잔차 RL 알고리즘이 이미 구현돼 있으면 동일 observation,
action, reward, termination, surface manifest로 비교한다. 알고리즘 변경이 논문 기반 물리모델과
GU proxy 정의를 바꾸지는 않는다.

## 15. 참고문헌

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*. [arXiv](https://arxiv.org/abs/1707.06347)
- Snoek et al. (2012), *Practical Bayesian Optimization of Machine Learning Algorithms*. [NeurIPS](https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html)
