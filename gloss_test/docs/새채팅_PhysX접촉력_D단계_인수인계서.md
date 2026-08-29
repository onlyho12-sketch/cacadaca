# PolyTwin — PhysX 접촉력 검증 + 재폴리싱(D단계) 작업 인수인계서

작성일: 2026-08-29
작업 경로: `/home/rokey/cacadaca`
선행 문서: `gloss_test/docs/새채팅_전체작업_인수인계서.md` (이 문서를 먼저 읽을 것 — 프로젝트 전체 목표·용어·근거 분류·판정값이 그쪽에 정의돼 있다)

이 문서는 위 선행 인수인계서의 "17장 PhysX 접촉력 검증"부터 "19장 재폴리싱 상태기계"까지를 실제로 구현한 세션의 기록이다.

---

## 0. 새 채팅에서 가장 먼저 지켜야 할 규칙

선행 인수인계서 0장과 동일하다. 요약:

1. 사용자가 요청한 범위만 작업한다.
2. 실제 작업(파일 수정·실행·학습) 전에 무엇을 왜 어떻게 할지 설명하고 승인을 받는다.
3. `scripts/polishing_v5.py`, `scripts/polishing_v5_modules/`는 참고만 하고 수정하지 않는다.
4. 기존 체크포인트(`learning/rl/champion/`, `learning/rl/thermal/`)는 덮어쓰지 않는다 — 이번 세션 산출물은 전부 `learning/rl/robot/` 아래 새 경로에 저장했다.
5. 문헌 기반 합성값과 실제 계측값을 구분한다. `GU proxy`는 실제 GU가 아니다.
6. 실제 PhysX 접촉력이 0인 상태를 "force sensor 검증 완료"라고 하지 않는다.

---

## 1. 이번 세션에서 한 일 요약 (A→D 단계)

선행 인수인계서 26장의 순서 기준:

| 단계 | 내용 | 상태 |
|---|---|---|
| A | 패드-작업면 PhysX 물리 충돌 활성화 | 완료 |
| B | ContactSensor normal force 측정 + 힘 로그 5종 분리 + 제거/온도 모델에 검증된 힘 연결 | 완료 |
| — | 접촉력 검증 게이트 6개 phase (자유공간·정적·이동·안전·병렬·품질연결) | 전부 PASS |
| C | 환경 통일 + RobotPolishEnv(물리모드) 기준 새 BC/PPO 학습·평가 | 완료 (단, §6 캐비어트 참고) |
| D | 재폴리싱 상태기계(닦기→검사→냉각→재시도) | 구현 + 버그 2건 발견·수정 + 스모크 검증 진행 중 |
| — | (D 작업 중 발견) 레스터 줄바꿈 힘 스파이크 근본원인 발견·수정 | 완료·검증됨 |

---

## 2. A+B — PhysX 접촉력 검증 (완료, 25장 완료조건 전부 충족)

### 2.1 무엇을 바꿨나

- `learning/rl/env/robot_polish_env_cfg.py`: `enable_pad_physical_contact`(기본 `False`) 플래그와 물리 모드 전용 파라미터 추가(dt 1/120s·decimation 6, compliant stiffness/damping, contact/rest offset, solver iteration, 센서 필터 alpha 등).
- `learning/rl/env/robot_polish_env.py`: `enable_pad_physical_contact=True`일 때만
  - v5 패드 생성 함수 호출 **직후**(v5 코드 자체는 무수정) collider를 켜고 compliant 재질을 바인딩
  - 작업면(Workpiece)을 kinematic rigid body로 스폰 (GPU contact filter가 static collider를 지원 안 해서 패드↔작업면 분리힘이 0으로 나오는 문제 해결)
  - ContactSensor에 `filter_prim_paths_expr`로 작업면 지정, `F_normal=|F·n|` 법선 투영, 1차 저역필터, NaN fault 판정
  - 폐루프: 어드미턴스 피드백을 센서 필터힘으로 교체(`contact.py`의 `control_force_override` 신규 인자, 기본 `None`이면 기존과 동일)
- `learning/rl/env/contact.py`: `step()`에 `control_force_override` 선택 인자만 추가(하위호환). 어드미턴스 게인(`admittance_damping`/`admittance_max_vel`)을 모듈 상수가 아니라 인스턴스 속성으로 빼서, 물리 모드에서만 값을 올릴 수 있게 함(재현 시험 보호).

### 2.2 튜닝 값 (PT-DESIGN, 실측 기반)

1차 시도(strength 5000/damping 100, admittance D=50/vmax=0.02)는 실효 접촉강성이 너무 높아 지연-포화 한계순환(정상상태 진동 std 2~3N, 10N 목표는 hard-limit 리셋 루프)이 났다. 2차로 안정화:

```
pad_compliant_stiffness_n_m = 2000.0
pad_compliant_damping_n_s_m = 200.0
sensor_filter_alpha = 0.5
physical_admittance_damping = 150.0
physical_admittance_max_vel_m_s = 0.012
```

### 2.3 검증 게이트 — 전부 PASS

스크립트: `learning/rl/tests/test_pad_contact_force.py` (phase: free/static/track/safety/parallel/quality)
결과: `learning/rl/outputs/contact_validation/` (`README.md`에 요약, phase별 JSON + CSV)

- **free**: 자유공간 raw force max 0.0000N
- **static 3·5·8·10N**: 정상상태 오차 ~0N, overshoot ≤0.23N, 안정화 ~5s, 관통 0.7~1.7mm
- **track**: BO recipe(5.778N) 이동 중 평균오차 0.034N — 단, 이 시점엔 **단일 짧은 구간**만 지나서 §5의 줄바꿈 스파이크를 못 잡아냈다(중요, 아래 참고)
- **safety**: 20N 과도명령→hard limit 종료, reset 직후 spike 없음, 패치 밖 즉시 무가공, NaN 가드
- **parallel**: 1·4·8·16 env 전부 PASS, NaN 없음
- **quality**: 품질모델이 실제로 센서힘(≈5.79N)을 소비, 모델힘(≈6.7N)과 다름 확인 — 비접촉 시 제거·발열 0

---

## 3. C — RobotPolishEnv(물리모드) 기준 새 BC/PPO

### 3.1 환경 통일

`bootstrap_bc.py`, `train_ppo.py`, `eval_ppo.py`, `eval_conditions.py` 4개 스크립트 전부 `PolishEnv`→`RobotPolishEnv`로 교체, `--contact_mode {physical,model}`(기본 physical) 인자 통일 추가.

### 3.2 새 체크포인트 (기존 덮어쓰지 않음, 전부 `learning/rl/robot/` 아래)

- BC: `learning/rl/robot/champion/model_bc_robot.pt` — 16env×5500step 물리모드 수집(88000샘플, 접촉 96.4%/자유공간 3.6%/에피소드종료 16회/센서fault 0), 모방검증 통과
- PPO: `learning/rl/robot/logs/2026-08-29_13-14-15/` — BC에서 resume, critic 워밍업 50iter + 본학습 750iter. 중간 체크포인트 다수 저장됨(model_0~model_final.pt)

### 3.3 보상 수정 (물리모드 전용, RobotPolishEnv 오버라이드로만 추가)

- `w_force_overshoot`: 명령 초과분만 벌하는 비대칭 overshoot 페널티
- `w_unstable_contact`: control step 내 힘 표준편차 초과분 페널티
- 비접촉 가공 오류 가드(계수+경고, `Errors/no_contact_removal` 로그)

### 3.4 평가 결과 — 6개 조건 비교 (1 pass 기준)

CSV: `learning/rl/robot/results/eval_conditions_robot.csv`, 요약: `learning/rl/robot/results/README.md`
시각화 아티팩트(발행됨): https://claude.ai/code/artifact/41ff84d4-b3bf-40e0-b79b-3af1b7ccc80f

| 조건 | GU(전→후) | Rz 통과 | 비고 |
|---|---|---|---|
| baseline | 53.7→62.1 | 5/8 | |
| bc_robot | 53.7→62.5 | 4/8 | Ra 0/8, 최고온도 46.9°C — 위험 |
| ppo_it400 | 53.7→62.9 | 6/8 | |
| **ppo_it700 (챔피언)** | **53.7→64.5** | **6/8** | GU개선폭 최대, 온도·열손상 양호 |
| ppo_final | 53.7→63.5 | 5/8 | |
| legacy_ppo(구환경, 참고) | 53.7→63.2 | 5/8 | |

전 조건 GU≥70 미달(동시통과 0/8)은 **정상** — 1 pass만으로는 목표 도달 못 하는 게 당연하고, 그래서 D단계(재폴리싱)가 필요하다는 근거 데이터다.

**챔피언: `learning/rl/robot/logs/2026-08-29_13-14-15/model_700.pt`** (reward 아니라 GU/Rz/온도·열손상 종합 기준으로 선정. model_final이 아니라 중간 체크포인트를 뽑은 이유는 학습 중 GU가 61→68→57→64→68로 진동해서 — 기존에 기록된 "PPO reward↑ 인데 최종품질과 안 맞는" 패턴이 이번에도 재현됨.)

---

## 4. D — 재폴리싱 상태기계 (구현 완료, 최종 대규모 검증은 미완)

### 4.1 설계

`learning/rl/env/robot_polish_env.py`에 `_repolish_mode`(기본 False, 켜기 전엔 완전히 기존과 동일) 추가. 켜면 `_get_dones()`가:

1. 경로 완주 시 `_repolish_decide()` 호출 → 품질(GU≥70·Ra≤0.20·Rz≤2.0) + 안전(clearcoat≥35μm·온도<80°C·힘 hard-violation 없음·접촉불안정 없음) 동시 판정
2. 통과 → `success` 종료
3. 미달+안전 → **냉각(무가공 상태로 quality_dt 간격 반복 진행) 후 같은 표면에서 다음 pass** — 표면(clearcoat/scratch/온도)은 그대로 이어받고, 로봇 위치·접촉기 내부상태만 리셋
4. 안전위반/최대pass초과(기본 6)/개선없음 → 실패 종료(사유 분리 기록)
5. 다음 pass 목표힘은 **미달분·안전예산 기반으로 재산정**한다 — "정해진 스텝만큼 무조건 올리기"는 clearcoat을 안전선 아래로 뚫을 위험이 있어 채택하지 않았다. 대신 직전 pass의 실측 (힘당 clearcoat 감소율)로 다음 힘을 역산하고, 남은 안전예산을 넘지 않는 선에서만 올리며, 그래도 안전예산 안에서 목표에 못 미칠 것 같으면 `fail_infeasible`로 정직하게 실패 처리한다.

외부 실행 스크립트: `learning/rl/repolish_eval.py` — 챔피언으로 여러 표면을 반복 재폴리싱시키고 성공률·평균pass수·실패원인 통계를 CSV로 남긴다.

### 4.2 스모크 테스트에서 발견·수정한 버그 2건

1. **종료 판정 순서 버그**: `_get_dones`가 "안전위반(died)"을 "경로완주(done_path)"보다 먼저 체크해서, 두 조건이 같은 tick에 겹치면 정상 완주를 즉시실패로 오분류했다. → 경로완주를 먼저 판정하도록 순서를 바꿈.
2. **안전성 판정 누락**: `_repolish_decide`의 `safety_ok`가 접촉불안정만 보고 힘초과/과열은 아예 안 봤다. → force_hard_violated/thermal_hard_violated 포함하도록 수정.

두 버그 다 기계적 결함이라 수정 자체는 이견 없이 진행했다.

### 4.3 재폴리싱 스모크 중 발견한 더 근본적인 문제 (§5에서 해결)

버그 수정 후에도 챔피언 재폴리싱이 8/8 `fail_force_overload`로 나왔다. 원인을 실측 로그로 추적한 결과, **정책과 무관하게(action=0에서도 재현) 레스터 경로의 줄바꿈 지점에서 접촉력이 5.77N→18N+ 로 튀는 현상**을 발견했다. 이건 재폴리싱 로직 버그가 아니라 경로추종/제어 쪽 문제였다 — §5 참고.

### 4.4 D단계 남은 일

- §5 수정 이후 재폴리싱 스모크를 아직 재실행 안 함 (**다음 채팅에서 제일 먼저 할 일**)
- 스모크(4env×2seq 수준) 통과 확인 후, 규모를 키운 정식 검증(예: 16env, seq 여러 개, 표면 다양화)
- 성공률·평균 pass 수·실패 사유 분포를 근거로 `repolish_force_gain_um`/`repolish_cc_safety_margin_um` 등 PT-DESIGN 파라미터 튜닝 여지 있음

---

## 5. 레스터 줄바꿈 힘 스파이크 — 근본원인 발견 및 수정 (완료·검증됨)

### 5.1 증상과 진단

`env.log_raw_steps=True`로 물리모드 1 pass를 action=0(정책 없이, BO 기준레시피만)으로 끝까지 돌려 확인:

```
t=121.09  raw=5.77N  (정상)
t=121.14  raw=9.62N   ← 급증 시작
t=121.19  raw=15.80N, filtered=18.22N   ← 0.1초 만에 5.77→18.22N
```

발생 지점(arc≈0.72m, path_len=1.44m)이 정확히 **BO recipe의 n_passes(=2) 경계 = 레스터 마지막 줄→첫 줄 복귀 지점**과 일치했다.

### 5.2 원인

`polish_env.py`의 `_pos_at_arc()`(공용, 수정 안 함)는 줄 끝→다음 줄 시작을 **순간이동**으로 반환한다. 정상 줄 전환은 step-over 거리(~20mm) 점프, n_passes 경계 복귀는 그보다 훨씬 큰 점프다. 이 좌표를 IK 목표에 그대로 먹이면 한 control step 안에 그만큼 순간 이동을 요구하게 되고, 실측으로 확인한 대로 접촉력이 크게 튄다.

### 5.3 수정 (RobotPolishEnv 안에서만, `_pos_at_arc`·`path_executor.py`는 무수정)

`_apply_action()`에 목표 (u,v) 이동을 `cfg.line_transition_speed_m_s`(기본 0.05 m/s)로 제한하는 램프 리미터 추가. 평소 이송 중엔 한 step 이동량이 이 한도보다 훨씬 작아 기존 동작과 동일하고, 줄바꿈처럼 큰 점프가 필요할 때만 여러 step에 걸쳐 부드럽게 이동한다. `_prev_uv`(env별 마지막 목표 좌표) 신규 상태를 `__init__`/`_reset_idx`/`_soft_reset_path`에서 관리.

### 5.4 검증 결과

동일 진단(action=0, 1 pass 완주, 물리모드)을 수정 후 재실행:

| | 수정 전 | 수정 후 |
|---|---|---|
| 최대 raw force | 18.22N | **7.07N** |
| 10N 초과 횟수 | 있음 | **0회** |
| 14N 초과 횟수 | 있음(hard-limit 트립) | **0회** |
| 종료 사유 | force_hard_violated | **정상 완주(done_path)** |

**회귀 확인**: `test_robot_polish_env_runtime.py`(기본 모드) 재실행 결과 기존과 동일(force_model 5.782N, gap 1.5mm) — 회귀 없음.

---

## 6. 중요 캐비어트 — 다음 채팅에서 반드시 먼저 판단할 것

**§3의 챔피언(`model_700.pt`)과 새 BC(`model_bc_robot.pt`), 그리고 §3.4의 6조건 비교 평가는 전부 §5의 줄바꿈 스무딩 수정 *이전* 환경에서 학습·평가됐다.**

줄바꿈 스무딩으로 접촉 동역학이 바뀌었으므로(특히 줄바꿈 지점에서 힘이 안 튀게 됨):

- 기존 평가 결과(§3.4 표, 시각화 아티팩트)는 **스무딩 전 환경 기준**이라는 걸 명시하고 봐야 한다.
- 챔피언 정책이 스무딩된 환경에서도 여전히 최선인지는 검증 안 됐다 — 스무딩 전 환경은 줄바꿈마다 힘이 안 튀도록 정책이 암묵적으로 보정해야 했을 수도 있고, 스무딩 후엔 그 보정이 불필요/부적합해졌을 수 있다.
- **권장**: 최소한 챔피언 정책을 스무딩된 환경에서 재평가(가능하면 BC/PPO 재학습)해서, §3.4 표를 스무딩 후 기준으로 다시 뽑는 게 좋다. 이건 사용자 승인 없이 진행하지 않았다.

---

## 7. 파일 목록

### 7.1 이번 세션에 수정한 파일

- `learning/rl/env/robot_polish_env_cfg.py`
- `learning/rl/env/robot_polish_env.py`
- `learning/rl/env/contact.py`
- `learning/rl/bootstrap_bc.py`
- `learning/rl/train_ppo.py`
- `learning/rl/eval_ppo.py`
- `learning/rl/eval_conditions.py`

### 7.2 이번 세션에 새로 만든 파일

- `learning/rl/tests/test_pad_contact_force.py` — 접촉력 검증 게이트(6 phase)
- `learning/rl/repolish_eval.py` — 재폴리싱 외부 드라이버
- `learning/rl/outputs/contact_validation/` — 검증 결과(JSON/CSV/README)
- `learning/rl/robot/champion/model_bc_robot.pt` — 새 BC
- `learning/rl/robot/logs/2026-08-29_13-14-15/` — 새 PPO 체크포인트들(챔피언 `model_700.pt`)
- `learning/rl/robot/results/eval_conditions_robot.csv`, `README.md` — 6조건 비교
- `learning/rl/robot/results/repolish_smoke_champion2.csv` — 재폴리싱 스모크(버그수정 후, 스무딩 전) 결과

### 7.3 절대 수정하지 않은 것

- `scripts/polishing_v5.py`, `scripts/polishing_v5_modules/` 전체
- `learning/rl/env/polish_env.py`, `polish_env_cfg.py` (RobotPolishEnv가 상속만 함)
- `learning/rl/champion/`, `learning/rl/thermal/` 기존 체크포인트
- `learning/polytwin/path_executor.py`, `polishing_model.py` (레스터 웨이포인트 생성·품질모델 원본)

---

## 8. 다음 채팅에서 시작할 순서

1. 이 문서 + 선행 인수인계서 전체 읽기
2. **§6 캐비어트 판단**: 스무딩 후 환경으로 챔피언 재평가할지(또는 BC/PPO 재학습까지 할지) 사용자와 확인
3. 재폴리싱 스모크(`repolish_eval.py`)를 스무딩 수정 반영된 상태로 재실행 — 8/8 fail_force_overload가 사라지고 실제 success/fail_max_passes/fail_infeasible 등 의미있는 분포가 나오는지 확인
4. 통과하면 규모를 키워 정식 재폴리싱 평가 → BMW 대표 6구역 연동(선행 인수인계서 20장)으로 진행

---

## 9. 한 줄 상태

> PhysX 접촉력 검증(A/B단계)과 RobotPolishEnv 기준 새 BC/PPO(C단계)는 완료됐고, 재폴리싱 상태기계(D단계)도 구현하고 종료판정 버그 2건을 고쳤다. 스모크 테스트 중 "레스터 줄바꿈 지점에서 힘이 18N까지 튀는" 근본 원인을 발견해 램프 리미터로 수정·검증(18.22N→7.07N, 14N 초과 0회)했다. 다만 §3의 챔피언·BC·평가결과는 이 수정 이전 환경 기준이라 재검증이 필요하고, 재폴리싱 스모크도 수정 반영 후 아직 재실행하지 않았다 — 다음 채팅의 첫 작업이다.
