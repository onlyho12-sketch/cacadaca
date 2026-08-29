# PhysX 패드 접촉력 검증 결과 (인수인계서 17장 / 25장)

작성일: 2026-08-29
실행: `/home/rokey/isaacsim-6.0.1/python.sh learning/rl/tests/test_pad_contact_force.py --phase <phase> --headless`

## 결과 요약 — 전 phase 통과

| Phase | 내용 | 결과 |
| --- | --- | --- |
| free | 자유공간 0 N + 비접촉 무가공·무발열 | 6/6 PASS (max raw 0.0000 N) |
| static | 정적 3·5·8·10 N 추종 | 8/8 PASS (오차 ~0 N, overshoot ≤0.23 N, 안정화 ~5 s, 관통 0.7–1.7 mm) |
| track | 이동 경로(BO recipe 5.778 N) 추종 | 5/5 PASS (평균오차 0.034 N, p95 0.031 N) |
| safety | 20 N 과도명령/reset spike/패치 밖 게이트/NaN 가드 | 5/5 PASS |
| parallel | 1·4·8·16 env 병렬 | 각 PASS (worst err_mean 0.062 N, NaN 없음) |
| quality | 검증힘 → 제거·온도 모델 연결(17.3) | 5/5 PASS |

## 이 검증이 말하는 것

- `force_sensor_n` 은 이제 PhysX ContactSensor 가 측정한 실제 시뮬레이션 접촉력이다
  (패드 collider ON + compliant 재질, `enable_pad_physical_contact=True` 일 때).
- 폐루프: 어드미턴스 피드백이 센서 필터힘 → 3/5/8/10 N 목표를 정상상태 오차 ~0 N 로 추종.
- 센서힘(목표 추종)과 모델힘(가상 스프링 포화값 ~6.6–6.9 N)이 로그에서 명확히 분리된다.
- quality phase: 품질 모델이 소비한 힘 5.785 N ≈ 센서 5.788 N ≠ 모델 6.704 N —
  제거량·온도는 검증된 센서힘으로 계산되며, 비접촉/패치 밖에서는 제거·발열 0.
- net force 와 패드↔작업면 분리힘(force_matrix_w) 일치 → 허위 접촉 없음.
  (GPU contact filter 지원을 위해 물리 모드에서 Workpiece 를 kinematic rigid body 로 스폰)

## 이 검증이 말하지 않는 것 (정직성 경계)

- 실제 로봇/실제 패드의 계측 접촉력이 아니다 — PhysX 시뮬레이션 내부 값이다.
- compliant 강성 2000 N/m 등 접촉 파라미터는 PT-DESIGN(수치 안정성 기준 튜닝)이며
  실측 패드 물성이 아니다. 실효 접촉강성은 요청값보다 높게 나타났다(관통 실측 기준 ~5–6e3 N/m).
- track 의 순간 최대 오차 12.8 N 스파이크(단일 스텝, 라인 전환 과도)는 존재한다.
  p95 0.031 N, hard-limit 미달, 리셋 없음 — 다만 기록해 둔다.
- 기본값은 여전히 `enable_pad_physical_contact=False` — 학습 스크립트가 물리 모드를
  쓰려면 명시적으로 켜야 한다 (RL 재학습 단계에서 결정).

## 튜닝 기록 (Phase B 1차 실측 → 2차 안정화)

1차(실패): stiffness 5000, damping 100, filter α 0.25, 어드미턴스 D=50/vmax=0.02
→ 실효 강성 ~1e4 N/m 에서 지연-포화 한계순환 (std 2.2–2.8 N, overshoot ~5 N,
10 N 은 hard-limit(14 N) 리셋 루프 9회).

2차(통과): stiffness 2000, damping 200, filter α 0.5, D=150, vmax 0.012
(`robot_polish_env_cfg.py` 물리 모드 필드 / contact.py 모듈 상수는 불변).

## 파일

- `<phase>.json` / `parallel_env{1,4,8,16}.json` — 판정 임계값·지표·PASS/FAIL
- `*_env0.csv` — env0 시계열: force_cmd / force_sensor_raw / force_sensor_filtered /
  force_model / force_used / sensor_matrix / pad_gap / in_patch / sensor_fault
- 센서 fault(NaN) 시 모델힘 fallback 은 `sensor_fault`·`fallback_steps` 로 로그·결과에
  명시된다. 이번 검증 전 구간에서 fault 0회.

## 회귀 확인

- `test_contact_replay.py` 7/9 — 인수인계서 15.3 의 기존 상태와 동일
  (실패 2건은 v5 force_log CSV 가 헤더만 남은 기지 문제, 이번 변경과 무관)
- `test_robot_polish_env_runtime.py` (기본 모드, 충돌 OFF) PASS —
  모델힘 5.78 N, gap 1.5 mm, 센서 0 N — 인수인계서 15.1 기록과 동일.
