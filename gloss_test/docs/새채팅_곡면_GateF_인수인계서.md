# 새 채팅용 곡면 Gate F 전체 인수인계서

최종 갱신 기준: 2026-09-01 KST  
저장소: `/home/rokey/cacadaca`  
현재 단계: **F1~F9 완료, F9 최종 FAIL, F10 설계 전 승인 대기**

이 문서는 곡면 Gate F에서 실제로 수행한 작업, 시행착오, 동결된 판단, 보호 대상,
공식 산출물과 다음 실행 순서를 새 채팅에 넘기기 위한 최신 기준 문서다. 이전에 이
문서에 적혀 있던 “F1 이식 전” 계획은 더 이상 현재 상태가 아니다.

새 담당자는 다음 세 문서를 처음부터 끝까지 읽고 실제 파일과 대조해야 한다.

- `gloss_test/docs/새채팅_전체작업_인수인계서.md`
- `gloss_test/docs/새채팅_평면관측확장_경로최적화_인수인계서.md`
- 이 문서

## 1. 사용자 작업 규칙

1. 사용자가 시킨 범위만 수행한다.
2. 작업 전에 무엇을 할지, 어떤 파일과 실행을 건드릴지, 그 단계부터 최종 완료까지
   예상시간이 얼마인지 구체적으로 설명한다.
3. 사용자가 `ㄱㄱ`, `진행`, `해`처럼 승인하기 전에는 코드 수정, PhysX 실행,
   데이터 수집, BC/PPO 학습을 시작하지 않는다.
4. 각 단계의 기준은 실행 전에 JSON 등으로 동결한다. 결과를 본 뒤 기준이나 후보를
   유리하게 바꾸지 않는다.
5. 각 단계는 신규 출력 경로와 CSV/JSON/README/checksum을 사용하고 결과를 설명한
   뒤 다음 단계 승인을 별도로 기다린다.
6. 기존 checkpoint, 모델, CSV, JSON, README, PPT용 결과와 현재 dirty worktree를
   삭제·정리·덮어쓰기 하지 않는다.
7. 중단이나 문맥 전환 뒤에는 프로세스와 산출물 완결성을 먼저 확인한다. 끝난 실행을
   추측으로 반복하지 않는다.

## 2. 절대 보호 대상과 현재 확인 해시

다음 공용 파일과 모델은 Gate F 과정에서 수정하지 않았다. 2026-09-01에 다시 계산한
SHA-256은 아래와 같다.

| 대상 | SHA-256 |
|---|---|
| `scripts/polishing_v5.py` | `234bb26fbdec318902b67aeb36e0cc1518a22c508f2a3fc4000a6ac8bcf0ac55` |
| `learning/rl/env/polish_env.py` | `79aa3c57ef4fc250ce157463d64adc80d67008e125c9ecd4cf07461f838c9630` |
| `learning/rl/env/polish_env_cfg.py` | `8711d99116ae3c21534b7f5e206288466268beb8c40a872d1912dce11e9ae581` |
| 기존 BC champion `learning/rl/robot/champion/model_bc_robot_tesla12p7.pt` | `5fa1a65a60a90ffca5116a8d75d20a34749ebd69d71bb6f27287e0957f49b14a` |
| 평면 E7 release `model_gate_e7_seed1_it211.pt` | `c733763bfebd7fcead01cf2574bbd1621c4660b9347143cccb35e0e83449d052` |

추가 보호 범위:

- `scripts/polishing_v5_modules/` 전체
- `learning/rl/robot/results/`의 모든 기존 산출물
- `learning/rl/robot/releases/` 전체
- 모든 기존 checkpoint, 학습 데이터, CSV, JSON, README, PPT용 자료
- 현재 staged/unstaged/untracked 변경 전체

작업트리는 원래부터 매우 크고 dirty하다. 임의의 `reset`, `checkout`, 삭제, 정리,
재포맷을 하지 않는다. Gate F 결과가 untracked라는 이유로 지우거나 다시 만들지 않는다.

## 3. 한눈에 보는 현재 결론

- 평면 Gate E7 release는 그대로 기준 checkpoint다. 재학습하거나 교체하지 않았다.
- F1~F5에서 곡면 높이·법선, mesh, 국소 정상력, 곡면 접촉과 pad 법선 정렬을 신규
  Gate F 파일로 격리 구현하고 검증했다.
- F6에서 E7 평면 release를 곡면에 zero-shot 적용했다. 64/64 실행은 완결됐지만
  cylinder/freeform의 힘 과부하 때문에 **zero-shot 충분성은 FAIL**이었다.
- F7에서 법선/곡률 관측 확장을 작은 BC ablation으로 비교했다. 확장 관측이 base14를
  이기지 못해 **관측 확장 NO-GO**, base14 유지로 결정했다.
- F8에서 곡면 전용 힘 action 제약을 작은 파일럿으로 비교했다. `static_cap`, 즉 곡면에서
  force action 상단을 `+0.50`으로 제한하는 후보가 5건의 과부하를 0건으로 줄여
  **소규모 안전 gate PASS**했다.
- F9 다중 seed paired validation 256 sequences를 완료했다. `static_cap(+0.50)`은
  freeform 과부하는 제거했지만 cylinder 과부하를 크게 늘려 최종 **FAIL**했다.
- 따라서 `static_cap(+0.50)`은 integration candidate가 아니며 production/public env나
  `polishing_v5`에 이식하면 안 된다. PPO도 하지 않았고 champion/release도 수정하지 않았다.
- 다음은 F9 실패 원인분석을 시작점으로 하는 F10이다. 차량 point cloud의 법선·주곡률,
  GU 기반 품질판정, 1차 고속 연속 패스와 2차 선택 재작업, 곡률 대응 안전제어/RL을
  격리 구현하고 검증한 뒤에만 `polishing_v5` 이식을 검토한다.
- 마지막 회귀 시험은 **66 passed**였다.
- F9 종료 확인 당시 실행 중인 Gate F 평가나 BC/PPO 학습 프로세스는 없었다. 새 담당자는
  현재 프로세스를 다시 읽기 전용으로 확인하되 사용자가 돌리는 백그라운드 프로세스를
  임의로 중단하지 않는다.

## 4. 기준 평면 release

공식 기준 경로:

`learning/rl/robot/releases/gate_e7_seed1_it211_20260831_201801/`

- training seed/iteration: `20260831 / 211`
- observation/action: `14 / 2`
- checkpoint: `model_gate_e7_seed1_it211.pt`
- 안전: `64/64`, sensor fault step `0`
- whole-surface 품질: `48/64`
- four-target cellwise all-4 면적: `94.02828125%`
- 기존 BC champion 대비 all-4: `+0.2440625%p`
- 평균 control step: `4040.734375`, 기존 대비 `+15.5145304948%`
- same-seed 반복 CSV 일치, CPU strict load/inference smoke PASS

Gate F에서는 이 checkpoint를 control policy로 명시적으로 불러왔을 뿐 release 패키지를
고치거나 새 모델로 승격하지 않았다. 평면은 재개발 대상이 아니라 parity와
non-regression 기준선이다.

## 5. F1 — 곡면 기능 격리 이식 완료

공식 결과: `learning/rl/robot/results/gate_f1_port_20260831_233329/`

새로 만든 핵심 파일:

- `learning/rl/gate_f_curved_geometry.py`
- `learning/rl/env/gate_f_curved_polish_env.py`
- `learning/rl/env/gate_f_curved_polish_env_cfg.py`
- `learning/rl/tests/test_gate_f_curved_geometry.py`
- `learning/rl/gate_f_curved_physx_smoke.py`

선택적으로 이식한 범위는 flat/cylinder/sphere/freeform의 height·unit normal,
상향 winding의 결정적 trimesh, 국소 법선에 대한 힘 투영, CPU geometry 시험이다.
외부 공용 env, checkpoint, 결과나 v5 pad-contact 구현은 통째로 복사하지 않았다.

- 문서 분석 당시 외부 commit: `1d020643687e516dbf56453caa8db88eab554adb`
- F1 실행 시 고정 commit: `cb9ee7e7beca0c40fb6669db659f90681bfcca1c`
- 사이 4개 commit은 선택한 곡면 source를 바꾸지 않았다. source별 해시는
  `provenance.json`에 기록했다.

결과는 CPU geometry **17/17 PASS**, static syntax **5/5 PASS**였다. F1에서는 PhysX와
학습을 실행하지 않았고 기존 파일 수정도 없었다.

## 6. F2 — flat parity 완료

공식 결과: `learning/rl/robot/results/gate_f2_flat_parity_20260831_234000/`

`learning/rl/tests/test_gate_f_flat_parity.py`로 flat/radius→∞의 높이, 법선, mesh와
힘 정의가 기존 평면 기준과 맞는지 확인했다. 평면 E7 전체를 재실행하거나 재학습한
단계가 아니다. 결과와 provenance/checksum은 해당 디렉터리에 보존돼 있다.

## 7. F3 — geometry/coverage screen 완료

공식 결과: `learning/rl/robot/results/gate_f3_geometry_coverage_20260831_235459/`

관련 코드:

- `learning/rl/gate_f_curved_coverage_screen.py`
- `learning/rl/tests/test_gate_f_curved_coverage_screen.py`

cylinder/sphere/freeform 후보의 geometry, 법선 변화, raster coverage와 tile 결과를
계산적으로 screening한 뒤 F4 후보를 줄였다. `..._235207/`은 superseded 표시가 있는
초기 산출물이고 공식 기준은 `..._235459/`이다. BC/PPO 학습은 없었다.

## 8. F4 — 곡면 PhysX 접촉 진단과 remediation 완료

- 사전 계획: `learning/rl/robot/results/gate_f4_physx_plan_20260831_150801/`
- 최종 결과: `learning/rl/robot/results/gate_f4_remediation_summary_20260831_155200/`

관련 코드는 `gate_f_curved_physx_diagnostic.py`, `gate_f_curved_physx_summarize.py`,
`tests/test_gate_f_curved_physx_summarize.py`다. 첫 진단의 flat/cylinder/sphere 결과와
freeform 불완전 시도는 그대로 보존했다. 관찰된 수직 추종 오차를 근거로 공용 env가
아닌 Gate-F 전용 cfg에 `+0.5 mm` 수직 tracking compensation을 적용해 remediation했다.
flat/cylinder/sphere/freeform 4개 후보 모두 최종 PASS했고 sensor fault·hard-force·
비접촉 제거 등의 동결 조건을 통과했다. 학습은 없었다.

## 9. F5 — pad 국소 법선 정렬 완료

- 사전 기준: `learning/rl/robot/results/gate_f5_alignment_plan_20260831_160000/`
- 최종 결과: `learning/rl/robot/results/gate_f5r_alignment_summary_20260831_172200/`

구현·진단에서 확정한 사항:

- IsaacLab의 올바른 `xyzw` target quaternion composition을 Gate F 전용 env에 적용
- flat 기준 quaternion `[1, 0, 0, 0]`
- 실제 pad local `+Y` 축을 측정 방향으로 사용
- 위치 translation을 target orientation과 결합
- curved에만 IK steady bias `(-0.35°, -0.03°)` 적용, flat 제외
- freeform normal alignment steady p95 약 `0.663° → 0.315°`

초기 quaternion/축/pose-coupling probe와 incomplete/superseded 결과는 감사 기록으로
보존했다. 최종 F5r은 4 geometry pair 전부 PASS, 당시 전체 시험 **44 passed**였다.
학습은 하지 않았다.

## 10. F6 — 평면 release zero-shot 곡면 평가 완료

- 사전 기준: `learning/rl/robot/results/gate_f6_zero_shot_plan_20260831_172500/acceptance_criteria.json`
- 최종 결과: `learning/rl/robot/results/gate_f6_zero_shot_summary_20260831_183000/`
- 코드: `gate_f6_zero_shot_eval.py`, `gate_f6_zero_shot_summarize.py`,
  `tests/test_gate_f6_zero_shot_summarize.py`

평가 규모는 4 geometry × 4 profile × 2 seed × 2 direction으로 **64 sequences,
1,600 tiles**였다. 모든 실행과 집계가 끝나 completion은 PASS지만 zero-shot
sufficiency는 FAIL이다.

| geometry | safety | quality | overload | mean all-4 area | mean control steps |
|---|---:|---:|---:|---:|---:|
| flat | 16/16 | 12/16 | 0 | 91.793125% | 4153.6250 |
| cylinder | 11/16 | 8/16 | 5 | 91.603750% | 3296.8125 |
| sphere | 16/16 | 12/16 | 0 | 91.786250% | 4150.0000 |
| freeform | 11/16 | 8/16 | 5 | 91.683750% | 3359.3125 |

sensor fault는 전부 0이었다. cylinder/freeform의 짧은 시간은 빨라진 것이 아니라
힘 과부하 조기 종료가 섞인 결과라 censoring해서 해석해야 한다. mean force action은
flat `0.67348`, cylinder `0.74679`, freeform `0.76110`으로 높아져 action/observation
일반화가 병목이라는 근거가 됐다. sphere는 동결 paired check를 통과했지만
cylinder/freeform은 실패했다.

## 11. F7 — 관측 ablation과 작은 BC pilot 완료

- 사전 기준: `learning/rl/robot/results/gate_f7_ablation_plan_20260901_033640/`
- 최종 결과: `learning/rl/robot/results/gate_f7_summary_20260901_040000/`

신규 파일은 `gate_f7_observation.py`, 두 `gate_f7_curved_observation_env*`,
`gate_f7_observation_smoke.py`, `gate_f7_bc_collect.py`, `gate_f7_bc_train.py`,
`gate_f7_summarize.py`, `tests/test_gate_f7_observation.py`다.

비교 arm은 기존 `base14`, local normal xyz를 더한 `normal17`, 정규화한 local graph
Hessian `huu, huv, hvv`까지 더한 `normal_curvature20`이었다. 4 geometry smoke는 모두
finite, sensor fault 0이었다. 총 **19,200 samples**를 수집했고 teacher action 변경률은
flat 0%, cylinder 97.0208%, sphere 98.8958%, freeform 92.2708%였다.

같은 data/split/seed/epoch의 test force MAE:

- base14: `0.014785069`
- normal17: `0.015087314`, base14보다 `+2.044%` 악화
- normal_curvature20: 약 `0.015251658`

completion은 PASS지만 observation extension gate는 NO-GO이며 `base14`를 선택했다.
사전 동결한 offline entry criterion 실패로 candidate PhysX 평가는 생략했다. 작은 BC
학습만 했고 PPO, promotion, release 수정은 없었다. 당시 회귀 시험은 **55 passed**였다.

## 12. F8 — 작은 곡면 힘 안전 제약 pilot 완료

- 사전 기준: `learning/rl/robot/results/gate_f8_force_safety_plan_20260901_041500/acceptance_criteria.json`
- 최종 결과: `learning/rl/robot/results/gate_f8_summary_20260901_063000/`

신규 파일은 `gate_f8_force_safety.py`, `gate_f8_force_safety_eval.py`,
`gate_f8_force_safety_summarize.py`, `gate_f8_summarize.py`와 두 관련 test다.

공용 env에는 이미 overshoot/instability reward penalty가 있었지만 F6 실패는 높은 force
action 뒤 조기 hard termination으로 나타났다. 지연 reward PPO 전에 결정적 action
shield가 문제를 직접 막는지 작은 범위에서 비교했다.

- `control`: E7 release action 그대로
- `static_cap`: curved에서만 force action 상단 `+0.50`; flat exact passthrough
- `predictive_shield`: static cap + base14 force/delta 2-step prediction, 9 N부터 감소,
  11 N에서 0, 최소 action `-0.50`; flat exact passthrough

첫 `..._042000/` smoke는 제외 README를 남겼다. 공식 smoke `..._042100/`은 4 env ×
24 step, finite, sensor fault 0이었다. Stage 1은 세 arm의 same direction **48
sequences**였다. static과 predictive 모두 과부하를 제거했고 더 단순하고 raw max
force가 낮은 `static_cap`을 Stage 2로 고정했다. Stage 2는 control/static의 cross
direction **32 sequences**, 공식 합계는 **80 sequences**다.

| arm/geometry | safety | quality | overload | raw max N | mean all-4 area |
|---|---:|---:|---:|---:|---:|
| control/flat | 8/8 | 6/8 | 0 | 9.4633 | 95.3075% |
| control/cylinder | 3/8 | 2/8 | 5 | 14.0412 | 94.91625% |
| control/sphere | 8/8 | 6/8 | 0 | 11.8337 | 95.3075% |
| control/freeform | 8/8 | 6/8 | 0 | 12.4116 | 95.3000% |
| static/flat | 8/8 | 6/8 | 0 | 9.4633 | 95.3075% |
| static/cylinder | 8/8 | 6/8 | 0 | 13.3809 | 95.1700% |
| static/sphere | 8/8 | 6/8 | 0 | 10.5371 | 95.1725% |
| static/freeform | 8/8 | 6/8 | 0 | 12.3353 | 95.1700% |

곡면 safety는 control `11/16 → static 16/16`, overload는 `5 → 0`이었다. candidate-
control curved all-4 delta는 `+0.061875%p`, flat safety/quality 비열화 없음, sensor
fault 0으로 작은 safety gate는 PASS했다.

문맥 전환 뒤 Stage 2가 `..._061500/`으로 한 번 더 실행됐다. 삭제하지 않는다. 8개
duplicate `sequences.csv` 해시가 공식 `..._051200/`과 각각 완전히 같으며 공식 80
sequence 집계에서는 제외했다. `gate_f8_final_summary_20260901_054300/`은 최종 정리
전 집계이고 canonical은 `gate_f8_summary_20260901_063000/`이다.

F8에서는 학습/PPO가 없었고 champion/release를 수정하지 않았다. production readiness는
`false`다. 최종 회귀 시험은 **66 passed**, checksum도 PASS였다.

## 13. F9 — 다중 seed paired validation 완료, 최종 FAIL

- 공식 결과: `learning/rl/robot/results/gate_f9_summary_20260901_132500/`
- 규모: 64 runs, **256 sequences**, 6,400 tile rows
- seed base: `40000`, `41000`, `42000`, `43000`
- arm: `control`, curved-only `static_cap(+0.50)`
- direction: same/cross
- 학습, PPO, promotion, release 수정: 전부 없음

완결성, flat passthrough, flat safety/quality, 곡면 all-4 면적 저하 한계, sensor fault는
통과했다. 그러나 candidate force overload가 반드시 0이어야 한다는 핵심 기준을
통과하지 못했다.

| 지표 | control | static cap candidate |
|---|---:|---:|
| cylinder+freeform safety | 52 | 53 |
| 전체 curved safety | 84/96 | 85/96 |
| force overload | 12 | 11 |
| curved all-4 candidate-control | - | `-0.064375%p` |

geometry별 과부하를 보면 고정 cap의 한계가 더 명확하다.

| geometry | control overload | candidate overload | 해석 |
|---|---:|---:|---|
| cylinder | 4 | 11 | 크게 악화 |
| sphere | 0 | 0 | 유지 |
| freeform | 8 | 0 | 개선 |

즉 `+0.50`이라는 하나의 상한은 freeform에는 도움이 됐지만 cylinder 접촉 동역학에는
맞지 않았다. 결과를 본 뒤 cap이나 seed를 바꾸지 않는다는 사전 원칙에 따라 F9는
있는 그대로 **FAIL**, `integration_candidate=false`, `production_readiness=false`다.
F9 결과 디렉터리의 `decision.json`, `README.md`, `acceptance_checks.csv`,
`arm_geometry_summary.csv`, `censoring_summary.csv`, `checksums.sha256`가 판단 기준이다.

F9 실행 중 기록된 network incident와 실행 스크립트의 현재 변경도 감사 기록이므로
임의로 되돌리거나 지우지 않는다. F9의 짧은 control step은 과부하 조기 종료가 섞여
있으므로 속도 향상으로 해석하지 않는다.

## 14. 현재 해석에서 지켜야 할 사항

- F8 PASS는 작은 synthetic pilot 결론이고 F9 확대 검증에서 최종적으로 기각됐다.
- `static_cap(+0.50)`을 public env, release, 실차 코드에 그대로 통합하지 않는다.
- F7의 관측 확장 NO-GO는 당시 `normal17/normal_curvature20` 작은 BC ablation 결과다.
  차량의 정확한 주곡률을 계산하거나 새로운 곡률 조건부 RL을 영구 금지한 결론은 아니다.
- 다만 곡률 feature를 넣는 것 자체가 성능 향상을 보장하지 않으므로 새 관측은 다시
  offline ablation과 작은 PhysX gate를 통과해야 한다.
- 타일/cell은 측정·판정 단위다. 각 cell마다 로봇을 정지·복귀·초기화하거나 독립 episode로
  만들면 차량 한 대 작업시간이 과도하게 늘어난다.
- GU/Ra/Rz/scratch의 cellwise all-4 면적과 whole-surface 품질을 구분한다.
- incomplete/retry/superseded/duplicate 산출물은 감사 기록으로 보존한다.
- 외부 코드는 앞으로 확인하거나 판단 근거로 사용하지 않는다. 로컬 코드, 데이터,
  시뮬레이션 결과만 사용한다.

## 15. 최종 목표 — 차량용 coarse-to-fine polishing

최종 목적은 Gate F 자체가 아니라 검증된 기능을 기존 `polishing_v5`에 안전하게 이식해
실제 차량 곡면을 합리적인 시간 안에 폴리싱하는 것이다. 전체 작업은 다음 흐름으로 만든다.

```text
차량 point cloud + 기존 C/SL/SR 경로
  -> 법선·주곡률·경계/신뢰도 계산
  -> 1차 고속 연속 polishing
  -> 전체 cell의 기존 GU/Ra/Rz/scratch 식 품질검사
  -> all-4 통과 cell은 PASS_LOCKED
  -> 기준 미달 cell을 인접 영역으로 군집화
  -> 필요한 영역만 2차 정밀 polishing
  -> 실제 재폴리싱된 cell만 all-4 재검사
  -> 종료/보호 판정
```

1차는 전체 표면을 빠르고 연속적으로 한 번 통과한다. cell마다 별도 접근/복귀하거나
파라미터를 급변시키지 않는다. 1차 뒤 네 품질항을 모두 통과한 cell은 2차 target에서
제외하고, 실제 표면 상태가 다시 바뀌지 않았다면 재검증하지 않는다. 2차는 all-4 미달
영역만 세밀하게 처리하며, 인접한 실패 cell을 하나의 재작업 영역으로 묶어 영역당 한 번
접근하고 한 번 빠져나온다.

파라미터 계층은 다음처럼 제한한다.

- 차량/job 단위: pad, compound, 기본 RPM, 전체 안전 한계
- panel/큰 영역 단위: 기본 feed, step-over, 목표 힘
- 연속 경로 단위: 곡률 위험에 따른 부드러운 목표 힘 감쇠와 힘 변화율 제한
- 재작업 영역 단위: 2차 feed/step-over/최대 추가 pass
- 빠른 RL 출력: 안전 범위 안의 작은 residual force/feed 보정

## 16. F10-A — F9 실패 원인분석과 시간 기준선

코드를 만들기 전에 F9의 11개 candidate overload를 profile, seed, direction, 접촉 진입/
정상 접촉 phase별로 분해한다. target/executed action, 측정 힘, gap, 법선 변화, 힘 상승률을
대조해 cylinder 악화가 접촉 진입 충격, 정상 방향 변화, cap saturation, controller 지연 중
어디서 발생했는지 확인한다. 결과를 보고 단순히 cap 숫자만 더 낮추지 않는다.

동시에 로컬 `scan_result/car/path_*.npy`의 실제 길이와 segment 수를 계산해 물리 시간
기준선을 만든다. 총시간은 다음을 분리한다.

- 경로 길이 / feed 속도
- turn/transition 시간
- approach/retract 횟수와 시간
- 기존 GU/Ra/Rz/scratch 식 계산과 재작업 mask 생성 시간
- 2차 재작업 비율과 경로 길이
- 로봇별 C/SL/SR 분담과 병렬 wall-clock

첫 3~5시간 진단 후 안전 원인이 설명되지 않거나 예상 물리 시간이 목표에 맞지 않으면
큰 구현·학습 전에 STOP/GO를 사용자에게 보고한다.

## 17. F10-B — 차량 표면 geometry adapter

로컬 실제 입력은 `scan_result/car/points/real_camera_surface_points.ply`, 부분 point cloud,
`scan_result/car/mesh/real_camera_surface_mesh.ply`, 기존 `path_*.npy`,
`rail_config.json`이다. 기존 경로 생성기를 실행하면 기존 path를 지울 수 있으므로 원본에
대해 재생성 함수를 직접 실행하지 않는다.

계획 파일:

- `learning/rl/gate_f10_vehicle_geometry.py`
- `learning/rl/tests/test_gate_f10_vehicle_geometry.py`
- 결과 `learning/rl/robot/results/gate_f10_vehicle_geometry_<timestamp>/`

각 waypoint 또는 평가 cell 주변을 KDTree로 찾고 국소 좌표계에서 robust quadratic
surface를 fit한다. 여기서 다음을 계산한다.

- 일관된 방향의 unit normal과 이전 waypoint 대비 normal angle
- signed principal curvature `k1`, `k2`
- convex/concave/saddle/near-flat 분류
- 곡률반경과 `pad_radius * max(abs(k1), abs(k2))`
- 실제 pad footprint 안의 normal spread
- fit RMS, 이웃 수, point density에 따른 신뢰도
- point-cloud 경계, hole, 경로 이탈, 급격한 곡률 변화 위험

기존 BC dataset의 `lambda_min/sum(lambda)` PCA scalar curvature는 표면 거칠기 지표에
가깝고 signed principal curvature나 곡률반경을 직접 제공하지 않는다. 따라서 그대로
차량 pad-contact 위험도로 쓰지 않는다. low-confidence 또는 경계 위험은 자동으로
속도를 높이지 않고 감속/skip/사람 확인 대상으로 보낸다.

## 18. F10-C — 1차 고속 연속 pass scheduler

계획 파일:

- `learning/rl/gate_f10_pass_scheduler.py`
- `learning/rl/tests/test_gate_f10_pass_scheduler.py`

기존 C/SL/SR 연속 경로와 다중 로봇 분담을 유지한다. 1차 목표는 cell별 최적화가 아니라
전체 표면을 한 번 균일하게 덮는 것이다.

- segment당 한 번 approach/retract
- cell 경계에서 episode reset/정지 금지
- panel 단위 기본 recipe
- waypoint 사이 목표 force/feed를 연속 보간
- 곡률 위험도에 따른 부드러운 force derating
- force command slew/rate limit
- 최초 접촉 구간의 soft-start
- 지나친 경로 중복과 turn 제거
- 로봇 3대의 예상 종료시간이 비슷하도록 경로 부하 균형

고속 feed, 넓은 step-over, RPM, 목표 힘은 임의로 확정하지 않는다. 먼저 실제 path length와
pad 유효 폭을 계산해 후보와 예상 coverage/time을 제시하고 사용자 승인 뒤 동결한다.

## 19. F10-D — 기존 GU 식과 cellwise all-4 품질판정

실제 GU 장비는 연결하지 않는다. 새 센서 모델이나 calibration 절차도 만들지 않고 로컬에
이미 구현된 품질 계산식을 그대로 기준으로 사용한다.

- GU 식: `learning/polytwin/gloss_proxy.py`
  - `LiteratureGlossProxyModel`, `gu_from_relative`
  - `GU = clip(25 + (78 - 25) * relative_gloss, 0, 100)`
  - 코드의 공식 모델명/필드는 `literature_gu_proxy_v1`,
    `predicted_20deg_gu_literature_proxy`다.
- cellwise 네 품질항: `learning/rl/gate_e6_area_quality.py`
  - 기존 2 mm surface cell마다 centered 5×5 cell, 즉 10×10 mm 국소창 사용
  - `GU >= 70`
  - `Ra <= 0.20 µm`
  - `Rz <= 2.0 µm`
  - scratch가 초기보다 개선됐거나 초기 scratch가 `0.05 µm` 미만
  - 위 네 조건의 AND가 `all4_pass`
- clearcoat와 temperature는 all-4에 섞지 않고 별도 안전조건으로 유지한다.

이 값은 프로젝트가 사용하는 문헌 기반 합성 GU 계산값이다. 사용자가 말한 “GU 식”은
이 기존 구현을 뜻하며 실제 glossmeter 입력은 추가하지 않는다. 기존 코드와 결과의 필드명을
마음대로 바꾸지 않고, 문서에서도 실측값이라고 허위 표기하지 않는다.

1차 종료 직후에는 전체 cell에 대해 딱 한 번 `local_gu`, `local_ra_um`, `local_rz_um`,
`scratch_pass`, `all4_pass`, clearcoat/temperature 안전 mask를 계산한다. 각 cell에는 다음을
보존한다.

- 1차 후 네 품질값과 개별 pass/fail
- `all4_pass`와 안전 mask
- 적용 recipe, pass count, 누적 접촉시간과 안전 event
- surface/cell ID, 위치, 법선, 곡률 위험도
- 재작업 여부와 최종 상태

상태 전이는 다음처럼 고정한다.

- `PASS_LOCKED`: 1차 all-4 통과. 2차 target에서 제외하며 실제 pad가 다시 닿지 않으면
  재검증하지 않는다.
- `REWORK_AFFECTED`: 2차 target은 아니었지만 pad footprint가 실제로 다시 닿아 상태가
  변경됨. `PASS_LOCKED`를 해제하고 2차 재검증 대상에 포함한다.
- `REWORK`: all-4 중 하나 이상 미달이지만 재작업 안전조건 통과
- `UNSAFE_GEOMETRY`: 곡률/경계/normal/fit confidence 위험으로 자동 재작업 금지
- `CLEARCOAT_GUARD`: clearcoat 또는 누적 작업량 안전 한계로 재작업 금지
- `NOT_REACHED`: 1차 경로가 유효하게 도달하지 못함
- `NO_IMPROVEMENT`: 재작업 뒤 개선이 동결 기준보다 작음
- `REWORK_PASS`: 2차 후 all-4 통과
- `REWORK_FAIL`: 허용한 2차 후에도 all-4 미달

핵심 불변조건은 **상태가 변경되지 않은 `PASS_LOCKED` cell을 다시 평가하지 않는 것**이다.
2차 품질 계산은 실제 pad footprint 때문에 표면 상태가 변경된 cell mask에만 수행한다.
10×10 mm 국소창 계산에 주변 입력이 필요해도 변경되지 않은 잠긴 cell의 기존 판정은
유지한다. 반대로 footprint가 실제로 닿은 잠긴 cell은 `REWORK_AFFECTED`로 바꾸고 반드시
재검증한다.

## 20. F10-E — 2차 선택 재작업 planner

계획 파일:

- `learning/rl/gate_f10_rework_planner.py`
- `learning/rl/tests/test_gate_f10_rework_planner.py`

개별 실패 cell을 각각 닦지 않는다. 인접한 `REWORK` cell을 connected component로 묶고
pad footprint와 정지거리를 고려해 연속 재작업 경로를 만든다. `PASS_LOCKED` 영역과의
불필요한 중첩은 최소화하되 실제 footprint가 닿은 cell은 `REWORK_AFFECTED`로 기록한다.
`UNSAFE_GEOMETRY`, point-cloud hole, 도막 보호 영역은 침범하지 않도록 clip하거나 해당
경로를 거부한다. 실제 접촉으로 변경된 cell mask를 별도로 기록한다.

2차는 1차보다 낮은 feed와 좁은 step-over를 사용할 수 있지만 처음에는 최대 1회 추가
pass로 제한한다. 영역당 한 번 접근/복귀한다. 종료 뒤에는 전체 차량을 다시 검사하지 않고
**실제로 변경된 cell만** 네 품질항을 재계산한다. 변경되지 않은 `PASS_LOCKED` cell은
재계산하지 않고, footprint가 닿은 `REWORK_AFFECTED` cell은 함께 재계산한다. 재작업 cell이
all-4를 통과하거나 개선이 정체되거나 안전/도막 한계에 도달하면 끝내며 무제한 반복은
금지한다.

## 21. F10-F — 곡률 대응 강화학습

곡률에 따른 강화학습은 한다. 다만 cylinder용, sphere용, freeform용 모델을 각각 만드는
것이 아니라 하나의 정책이 연속적인 표면 상태에 조건부로 반응하도록 설계한다.

후보 관측은 signed `k1/k2`, pad 대비 곡률 위험도, footprint normal spread, fit confidence,
접촉 gap, 측정 normal force, force 변화율이다. 출력은 기존 policy 전체를 대체하는 큰
명령이 아니라 baseline recipe에 더하는 bounded residual force/feed 보정을 우선한다.

안전 역할은 RL에 전부 맡기지 않는다.

- geometry 기반 목표 force envelope
- force command slew/rate limiter
- contact-entry soft-start
- hard force/thermal/instability termination
- low-confidence·경계 영역 감속/skip

위 결정론적 보호 안에서만 RL residual을 적용한다. F9처럼 geometry 종류 전체에 동일한
고정 cap 하나를 쓰는 방식은 후보에서 제외한다. 비교 arm은 실행 전에 다음처럼 동결한다.

1. 기존 E7 control
2. curvature-risk 기반 deterministic derating
3. force slew limiter + entry soft-start
4. 2와 3의 결합
5. 앞선 안전 baseline 위 curvature-conditioned bounded residual RL

F7에서 feature 추가 BC가 악화된 전례가 있으므로 먼저 deterministic unit test와 offline
ablation을 한다. 새 관측/RL이 기준을 못 넘으면 PhysX 본평가나 큰 PPO로 넘어가지 않는다.
학습은 별도 신규 checkpoint/output에만 쓰며 기존 E7 release/champion은 건드리지 않는다.

## 22. F10-G — 단계별 검증 gate

작은 PhysX pilot은 대략 24~48 sequences로 flat, 완만/중간/고곡률, F9형 cylinder,
freeform, concave/saddle, 경계/저신뢰 구간을 포함한다. seed, profile, direction, threshold,
checkpoint hash를 결과 전에 동결한다.

최소 합격 조건:

- candidate force/thermal/instability overload 0
- sensor fault 0, NaN/Inf 0, sequence/tile 완결
- flat action/parity와 기존 품질 비열화 없음
- cylinder와 freeform 모두 안전 개선; 한쪽 개선으로 다른 쪽 악화 금지
- 곡률 구간 사이 force/feed command 불연속과 saturation 없음
- 기존 GU 식과 all-4 품질 저하가 동결 한계 이내
- 변경되지 않은 `PASS_LOCKED` cell의 재판정 수 0
- 2차 품질 재계산 대상이 실제 변경 cell mask와 정확히 일치
- 같은 입력/seed 재현성과 결과 checksum PASS
- 실패/중단/censoring을 성공 또는 속도 개선으로 계산하지 않음

작은 pilot PASS 후에만 독립 seed 확대 검증과 필요한 최소 RL 학습을 승인받는다. 결과를
본 뒤 threshold, arm, seed를 바꾼 재평가는 새로운 gate로 분리한다.

물리 시간 acceptance도 별도로 둔다. simulator wall-clock과 실제 차량 예상시간을 섞지
않는다. 20시간 기준 대비 최소 50% 단축을 1차 목표로 보고, 현재 잠정 목표는 로봇 3대가
충돌 없이 병렬 작업할 때 차량 한 대 **2~4시간**이다. 이는 아직 계산/실차 검증 전
목표치이며 보장값이 아니다.

## 23. F11 — `polishing_v5` 이식과 최종 회귀

F10의 geometry, safety, GU, scheduler, rework, RL gate가 모두 통과한 뒤 별도 승인을 받아
이식한다. 계획 모듈은 다음과 같다.

- `scripts/polishing_v5_modules/surface_geometry.py`
- `scripts/polishing_v5_modules/pass_scheduler.py`
- `scripts/polishing_v5_modules/rework_planner.py`
- `scripts/polishing_v5_modules/force_safety.py`

가능하면 `scripts/polishing_v5.py`는 그대로 두고 `agent.py`와 `runner.py`에 최소 hook만
추가한다. 새 기능은 feature flag 기본 OFF로 두어 기존 동작을 즉시 복원할 수 있게 한다.
기존 평면 동작 exact regression, dry-run 경로 검증, 작은 simulated run, 독립 seed 검증,
실차 저위험 구간 순으로 넓힌다. 기존 release 교체와 자동 promotion은 하지 않는다.

최종 검증에서는 기존 GU/Ra/Rz/scratch 식의 동일성, 도막/온도 안전, 충돌·가동범위,
비상정지와 사람 확인 절차를 확인한다. 실제 GU 센서 연결은 범위에 포함하지 않는다.
시뮬레이션 통과만으로 무인 전체 차량 production-ready라고 선언하지 않는다.

## 24. 앞으로 만들 예정인 파일과 예상시간

아래 파일은 **계획**이며 현재 이 문서 갱신 시점에는 아직 만들지 않았다.

| 단계 | 계획 파일/산출물 | 예상 경과시간 |
|---|---|---:|
| F10-A | F9 분석 및 차량 시간 기준선 신규 결과 디렉터리 | 3~5시간 |
| F10-B | `gate_f10_vehicle_geometry.py`와 test | 3~5시간 |
| F10-C~E | pass scheduler, GU 판정 구조, rework planner와 test | 4~7시간 |
| F10-F/G | 안전 후보, 곡률 조건부 RL 소규모 학습, PhysX pilot/집계 | 8~16시간 |
| F11 | v5 모듈 4개, 최소 hook, 회귀·최종 문서 | 8~14시간 |

전체 예상은 **약 30~50시간, 보통 2~4일**이다. GPU/Isaac Sim 재시작, 실패 원인,
학습 반복 횟수에 따라 늘어날 수 있다. 실제 GU 센서 연결·calibration 시간은 이 계획에
포함하지 않으며 추가하지 않는다.

첫 3~5시간의 F10-A는 STOP/GO 지점이다. 여기서 원인과 실제 시간 구조를 확인하면 큰
구현·학습을 계속할 가치가 있는지 다시 사용자에게 보고한다.

## 25. 다음 담당자의 첫 행동

1. 세 인수인계서를 처음부터 끝까지 읽는다.
2. `git status`, 현재 프로세스, 보호 해시, E7 manifest/checksum을 읽기 전용 확인한다.
3. F4~F9 canonical decision/README/checksum과 실제 코드 파일을 대조한다.
4. 사용자가 돌리는 백그라운드 프로세스를 중단하지 않는다.
5. 외부 코드나 외부 저장소를 확인하지 않는다.
6. 차이가 있으면 고치지 말고 먼저 사용자에게 보고한다.
7. 첫 실행은 F10-A의 **읽기 전용 F9 overload 원인분석과 로컬 차량 경로 시간 산정**으로
   제안한다. 이 단계에서는 코드 수정, PhysX, 학습을 하지 않는다.
8. F10-A에서 읽을 입력, 만들 결과 파일, 명령, 3~5시간 예상과 전체 30~50시간 예상을
   먼저 설명하고 사용자 승인을 기다린다.
9. 이후에도 구현, PhysX, RL 학습, v5 이식을 각각 별도 승인받는다.

## 26. 새 채팅에 그대로 붙여 넣을 프롬프트

```text
다음 세 인수인계서를 처음부터 끝까지 읽고 실제 로컬 상태를 이어받아줘.

1. /home/rokey/cacadaca/gloss_test/docs/새채팅_전체작업_인수인계서.md
2. /home/rokey/cacadaca/gloss_test/docs/새채팅_평면관측확장_경로최적화_인수인계서.md
3. /home/rokey/cacadaca/gloss_test/docs/새채팅_곡면_GateF_인수인계서.md

외부 코드와 외부 저장소는 확인하지 마. 백그라운드에서 돌아가는 프로세스도 중단하지 마.
F1부터 다시 만들거나 완료된 F4~F9 평가를 반복하지 마.

현재 F9은 256 sequences와 6,400 tiles까지 완결됐지만 최종 FAIL이야. canonical 결과는
/home/rokey/cacadaca/learning/rl/robot/results/gate_f9_summary_20260901_132500/이야.
static_cap(+0.50)은 freeform overload를 8에서 0으로 줄였지만 cylinder overload를 4에서
11로 늘려 integration_candidate=false야. 그대로 polishing_v5에 이식하면 안 돼.
F9에서는 학습/PPO/promotion/release 수정이 없었어.

최종 방향은 차량 point cloud에서 법선과 signed principal curvature k1/k2, 곡률반경,
pad footprint normal spread와 경계 신뢰도를 계산하고, 기존 C/SL/SR 연속 경로로 1차를
빠르게 한 번 폴리싱한 뒤 기존 GU/Ra/Rz/scratch 식으로 모든 cell을 한 번 검사해. 네 조건을
모두 통과한 cell은 PASS_LOCKED로 두고 2차 target에서 제외해. 실패 cell만 인접 영역으로
묶어 2차 정밀 폴리싱하고, 2차 뒤에는 실제로 표면 상태가 변경된 cell만 all-4를 재계산해.
변경되지 않은 PASS_LOCKED cell은 재검증하지 말고, pad footprint가 실제로 다시 닿은 통과
cell은 REWORK_AFFECTED로 잠금을 풀어 함께 재검증해. 전체 차량을 다시 검사하지 마.
cell마다 로봇을 정지/복귀/reset하거나 독립 episode로 만들지 마.

실제 GU 센서는 연결하지 않아. learning/polytwin/gloss_proxy.py의 기존 GU 계산식과
learning/rl/gate_e6_area_quality.py의 2 mm cell·10×10 mm 국소창 all-4 판정을 그대로
사용해. 기준은 GU>=70, Ra<=0.20µm, Rz<=2.0µm, scratch 개선(또는 초기<0.05µm)이야.
clearcoat와 temperature는 별도 안전조건으로 유지해. 곡률 대응 강화학습도 하되 geometry별
개별 모델이나 하나의 고정 cap이 아니라 연속 곡률/법선/접촉 위험 관측에 조건부인 bounded
residual 정책으로 설계해. 목표 힘 envelope, force slew limit, contact-entry soft-start,
hard termination 같은 결정론적 안전장치 안에서만 RL을 허용해.

scripts/polishing_v5.py, scripts/polishing_v5_modules/,
learning/rl/env/polish_env.py, learning/rl/env/polish_env_cfg.py, 기존 BC champion,
Gate E7 release, 모든 기존 checkpoint/CSV/JSON/README/PPT 결과와 dirty worktree를
삭제·정리·덮어쓰기 하지 마. 기존 차량 path 생성 함수를 원본 scan_result/car에 직접
실행하지 마. 기존 path가 삭제될 수 있어.

지금은 수정, PhysX, 데이터 수집, BC/PPO/RL 학습을 하지 마. 먼저 문서와 F9 결과,
git status, 프로세스, 보호 해시를 읽기 전용으로 대조해. 그다음 F10-A에서 읽을 정확한
입력과 분석 항목, 새로 만들 결과 파일, 예상 3~5시간, 최종 이식까지 전체 30~50시간을
설명하고 내 승인을 기다려. 첫 승인은 F9 overload 원인분석과 로컬 차량 path 길이/물리
시간 계산까지만 의미하며 이후 구현·PhysX·학습·v5 이식은 각각 다시 승인받아.
```
