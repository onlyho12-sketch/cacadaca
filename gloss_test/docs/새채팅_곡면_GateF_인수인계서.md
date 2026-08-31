# 새 채팅용 곡면 Gate F 전체 인수인계서

최종 갱신 기준: 2026-09-01 KST  
저장소: `/home/rokey/cacadaca`  
현재 단계: **F1~F8 완료, F9 미착수**

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
- F8은 production release가 아니다. PPO도 하지 않았고 champion/release도 수정하지
  않았다. 다음 단계는 F9 다중 seed 대규모 검증이며 아직 시작하지 않았다.
- 마지막 회귀 시험은 **66 passed**였다.
- 마지막 확인 시 실행 중인 Isaac Sim, Gate F 평가, BC/PPO 학습 프로세스는 없다.

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

## 13. 해석 시 주의점

- F8 PASS는 작은 synthetic pilot 결론이지 production release가 아니다.
- `static_cap(+0.50)`은 release env에 통합되지 않은 평가 wrapper 후보다.
- F7 모델 3개는 ablation 산출물이며 champion/release가 아니다.
- incomplete/retry/superseded 디렉터리는 시행착오 감사 기록이므로 보존한다.
- F8 `061500`은 재현성이 확인된 중복이며 공식 표본 수에 더하지 않는다.
- 타일 평균과 내부 cell의 GU/Ra/Rz/scratch all-4 면적, whole-surface quality를 구분한다.
- 외부 `147/150`은 실제 차량 곡면 PhysX 성공률이 아니라 합성 평면 patch 평가다.

## 14. 다음 단계 F9 — 아직 시작하지 않음

목표는 `static_cap(+0.50)`을 control과 더 큰 독립 seed 범위에서 paired 검증하는 것이다.
F9 전에 실제 상태와 범위를 보고하고 승인을 받아야 한다.

권장 본안:

- 4 geometry × 4 profile × 2 direction × 4 seed × 2 arm
- 총 **256 sequences**
- 예상 실행·집계 시간 **6~10시간**

축소안은 2 seed, 128 sequences, 약 3~5시간이지만 근거가 약해 권장하지 않는다.
F9는 PPO/BC 학습이 아니라 control 대 static cap의 확대 검증이다.

실행 전에 동결할 것:

1. surface seed base 4개와 policy/physics seed pairing
2. same/cross 두 direction
3. 동일 E7 checkpoint와 SHA-256
4. curved-only `force_action <= +0.50` 및 flat exact passthrough
5. 조기 종료 time delta censoring/reporting
6. candidate curved safety 개선, candidate overload 0
7. flat safety·quality exact non-regression
8. curved quality/all-4 drop 최대 `1%p`
9. sensor fault 0, sequence/tile 완결성, 재현성, checksum
10. 결과 후 threshold/cap/seed 변경 금지

F9에서는 production/public env 통합, E7 release/champion 수정, 자동 승격, 사후 튜닝,
기존 결과 삭제, 중복 합산, 별도 승인 없는 학습을 하지 않는다. F9가 PASS해도
“integration candidate”까지며 실제 통합·release는 별도 승인 단계(F10 권장)다.

## 15. 다음 담당자의 첫 행동

1. 세 인수인계서를 처음부터 끝까지 읽는다.
2. `git status`, 프로세스, 보호 해시, E7 manifest/checksum을 읽기 전용 확인한다.
3. F4~F8 canonical decision/README/checksum과 실제 코드 파일을 대조한다.
4. 차이가 있으면 수정하지 말고 먼저 사용자에게 보고한다.
5. F9의 정확한 신규 파일·출력 경로·seed/기준과 F9부터 최종 release까지 예상시간을
   설명한다.
6. 사용자 승인을 기다린다. 승인 전 코드 수정, PhysX, 수집, 학습을 하지 않는다.

## 16. 새 채팅에 그대로 붙여 넣을 프롬프트

```text
다음 세 인수인계서를 처음부터 끝까지 읽고 실제 상태를 이어받아줘.

1. /home/rokey/cacadaca/gloss_test/docs/새채팅_전체작업_인수인계서.md
2. /home/rokey/cacadaca/gloss_test/docs/새채팅_평면관측확장_경로최적화_인수인계서.md
3. /home/rokey/cacadaca/gloss_test/docs/새채팅_곡면_GateF_인수인계서.md

현재 곡면 Gate F는 F1~F8까지 끝났고 다음은 F9 다중 seed 검증이야. F1부터 다시
만들거나 완료된 F4~F8 PhysX를 반복하지 마. 문서, git status, 실행 프로세스,
F4~F8 canonical decision/README/checksum, Gate E7 release와 보호 해시를 먼저 읽기
전용으로 대조해.

scripts/polishing_v5.py, scripts/polishing_v5_modules/,
learning/rl/env/polish_env.py, learning/rl/env/polish_env_cfg.py, 기존 BC champion,
Gate E7 release, 기존 checkpoint/CSV/JSON/README/PPT용 결과와 현재 dirty worktree를
삭제·정리·덮어쓰기 하지 마.

F8 결론은 curved-only force action 상단 +0.50 static_cap이 작은 synthetic pilot에서
curved safety 11/16을 16/16으로 올리고 overload 5건을 0건으로 줄였다는 거야. 하지만
production-ready가 아니고 release에 통합되지 않았어. F7 관측 확장은 NO-GO라 base14를
유지하며 F8에서는 학습/PPO/promotion이 없었어. 20260901_061500 Stage 2는 공식
051200 실행과 CSV 해시가 같은 중복 재실행이라 보존하되 공식 표본에 합치지 마.
canonical F8 summary는 gate_f8_summary_20260901_063000/이야.

권장 F9는 control 대 static_cap(+0.50)의 4 geometry × 4 profile × 2 direction ×
4 seed × 2 arm = 256 sequence paired validation이고 예상 6~10시간이야. 학습이나
release 통합은 하지 않고 seed pairing·checkpoint hash·flat exact passthrough·curved
safety 개선·candidate overload 0·flat 비열화 없음·curved all-4 drop <=1%p·sensor
fault 0·censoring·checksum을 실행 전에 동결해. F9 PASS도 integration candidate일
뿐 실제 통합/release는 별도 승인 단계로 둬.

지금은 수정하거나 PhysX/학습을 실행하지 마. 이해한 현재 상태, 보호 대상, 문서와
실제 파일의 차이, F9에서 정확히 만들거나 사용할 파일과 출력 경로, F9부터 최종
통합/release까지 전체 예상시간을 먼저 자세히 설명하고 내 승인을 기다려.
```
