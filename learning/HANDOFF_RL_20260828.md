# 폴리싱 디지털 트윈 RL — 인수인계 문서 (2026-08-28)

> 이 문서 하나로 "여태까지 뭘 했고, 어떻게 했고, 지금 어디까지 왔는지"를 이어받을 수 있게 쓴다.
> 세부 실험 기록은 `RL_WORKLOG.md`(회차별 전체 기록), 공식 수치는 `rl/results/RESULTS.md`.
> 문장 규칙: 모든 품질 수치는 논문 기반 모델의 **SYNTHETIC(합성)** 출력이다 — 실측이 아니다.

---

## 1. 한눈에 보기

**목표**: 자동차 도장면 폴리싱을 시뮬레이션(디지털 트윈) 안에서 학습시켜,
스크래치 위치에 따라 힘·속도를 스스로 조절하는 제어 정책을 만든다.

**현재 챔피언**: `rl/champion/model_bc.pt` — **BC 정책** (사람이 짠 두 줄 규칙
"스크래치 보이면 힘 +30%·속도 −50%"를 모방학습한 것). PPO(강화학습)가 아니다.

**핵심 결론 3줄**:
1. 고정 레시피(BO 최적)만으로는 광택 GU ~67.5가 천장 → 위치 적응 제어(RL/BC)가 필요하다.
2. 백지 PPO도, BC에서 이어 학습한 PPO도 **스텝 점수(대리 보상)로는 BC를 못 이겼다** —
   점수는 오르는데 실제 품질이 떨어지는 현상을 실험으로 확정했다.
3. **에피소드가 끝났을 때 최종 품질로 크게 채점하는 "종말 보상"**을 넣자, 5종 품질 판정
   기준으로는 PPO가 처음으로 BC를 이겼다 (단, scratch 단일 지표는 여전히 BC가 최강 —
   챔피언 교체 여부는 미결, 8장).

---

## 2. 시스템이 어떻게 생겼나

```
[디지털 트윈 — 순수 numpy, Isaac Sim 불필요]  learning/polytwin/
   surface_state.py    합성 표면 생성 (거칠기 + 스크래치 + 클리어코트 두께)
   polishing_model.py  "문지르면 얼마나 깎이나" — 논문 기반 제거 모델
   gloss_proxy.py      깎인 표면 → 20° 광택(GU proxy) 추정. 목표 70
   bo_runner.py        공정 레시피(힘/속도/rpm/간격/횟수) 자동 탐색 (베이지안 최적화)
        └→ outputs/bo_best_recipe.json : 5.78 N / 5.95 mm/s / 5436 rpm / 0.184 / 2 pass

[RL 환경 — Isaac Lab]  learning/rl/
   env/contact.py      가상 스프링 + 어드미턴스 힘제어 (원 시뮬에서 이식, 9/9 검증)
   env/polish_env.py   위 재료를 합친 학습 환경. 정책은 레시피 위에 ±30% 힘 / ±50% 속도
                       "잔차(보정값)"만 얹는다. action=0 이면 기준 제어기 그대로.
   bootstrap_bc.py     수제 규칙 → 신경망 모방학습(BC). 챔피언 제조기 (~5분)
   train_ppo.py        PPO 학습 (+ --resume BC 미세조정, --freeze_actor_iters 워밍업,
                       --lr/--clip_param/--desired_kl/--gamma 재정의)
   eval_ppo.py         짝지은 판정: 같은 표면 seed에서 baseline vs 정책
   eval_conditions.py  여러 정책을 같은 seed로 한 번에 비교 + 에피소드별 CSV

[차량 연동]  learning/vehicle_export/
   export_vehicle_results.py  검사 시스템 입력 CSV(150셀) → 정책 실행 → 셀별 전후 품질 CSV
```

**판정 기준 (literature-derived project target — 문헌 기반 목표값, 실측·보정값 아님)**:
GU proxy ≥ 70 / Ra ≤ 0.20 μm / Rz ≤ 2.0 μm / 잔여 클리어코트 ≥ 35 μm / scratch 초기 대비 감소.
미달이면 목표를 낮추지 않고 그대로 실패로 판정한다.

---

## 3. 시간순 — 뭘 했고 어떻게 했나

### 3-1. 트윈 구축과 BO (2026-08-27, 이전 담당)
- 논문 수치로 표면·제거·광택 모델을 만들고 단위시험 10+8개로 검증.
- 제거계수 k는 "기준 공정 30분 = 평균 3μm 제거"가 되도록 역산 (k=1.3367e-06).
- BO 32회 탐색으로 최적 레시피 도출. 이때 "고정 레시피 천장 = GU ~68 < 목표 70"을
  정량화 — **이것이 RL을 도입한 근거다.**

### 3-2. PPO 5회 실패와 BC 챔피언 (2026-08-27, 이전 담당)
- 보상 설계 실수 3종(합계 함정·farming·관측 무변별 — 5장 참고)을 실측으로 잡아가며 4회 학습.
- 그래도 수제 규칙(존재 증명 ΔGU +0.81)에 못 미쳐 **모방학습(BC)으로 전환** → 챔피언 등극.
- BC에서 PPO로 이어 학습(미세조정)하자 오히려 크게 나빠짐(ΔGU −2.40) → "대리보상 문제" 의심.

### 3-3. 환경 결함 수정 (2026-08-28, 이번 세션) — 이후 모든 수치의 기준선
코드 리뷰에서 발견한 4가지를 고치고 챔피언을 재생성했다:
1. **보상 한 박자 지연**: 매 스텝 "직전 행동"의 결과로 채점되고 있었다 (Isaac Lab 훅 순서
   문제). 품질 계산을 `_get_dones` 첫머리로 이동해 "이번 행동 → 이번 점수"로 정렬.
2. **리셋 오염**: 새 에피소드의 표면이 이전 에피소드의 힘으로 한 번 문질러진 채 시작했다.
3. **타임아웃 비대칭**: 완주 242초에 상한 300초 — 속도를 늦추는(dwell) 정책만 중간에
   잘렸다. 500초로 확대하자 챔피언 성적이 scratch −30% → −58%로 뛰었다
   (기존 수치가 "잘린 조건"의 값이었다는 뜻).
4. **초기 정책 zero-init**: 설계 문서는 "PPO 시작 시 출력 ≈ 0(기준 제어기와 동일)"을
   요구했는데, 코드는 이를 init_std(노이즈 폭)로 착각하고 있었다. 마지막 층 0 초기화 추가.
- 부수: /home/eon 경로 하드코딩 16곳 제거(다른 머신 재현 가능), .gitignore 재작성
  (BO 레시피 JSON 추적 + 112MB 로그 무시), 레시피 없을 때 조용한 폴백 → 대문짝 경고.

### 3-4. 랜덤 critic 검증 (이번 세션)
- BC는 "행동"만 배우고 "이 상황이 좋은지 판단하는" critic은 랜덤인 채 저장한다.
- 미세조정 실패가 이것 때문인지 확인하려고 **critic 워밍업**(첫 200 iter는 actor를 얼려두고
  critic만 학습)을 넣고 재실험 → 붕괴가 −2.40에서 −0.33으로 줄었지만 여전히 BC 미달.
- **결론 확정: 랜덤 critic은 증폭기였을 뿐, 근본 원인은 "스텝 점수 최적점 ≠ 품질 최적점".**

### 3-5. 클리어코트 35μm 통일 (이번 세션, 검사 시스템 요구)
- 안전기준 30→35μm. **주의**: GU proxy가 이 상수를 품질항에 쓰기 때문에 GU 스케일이
  전체적으로 ~0.7 내려갔다 (아무것도 안 바꾼 baseline도 68.18→67.45).
  → **30 기준 시절 수치와 35 기준 수치를 직접 비교하면 안 된다.**

### 3-6. 차량 150셀 연동 (이번 세션, 검사 시스템 요구)
- 6영역(보닛/루프/도어×2/펜더×2)×5×5셀 입력 CSV → 셀마다 대표 패치를 합성해 정책 실행
  → 셀별 전후 품질 + 판정 CSV/JSON. Isaac Sim 앱 없이 돈다 (~1분/150셀).
- 결과 62/150 통과. 실패 1위는 GU<70 (87건) — 초기 스크래치가 깊은 셀(1μm 이상)은
  한 에피소드로 70에 못 닿는 구조적 한계 + 35μm 통일로 빠듯해진 GU 스케일.
- **한계 명시**: 정책은 평면 패치 학습본이라 차량 곡면/수직면 결과는 "추론"일 뿐
  검증된 곡면 일반화가 아니다 (`evaluation_mode` 열, README 참고). 수직면(도어)은
  side 접촉 물리의 도달 상한(~3.5N)에 걸려 제거량이 절반이다.

### 3-7. 종말 보상 (이번 세션의 마지막 실험)
- **아이디어**: 매 스텝 잘게 점수 주는 대신, 에피소드가 끝났을 때 같은 에피소드의
  전·후 품질(GU proxy·scratch·Ra·Rz·클리어코트)로 크게 채점한다. 최종값과 개선량(Δ)을
  같이 쓰고, 판정 5종 전부 통과하면 +500, 클리어코트 35μm 미달이면 −1500
  ("GU만 노리고 도장을 갈아버리는" 전략을 막는 안전핀).
- **표기 규칙**: 실측 GU가 없으므로 "실제 GU 종말 보상"이 아니라
  **"논문 기반 최종 GU proxy 종말 보상"**이라고 부른다.
- **γ(할인율) 함정**: 에피소드가 ~4,800스텝인데 γ=0.99면 끝의 보상이 앞까지 전달되지
  않는다 (0.99⁴⁸⁰⁰ ≈ 10⁻²¹). γ=0.9995로 올려서 해결.
- **BC 보호**: BC에서 출발 + 워밍업 200 + 작은 lr(1e-4)/clip(0.1)/KL(0.005)로
  정책이 초반에 크게 이탈하지 않게 했다.
- **체크포인트 선택**: PPO reward가 아니라 **품질 판정**으로 골랐다
  (50 iter마다 저장분을 같은 seed로 판정 → it400 채택).

---

## 4. 현재 공식 수치 (같은 표면 seed, 조건당 16 에피소드 — rl/results/eval_conditions.csv)

| 조건 | GU proxy | GU≥70 | scratch [μm] | Ra≤0.20 | 동시통과 |
|---|---:|---:|---:|---:|---:|
| baseline (레시피 그대로) | 55.06→67.46 | 1/16 | 1.628→1.073 | 16/16 | 1/16 |
| **BC 챔피언** (현 챔피언) | →67.77 | 1/16 | →**0.448** | **2/16** | **0/16** |
| 기존 BC+PPO (폐기) | →67.49 | 3/16 | →0.788 | 6/16 | 1/16 |
| **종말보상 BC+PPO (it400)** | →**68.57** | 3/16 | →0.859 | **16/16** | **3/16** |

**중요한 발견**: 5종 판정을 켜자 BC 챔피언의 숨은 약점이 드러났다 — 스크래치는 제일 잘
지우지만 공격적으로 갈아내느라 **표면 거칠기(Ra)를 0.234μm로 악화**시켜 동시통과가 0이다.
종말보상 PPO는 "스크래치는 덜 지워도 Ra·Rz·GU를 함께 챙기는" 균형 전략을 배웠다.
**어느 지표를 우선하느냐에 따라 승자가 바뀐다** — 이것이 챔피언 미결 사유다.

---

## 5. 겪은 문제 전체 목록 (같은 실수 반복 방지용)

**보상 설계** — 제일 어려웠던 것:
1. 제거량 "합계" 보상 → 정상셀이 결함셀보다 25배 많아 감점이 지배 → "덜 문지르기" 학습. **셀당 평균으로.**
2. 결함 위치를 고정 지도로 주면 → 이미 지운 자리를 계속 문질러 점수 수확(farming). **잔여량으로 게이팅.**
3. 패드(Ø110mm)가 패치(120mm)를 다 덮으면 → "스크래치 없음" 신호가 안 나옴. **관측을 패드 중심부로 축소.**
4. 스텝 점수가 오른다고 품질이 오르지 않는다 — 반대로 갈 수 있다 (실험으로 확정). **최종 품질로 채점.**
5. 긴 에피소드 + 낮은 γ → 끝의 보상이 앞에 안 닿음. **γ를 에피소드 길이에 맞춰라.**
6. 단일 지표(scratch)만 보면 다른 지표(Ra)를 희생하는 정책을 못 알아챈다. **판정은 다중 지표 동시로.**

**환경 구현**: 보상 1스텝 지연 / 리셋 오염 / 타임아웃 비대칭 / 테스트-레시피 불일치 (3-3장).

**학습 안정성**: 백지 PPO < 수제 규칙 모방(BC) / init_std ≠ zero-init / BC의 critic은 랜덤.

**모델 정합**: "명령한 힘 = 달성한 힘" 가정은 틀렸다 (어드미턴스 도달 상한 top 8.4N,
side ~2.8N) / GU proxy가 안전기준 상수에 의존해 기준 변경 시 스케일이 통째로 움직인다.

---

## 6. 실행 방법 (이 머신 — rokey)

```bash
cd /home/rokey/Desktop/cacadaca
# 전부 이 래퍼 하나로 실행 (venv 활성화 불필요, EXP_PATH 자동 설정):
~/isaacsim/python.sh <스크립트> --headless

# ① 환경 검증 (Isaac Sim 불필요/필요)
~/isaacsim/kit/python/bin/python3 learning/rl/tests/test_contact_replay.py     # 9/9
~/isaacsim/python.sh learning/rl/tests/test_lab_env_baseline.py --headless     # 10/10

# ② 챔피언 재현 (~5분) + 판정
~/isaacsim/python.sh learning/rl/bootstrap_bc.py --headless
~/isaacsim/python.sh learning/rl/eval_ppo.py --headless \
    --checkpoint learning/rl/champion/model_bc.pt --num_envs 8 --episodes 2

# ③ 종말 보상 PPO 재현 (~20분)
~/isaacsim/python.sh learning/rl/train_ppo.py --headless --num_envs 16 --max_iterations 1500 \
    --resume learning/rl/champion/model_bc.pt --freeze_actor_iters 200 \
    --lr 1e-4 --clip_param 0.1 --desired_kl 0.005 --gamma 0.9995

# ④ 다조건 비교 (체크포인트 선택·최종 판정 겸용)
~/isaacsim/python.sh learning/rl/eval_conditions.py --headless --episodes 2 \
    --conditions "baseline=,bc=learning/rl/champion/model_bc.pt,terminal=<ckpt>"

# ⑤ 차량 150셀 export (~1분)
~/isaacsim/python.sh learning/vehicle_export/export_vehicle_results.py \
    --checkpoint learning/rl/champion/model_bc.pt \
    --input learning/vehicle_export/vehicle_150_cells.csv \
    --output learning/vehicle_export/rl_vehicle_150_results.csv --workers 8
```

---

## 7. 산출물 위치

| 파일 | 내용 |
|---|---|
| `rl/champion/model_bc.pt` | ★ 현 챔피언 (BC, 35μm 기준 재생성본) |
| `rl/champion/model_terminal_ppo_it400.pt` | 종말보상 PPO 후보 (품질 판정 1위, 미채택) |
| `rl/results/eval_conditions.csv` | 4조건 에피소드별 전·후 품질 (판정 포함) |
| `rl/results/ckpt_selection.csv` | 체크포인트 품질 sweep (it200~1498) |
| `rl/results/terminal_ppo_curves.csv` | 학습곡선 (종말보상 run + 구 미세조정 run) |
| `rl/results/RESULTS.md` | 공식 수치 총정리 (6장 = 35μm 기준선, 9.4장 = 종말보상) |
| `vehicle_export/rl_vehicle_150_results.csv` + `_summary.json` | 차량 150셀 결과 + 실패 원인 |
| `RL_WORKLOG.md` | 회차별 전체 실험 기록 (9장 = 이번 세션) |

---

## 8. 미결 사항 — 다음 사람이 결정/진행할 것

1. **챔피언 교체 여부** (결정 필요): 5종 동시통과로는 종말보상 PPO(3/16)가 BC(0/16)보다
   낫지만, scratch 단일로는 BC가 최강. n=16 소표본이므로 **에피소드 4~8라운드로 재검증
   후 결정**을 권장. 사용자 지시로 현재는 BC 유지 중.
2. **git 커밋**: 오늘 작업 전체가 아직 미커밋 상태다 (`git add learning/` ≈ 44+파일, ~3MB —
   .gitignore 정리돼 있어 대용량 로그는 안 들어간다).
3. **곡면 일반화 (Gate 4)**: 평면→원통/구면/자유곡면 train/val/test 분리. 차량 export의
   "추론일 뿐" 딱지를 떼려면 필수.
4. **BO outer loop**: 챔피언 정책을 고정하고 레시피를 재탐색 (05 문서 9장) — 현 레시피는
   "명령=달성" 가정의 잠정치.
5. **로봇 팔 GUI 시연**: `rl/demo_arm.py` (경로 수정 완료, 이 머신에서 미실행).
6. **실측 보정 (Gate 7)**: GU proxy·제거 모델의 실측 검증 — 그 전까지 모든 수치는 SYNTHETIC.
