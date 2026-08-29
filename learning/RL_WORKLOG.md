# Isaac Lab 폴리싱 디지털 트윈 — 작업 기록 및 인수인계

> 2026-08-27 작성. 이 문서 하나로 오늘까지의 전체 작업을 다른 사람/AI가 이어받을 수 있게 정리한다.
> 기준 문서: `learning/polytwin_docs/` (README + 00~07). 그 이전 문서(`POLYTWIN_RL_SPEC.md`,
> `PLAN_BC_V2.md` 7장)와 충돌하면 polytwin_docs 가 우선한다.

---

## 0. 30초 요약

- 논문 근거 폴리싱 디지털 트윈(표면·제거·GU proxy)을 **순수 numpy 로 구현·검증 완료** (`learning/polytwin/`).
- Isaac Lab DirectRLEnv 환경 구축, action=0 baseline 검증 완료 (`learning/rl/`).
- BO 로 "고정 레시피의 천장 = GU ~68 (< 목표 70)" 정량화 → **RL 의 존재 이유 확보**.
- PPO 5회 학습 (실패 4 + 성공 1): 보상 해킹 3종을 실측 진단·수정. **최종 챔피언은
  수제 정책 BC 부트스트랩** — baseline 대비 **GU +1.06, 잔존 스크래치 −30%**.
- 확정된 미결 과제: 스텝 대리 보상의 최적점 ≠ GU 최적점 (실험으로 증명됨) → 종말 보상 설계 필요.
- 다음 작업: 로봇 팔(M0609) 씬 추가 + GUI 시연.

---

## 1. 실행 환경 (이 머신 전용 — 반드시 이대로)

```bash
# (2026-08-28 갱신 — rokey 머신) venv 활성화 불필요. 래퍼 하나면 된다:
~/isaacsim/python.sh <스크립트> --headless
# python.sh 가 EXP_PATH/ISAAC_PATH/PYTHONPATH 를 자동 설정한다.
# (구 eon 머신의 ~/IsaacLab/.venv 절차는 이 머신에 없음 — isaaclab 6.1.17 + rsl-rl 5.0.1 이
#  ~/isaacsim/kit/python 에 직접 설치돼 있다.)
```

- 순수 numpy 부분(`learning/polytwin/`)은 `~/isaacsim/kit/python/bin/python3 -m ...` 으로 실행 가능.
- 경로 하드코딩(/home/eon)은 2026-08-28 전부 제거 — 스크립트가 자기 위치에서 repo 루트를 찾는다.

---

## 2. 만든 것 — 파일 맵

### 2.1 디지털 트윈 (Isaac Sim 불필요) — `learning/polytwin/`

| 파일 | 내용 | 기준 문서 |
|---|---|---|
| `config.py` | 전체 파라미터 + 근거 태그(L-DIRECT/L-DERIVED/PT-DESIGN) | 01, 02 |
| `surface_state.py` | SurfaceState 13개 상태맵. 초기 표면 생성: Ra 0.08μm 거칠기, Clearcoat 40~50μm 상관장, **스크래치 절차 생성**(랜덤 선분 + 가우시안 단면 홈, 깊이 0.05~2μm=L-DIRECT, 폭·개수=PT-DESIGN) | 02 §2·3 |
| `polishing_model.py` | 제거식(Preston 형) + 돌출부 선택제거 + dwell/pass/heat proxy + **RL 보상용 분해**(잔여 게이팅) | 02 §4~13 |
| `roughness_metrics.py` | Ra/Rz/잔존 스크래치 — 전부 높이맵에서 직접 계산 (Rz=Ra×n 환산 금지) | 02 §9 |
| `path_executor.py` | raster 실행기 + **k 캘리브레이션** | 02 §7 |
| `gloss_proxy.py` | GU proxy — 앵커 25/78/89, 목표 70. 4개 품질항 geometric 결합. q_optical 은 w=0 훅 (RTX 는 Step 8) | 03 |
| `bo_runner.py` | Constrained BO — scipy GP + EI, Sobol 초기표본, feasible 게이팅 | 05 |
| `tests/test_unit.py` | 02 §15 단위시험 10개 | |
| `tests/test_gloss.py` | 03 §12 단위시험 8개 | |
| `outputs/polishing_model_config.json` | **k = 1.3367e-06** (재현 기준값) | |
| `outputs/bo_best_recipe.json` | BO 최적 레시피 (05 §10 스키마) | |

실행:
```bash
PY=~/isaacsim/kit/python/bin/python3        # (2026-08-28 갱신)
$PY -m learning.polytwin.tests.test_unit    # 10/10 (캘리브레이션 포함, 수 분)
$PY -m learning.polytwin.tests.test_gloss   #  8/8
$PY -m learning.polytwin.bo_runner          # BO 32회 평가 (~10분)
```

### 2.2 Isaac Lab RL — `learning/rl/`

| 파일 | 내용 |
|---|---|
| `env/contact.py` | 원 시뮬 가상 스프링+어드미턴스의 벡터화 이식 (병렬, env 당 상태 3개) |
| `env/polish_env.py` | **DirectRLEnv** — 기준 제어기(raster+어드미턴스) + 임피던스형 잔차 2축 + 품질모델 결합. recipe 는 BO JSON 을 process context 로 로드 (원본 수정 금지) |
| `env/polish_env_cfg.py` | 60Hz physics / 20Hz 품질, 보상 가중치, 잔차 bound (±30% 힘 / ±50% 이송) |
| `tests/test_contact_replay.py` | 이식 검증 9종 (Isaac Sim 불필요) |
| `tests/test_lab_env_baseline.py` | **Gate 2**: action=0 baseline 10종 (Isaac Lab 필요) |
| `ppo_cfg.py` | rsl_rl 러너 설정 (train/eval 공용 — train_ppo 를 import 하면 argparse 가 실행되므로 분리) |
| `train_ppo.py` | 학습. `--resume <ckpt>` 로 BC 부트스트랩 미세조정 |
| `eval_ppo.py` | **짝지은 판정** — 같은 표면 seed 에서 baseline vs 정책 |
| `bootstrap_bc.py` | 수제 dwell 정책 → actor 모방학습 (BC) |
| `champion/model_bc.pt` | ★ **최종 챔피언 체크포인트** (2026-08-28 재생성 — RESULTS 6장) |

실행:
```bash
~/isaacsim/python.sh learning/rl/tests/test_lab_env_baseline.py --headless   # Gate 2
~/isaacsim/python.sh learning/rl/train_ppo.py --headless --num_envs 16 --max_iterations 4000
~/isaacsim/python.sh learning/rl/bootstrap_bc.py --headless                  # BC 부트스트랩
~/isaacsim/python.sh learning/rl/eval_ppo.py --headless \
    --checkpoint learning/rl/champion/model_bc.pt --num_envs 8 --episodes 2
```

주의: rsl_rl cfg 는 `handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))`
전처리 필수 (안 하면 `MLPModel got unexpected keyword 'stochastic'`).

### 2.3 문서·기타

- `learning/polytwin_docs/` — 현재 기준 문서 세트 9개 (2026-08-27 전달본 사본)
- `learning/POLYTWIN_RL_SPEC.md` — **구버전** (상단 폐기 배너). 논문 출처 추적용만
- `learning/RL_ISAACLAB_GUIDE.md` — 초기 가이드 (2026-08-26). 환경 활성화·마일스톤은 유효,
  Preston 보상 우선순위 등 일부는 polytwin_docs 로 대체됨

---

## 3. 검증된 수치 앵커 (재현 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 제거계수 k_literature_synthetic | **1.3367e-06** | reference(8N/5mm/s/RPM 4000·3250·2750/30분/raster) 평균 제거량 = 정확히 3.0000 μm 로 역산 |
| GU 매핑 회귀 | 0.2180→36.6 / 0.8815→71.7 GU | 03 §9 기준값 재현 |
| 어드미턴스 힘 도달 상한 (top) | **8.4 N** = K·(2·cdist−press_min) = 350·0.024 | Gate 2 실측. 그 위 명령은 포화 limit cycle |
| 도달가능 명령 수렴 | 6.0N 명령 → 6.000±0.000N | Gate 2 |
| RMPFlow 추종지연 (원 시뮬) | 1.91 cm **지속 바이어스** (1차 지연 아님) | contact 이식 검증. 새 환경은 lag=0 |
| BO 최적 레시피 (force≤8.4 재탐색) | 5.78N / 5.95mm/s / 5436rpm / spacing 0.184 / 2pass → GU 67.8 | outputs/bo_best_recipe.json |
| 고정 레시피 천장 | **GU ~68 < 목표 70** (64회 평가 전체) | RL 존재 이유 |

---

## 4. Gate 3 — PPO 학습 5회의 전체 기록 (보상 정렬 문제)

> 문서 06 §9 가 요구하는 reward hacking 회귀 기록. 각 원인·수치는 해당 코드 주석에도 있다.

| 회차 | 설정 | 결과 | 원인 진단 | 수정 |
|---|---|---|---|---|
| 1차 | 제거량 **합계** 보상 (defect +0.5 / healthy-over −1.5) | reward 3배↑ 인데 **GU 70→65, scratch 0.82→1.33μm 악화** | footprint 정상셀:결함셀 = 25:1 → 감점 총합이 지배 → "덜 문지르기" 학습 | 셀당 **평균**으로 변경 |
| 2차 | 평균 보상, 800 iter | reward 정상 상승, 짝지은 판정 ΔGU −0.45 (노이즈) | env 당 11 에피소드 — 데이터 부족 | **존재 증명** 수행(아래) 후 4000 iter |
| — | **존재 증명 (수제 정책 probe)** | "스크래치면 힘+30%/이송−50%" 두 줄 규칙이 **ΔGU +0.81, scratch −29%** | 환경·보상·관측이 유효하고 이길 전략이 존재함을 확인 | |
| 3차 | 4000 iter, 탐색↑ | reward 조기 수렴, ΔGU −0.71. 행동 진단: **스크래치 위/밖 전부 최대 포화** (off-scratch 스텝 12000중 34) | ① 정적 defect_mask → 지운 자리 farming ② 패드 Ø110 vs 패치 120mm → 관측 무변별 | ① 지급을 min(제거, **잔여**)로 게이팅 ② 관측을 코어(반경 절반) **잔여** 스크래치로 |
| 4차 | 게이팅+코어 관측 | **판정 "개선"**: ΔGU +0.08, scratch −11%. 행동: 스크래치 위 감속 > 밖 (방향성 학습) | 그러나 수제 정책에 못 미침 | BC 부트스트랩으로 전환 |
| BC | 수제 정책 모방 (4만 샘플, MSE 0.002) | **ΔGU +1.06 (68.18→69.24), scratch −30% (1.073→0.755μm)** | | ★ 최종 챔피언 |
| 미세조정 | BC 에서 PPO 1500 iter 이어감 | reward −670→−149 상승하는 동안 **GU 68.6→64.5 단조 하락**, 최종 ΔGU −2.40 | 같은 정책에서 출발한 단조 발산 = **스텝 대리 보상 최적점 ≠ GU 최적점 확정** | 산출물 폐기. 종말 보상 설계가 다음 과제 |

**교훈 요약**
1. 보상 스케일은 셀 개수 비율까지 봐야 한다 (합계 vs 평균).
2. 정적 마스크 보상은 farming 된다 — 잔여량으로 게이팅할 것.
3. 관측 범위가 패치 크기와 비슷하면 공간 변별 신호가 소멸한다.
4. **조밀 대리 보상만으로는 품질 목표에 못 간다** (미세조정 발산으로 확정).
5. 이기는 시연이 있으면 모방 부트스트랩이 백지 RL 을 압도한다 — 프로젝트의
   BC→잔차RL 철학이 디지털 트윈 단계에서도 재확인됨.

---

## 5. 정직성 경계 (발표·보고 시)

- 모든 품질 수치(GU·제거량·Ra/Rz·스크래치)는 **SYNTHETIC** — 논문 근거 모델의 출력이지
  실측이 아니다. "실제 스크래치를 제거했다" 표현 금지.
- 스크래치는 절차 생성한 가짜다. 정책이 배운 것은 "결함 **맵**에 적응하는 제어"라는
  능력이며, 실전에서는 그 맵을 μm급 결함 검사 장비가 공급해야 한다
  (**현 형상 스캔으로는 불가** — 실측 공백 목록 항목).
- 판정은 스크래치 셀에서의 기하학적 valley 깊이(주변 median 기준면 대비)로 하고,
  정책 관측은 값싼 근사(초기−누적제거, 약간 낙관적)를 쓴다 — 성적과 관측의 분리.
- 올바른 문장 예: *"논문 통계 기반 합성 결함 위에서, 결함 위치 적응 제어가 고정 레시피
  대비 합성 잔존 결함을 30% 줄임을 확인했다. 실측 보정은 Gate 7."*

---

## 6. 문서와 다르게 구현한 것 (근거 포함)

1. **footprint 가중치** — 02 §4 의 sum=1 정규화를 깊이에 곱하면 해상도 의존
   (02 §15 단위시험 10 위반). 깊이엔 무차원 모양함수, sum=1 은 통계 전용. 스케일은 k 가 흡수.
2. **k 캘리브레이션 probe** — 문서의 k=1 은 Clearcoat 클램프 포화로 선형 역산 붕괴
   (실측 42.5μm). 작은 probe(1e-12)로 재고 선형 스케일 + 검산.
3. **q_optical** — RTX 측정 파이프라인 부재로 w=0 + 훅 유지 (07 순서상 Step 8).
4. **BO force 상한** — 문헌 후보 상한 대신 Gate 2 실측 도달 상한 8.4N
   (kinematic 평가자의 "명령=달성" 가정 오류를 수정).
5. **잔차 행동** — 04 §7 두 안 중 임피던스형 2축 채택 (기존 프로젝트 잔차 개념과 일치).

---

## 7. 남은 작업 (우선순위 순)

1. **로봇 팔 시연** — `usd/env/m0609_with_polisher.usd` 를 `_setup_scene()` 에 추가,
   GUI 에서 baseline vs BC 챔피언 재생. RMPFlow 는 1~4 env 검증 전용, 학습은 배치 IK.
2. **종말 보상 설계** — 에피소드 끝 GU 직접 지급 (04 "Completion bonus" 자리).
   미세조정 발산 문제의 정공법.
3. **곡면 일반화 (Gate 4)** — 02 §6 표면 family (원통/구면/자유곡면/전이),
   train/val/test 형상 분리 manifest.
4. **BO outer loop** (05 §9) — 챔피언 정책 고정 후 BO 재탐색 → 정책 추가학습 반복.
5. **RTX 20° 측정** (Step 8) — q_optical 연결.
6. 실측 공백 목록 관리: μm급 결함 검사 장비, Clearcoat 두께·온도 한계, Gloss meter 보정.

---

## 8. 재현 절차 (처음부터 끝까지)

```bash
# 0. 활성화 (1장)
source ~/IsaacLab/.venv/bin/activate && source ~/isaacsim/setup_python_env.sh
cd /home/eon/Desktop/cacadaca

# 1. 디지털 트윈 검증 + k 캘리브레이션 (~5분)
python -m learning.polytwin.tests.test_unit
python -m learning.polytwin.tests.test_gloss

# 2. BO (~10분) → outputs/bo_best_recipe.json
python -m learning.polytwin.bo_runner

# 3. Isaac Lab 이식 검증
python learning/rl/tests/test_contact_replay.py          # Isaac Sim 불필요
python learning/rl/tests/test_lab_env_baseline.py --headless

# 4. 챔피언 재현: 부트스트랩 (~5분) → 판정
python learning/rl/bootstrap_bc.py --headless
python learning/rl/eval_ppo.py --headless \
    --checkpoint learning/rl/logs/polish_ppo/bootstrap/model_bc.pt --num_envs 8 --episodes 2
# 기대: ΔGU ≈ +1.0, Δscratch ≈ −0.3μm, 판정 "개선"
```


---

## 9. 2026-08-28 — 환경 결함 3종 수정 + 재판정 (rokey 머신 인수)

> 코드 리뷰에서 발견한 정확성 결함을 수정하고 챔피언을 재생성했다. 수치는 RESULTS 6장.

### 수정 내역

1. **보상 1-스텝 지연** — `_quality_update` 가 `_get_observations` 안에 있었는데, DirectRLEnv
   훅 순서는 `_get_dones → _get_rewards → _reset_idx → _get_observations` 다. 즉 스텝 t 의
   보상이 t−1 의 품질/힘으로 계산됐다 (힘 항은 `f̄(t−1) − cmd(t)` 비교). `_get_dones` 첫머리로 이동.
2. **리셋 오염** — 리셋 직후 관측 경로에서 이전 에피소드의 `_force_accum` 으로 **새 표면을 1스텝
   연마 + 그 값이 다음 보상에 지급**됐다. 1번 이동으로 구조적으로 소멸 + `_reset_idx` 에 env 별
   버퍼 청소 추가 (`_force_mean/_force_accum/_defect_removal/_healthy_over/cmd` 류).
3. **에피소드 상한 300→500 s** — 공칭 완주 242 s 에 여유 24% 뿐이라 경로의 37% 이상에서
   feed −50% 를 쓰는 dwell 정책만 truncation 에 잘렸다 (baseline 은 완주 → 판정 비대칭).
4. **(예방) actor mean 층 zero 초기화** — ppo_cfg 주석의 "작은 init_std → 초기 정책 ≈ 기준
   제어기"는 오해였다: rsl_rl 5.0.1 의 init_std 는 별도 std 파라미터일 뿐, mean 층은 기본 랜덤
   초기화라 초기 출력 |a|≈0.03~0.14 (최대 0.4, seed 별 방향 상이). train_ppo 신규 학습 시
   zeros_ 로 04 §2.2 요구를 실제로 충족. (05 §2.2 의 "std 는 rollout 비교로 결정"은 여전히 미수행.)
5. 부수: /home/eon 하드코딩 16곳 제거(리포 상대경로), `.gitignore` 재작성(BO recipe·k 캘리브레이션
   JSON 추적 + rl/logs 무시), recipe 폴백 시 대문짝 경고, 챔피언을 `rl/champion/` 로 이동,
   Gate 2 포화 검증을 명시적 10 N 주입으로 갱신(구 recipe 8.86 N 전제 제거), 죽은 코드
   (`_scratch_cost`, `_prev_scratch_cost`) 제거.

### 재검증 결과 (이 머신)

| 검증 | 결과 |
|---|---|
| contact replay | 9/9 |
| Gate 2 (action=0 baseline) | 10/10 — 6 N 명령 6.000±0.000 N, 10 N 명령 9.06 N 포화 |
| BC 부트스트랩 | MSE 0.0021, 모방 충실도 종전과 동일 |
| 짝지은 판정 | baseline 68.18/1.073 (종전 기록과 일치) vs **BC 69.74/0.448 — ΔGU +1.56, scratch −58%** |

개선폭이 커진 주원인은 3번(상한 확대): 종전 챔피언 수치(+1.06/−30%)는 dwell 2회차 pass 가
잘린 조건의 값이었다. **4장의 미세조정 발산 결론도 이 결함들 위에서 나온 것이므로, critic
워밍업(BC 가 critic 을 랜덤으로 저장하는 문제)과 함께 재실험해야 확정된다 — 7장 과제 2 앞에 배치.**

### 9.1 미세조정 재실험 (critic 워밍업, 2026-08-28 17:36 run)

`--freeze_actor_iters 200` 추가 (bootstrap 이 critic 을 랜덤 저장하는 문제 대응 — 첫 200 iter
actor 동결로 critic 만 학습 후 해제, lr 3e-4 복원). 수정된 환경에서 챔피언 resume, 총 1500 iter.

| 실험 | ΔGU vs baseline | 챔피언 대비 |
|---|---:|---:|
| 구 미세조정 (버그 환경·워밍업 없음) | −2.40 | −3.46 |
| **재실험 (수정 환경·워밍업 200)** | **−0.33** (67.85±2.44 / scratch 0.917) | **−1.89** |

**결론 확정**: 랜덤 critic + 환경 버그가 붕괴를 7배 증폭했던 것은 사실이나, 제거 후에도
PPO 미세조정은 챔피언을 밑돈다 → **스텝 대리보상 최적점 ≠ GU 최적점은 진짜다** (4장 교훈 4 유지).
다음 수단은 종말 보상(에피소드 끝 GU 지급, 7장 과제 2). 체크포인트: logs/polish_ppo/2026-08-28_17-36-53.

### 9.2 Clearcoat 안전기준 35 μm 통일 (2026-08-28 — 차량 검사 시스템 요구)

`CLEARCOAT_SAFETY_LIMIT_UM` 30→35 (polytwin/config + polish_env_cfg). 영향:
- **GU proxy 스케일이 함께 내려간다** — q_clearcoat 항이 이 상수를 쓰므로, 행동이 동일한
  baseline 도 GU 68.18→67.45. 30 기준 수치와 35 기준 수치는 **직접 비교 금지**.
- BC 챔피언 재생성 + 재판정 (35 기준 공식 수치): baseline 67.45/1.073 μm vs
  **BC 67.77/0.448 μm — ΔGU +0.32, Δscratch −58%**. scratch 개선은 불변, GU 이득은
  "결함부를 더 깎는" 정책이 clearcoat 항에서 더 감점되어 축소됨.
- "고정 레시피 천장 ≈ GU 68 < 70" 서술도 35 기준으로는 "≈ 67.5" 로 읽어야 함.
- BO recipe(recipe_00020) 는 30 제약 탐색본이나 clearcoat_min 38.38 ≥ 35 — feasible 유지.

### 9.3 판정 기준 확정 (2026-08-28)

차량 셀 판정 4종 — Ra ≤ 0.20 μm / Rz ≤ 2.0 μm / 20° GU proxy ≥ 70 / 잔여 Clearcoat ≥ 35 μm —
은 **literature-derived project target** (제공된 도장·연마 논문 결과를 근거로 프로젝트가 채택한
문헌 기반 목표값, 논문 기반 디지털 트윈 판정값)이다. 프로젝트 자체 실측·보정값이 아니며,
결과가 미달해도 목표값을 낮추지 않고 실패로 판정한다. 코드·CSV·JSON·README 통일 완료
(learning/vehicle_export/).

### 9.4 종말 보상 실험 — "논문 기반 최종 GU proxy 종말 보상" (2026-08-28 18:51 run)

**구현**: 에피소드 종료 시 같은 에피소드의 전·후 품질로 지급 (polish_env_cfg 주석 참고).
최종값(GU−70)과 개선량(ΔGU·Δscratch·ΔRa·ΔRz)을 함께 사용, 판정 5종 전부 통과 +500,
잔여 clearcoat 최소 <35 μm 는 −1500 (GU 를 위해 clearcoat 를 희생하는 전략 차단).
dense 보상은 4용도(접촉력 안전/급변 억제/clearcoat 보호/결함 방향)로 유지.
BC 보호: champion resume + 워밍업 200 + lr 1e-4 + clip 0.1 + desired_kl 0.005 + γ 0.9995
(종말 신호가 ~4800스텝 에피소드 앞까지 닿도록 — 0.99 로는 0.99^4800≈1e-21 로 소멸).

**체크포인트 선택**: PPO reward 가 아니라 품질 판정 (results/ckpt_selection.csv, 50 iter 저장분
중 200~1498 sweep) → **it400** 채택. 조기 종료는 50 iter 스냅샷 + 사후 품질 선택으로 갈음.

**같은 seed 4조건 최종 비교** (results/eval_conditions.csv, 조건당 16 에피소드):

| 조건 | GU after | GU≥70 | scratch | Ra pass | Rz pass | CC pass | ★동시통과 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (action=0) | 67.46 | 1 | 1.073 | 16/16 | 16/16 | 16/16 | 1/16 |
| BC 챔피언 | 67.77 | 1 | **0.448** | **2/16** | 16/16 | 16/16 | **0/16** |
| 기존 BC+PPO (17-36-53 final) | 67.49 | 3 | 0.788 | 6/16 | 15/16 | 16/16 | 1/16 |
| **종말보상 BC+PPO (18-51-54 it400)** | **68.57** | 3 | 0.859 | **16/16** | 16/16 | 16/16 | **3/16** |

**발견 — 5종 판정이 BC 챔피언의 숨은 약점을 드러냄**: BC dwell 은 scratch 를 가장 잘 지우지만
(0.448) 공격적 연마로 **전역 Ra 를 0.234 μm 로 악화**시켜 Ra≤0.20 을 14/16 실패 → 동시통과 0.
종말보상 PPO 는 균형 전략(scratch 는 덜 지우지만 Ra 16/16·Rz 최저 1.469·GU 최고 68.57)을 학습.
"scratch 만 보면 BC, 문헌 기반 5종 판정으로는 종말보상 PPO" — 목적함수가 승자를 바꾼다.

**챔피언 결정**: 사용자 지시(BC 유지, BC+PPO 로 교체 금지)에 따라 **champion/model_bc.pt 유지**.
종말보상 후보는 champion/model_terminal_ppo_it400.pt 로 보존 (판정 1위, n=16 소표본 주의).
교체 여부는 프로젝트 결정 사항으로 남긴다.
