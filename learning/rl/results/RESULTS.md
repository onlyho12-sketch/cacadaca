# 강화학습 결과 데이터 — Gate 3 (2026-08-27)

> 원시 데이터 위치와 최종 수치의 공식 기록. 서사·진단은 `learning/RL_WORKLOG.md` 4장.

## 1. 최종 스코어보드 (짝지은 판정 — 같은 표면 seed, patch 0.12 m / 2 pass)

| 정책 | GU | 잔존 scratch [μm] | ΔGU | Δscratch | 판정 |
|---|---:|---:|---:|---:|---|
| baseline (action=0, 기준 제어기) | 68.18 ± 0.04 | 1.073 ± 0.054 | — | — | — |
| 백지 PPO 4차 (`15-33-16` → 재학습 `16-01-35`) | 68.26 ± 0.75 | 0.959 | +0.08 | −11 % | 개선(미미) |
| **★ BC 부트스트랩** (`bootstrap/model_bc.pt`) | **69.24 ± 0.09** | **0.755 ± 0.034** | **+1.06** | **−30 %** | **개선 — 챔피언** |
| BC + PPO 미세조정 (`16-36-18`) | 65.78 ± 3.66 | 1.110 | −2.40 | +3 % | 미개선 — 폐기 |

참고 대조:
- 수제 dwell 정책 probe: ΔGU +0.81 / scratch −29 % (존재 증명)
- 시연 seed 1000 (patch 0.12, 2pass, 로봇 팔 포함): baseline GU 52.8/1.35 μm vs BC 66.4/0.60 μm

## 2. 챔피언 체크포인트

```
learning/rl/champion/model_bc.pt          # 2026-08-28 환경수정 후 재생성 (공식)
learning/rl/champion/model_bc_20260827_prefix.pt   # 수정 전 원본 (1장 표의 수치)
```
- 학습법: 수제 dwell 정책(스크래치 시 힘 +30 %/이송 −50 %, 아니면 −9 %/+25 %)을
  40,000 샘플 모방학습. 60 epoch, 최종 MSE 0.0021.
- 모방 충실도: 스크래치 위 Δforce +0.99/Δfeed −0.99 (목표 ±1.0), 밖 −0.25/+0.44 (목표 −0.3/+0.5)
- 입력: 11차원 (힘·힘오차·Δ힘·이송·진행률·코어 잔여 scratch 평균/최대·누적제거·clearcoat 여유·직전행동 2)
- 출력: 잔차 2축 [Δforce_ratio ±0.3, Δfeed_ratio ±0.5]
- 로드: rsl_rl `actor_state_dict` — `learning/rl/demo_arm.py` 의 `_load_policy()` 참고

## 3. 원시 데이터 위치

| 데이터 | 경로 |
|---|---|
| 학습곡선 (4 run, 29,688행) | `results/training_curves.csv` — 열: run, metric{reward,gu,scratch}, iteration, value |
| 학습곡선 그래프 | `results/training_curves.png` |
| tensorboard 이벤트 | `logs/polish_ppo/<타임스탬프>/events.out.tfevents.*` |
| 체크포인트 (50 iter 마다) | `logs/polish_ppo/<타임스탬프>/model_*.pt` |
| run ↔ 회차 매핑 | 15-25-00=2차 / 15-33-16=3차 / 16-01-35=4차 / 16-36-18=미세조정 / bootstrap=BC |
| BO recipe (process context) | `../polytwin/outputs/bo_best_recipe.json` — 5.78 N / 5.95 mm/s / 5436 rpm / 0.184 / 2 pass |
| 판정 스크립트 | `eval_ppo.py` (짝지은 비교), `train_ppo.py --resume` (미세조정 재현) |

※ 1차 학습(합계 보상)의 이벤트 파일은 초기 스모크 디렉토리(15-18-04 등)와 섞여 있고
  수치는 WORKLOG 4장에 기록돼 있다 (reward −3346→−1091, GU 70→65).

## 4. 곡선이 말하는 것 (training_curves.png)

- **run3 (주황)**: reward 조기 수렴 후 3,200 iter 평탄 — farming exploit 이 보상을 다 빨아먹은 상태.
- **run4 (파랑)**: 게이팅 후 reward 가 끝까지 상승 — 정책이 진짜 전략을 탐색.
- **finetune (보라)**: reward ↑ 인데 GU ↓ scratch ↑ — 스텝 대리 보상 최적점 ≠ GU 최적점의
  단조 발산 증거. 이 곡선 자체가 "종말 보상 필요"의 근거 그림이다.
- GU/scratch 는 에피소드 단위 랜덤 표면 평균이라 노이즈 ±2 GU — 최종 판정은 곡선이 아니라
  1장의 짝지은 비교로 한다.

## 5. 재현 명령

```bash
cd <repo>          # 경로 하드코딩 제거됨 — 어느 머신이든 동일 (2026-08-28)
# 챔피언 재학습 (~5분) + 판정
~/isaacsim/python.sh learning/rl/bootstrap_bc.py --headless
~/isaacsim/python.sh learning/rl/eval_ppo.py --headless \
    --checkpoint learning/rl/champion/model_bc.pt --num_envs 8 --episodes 2
# 기대 (6장): baseline GU 68.18/1.073μm vs BC GU 69.74/0.448μm — ΔGU +1.56, Δscratch −58%
```

## 6. 2026-08-28 환경 수정 후 재판정 (새 공식 기준선)

환경 3개 결함 수정 후 같은 절차로 재생성·재판정한 결과. 상세는 RL_WORKLOG 9장.

| 수정 | 내용 |
|---|---|
| 보상 1-스텝 지연 | `_quality_update` 를 관측→`_get_dones` 로 이동 (보상이 직전 스텝 품질로 계산되던 문제) |
| 리셋 오염 | 새 표면이 이전 에피소드의 힘으로 1회 연마되던 문제 + env 별 버퍼 청소 |
| 에피소드 상한 | 300→500 s — 공칭 완주 242 s 라 감속(dwell) 정책만 truncation 에 잘리던 판정 비대칭 제거 |
| (예방) actor zero-init | train_ppo 신규 학습 시 mean 층 zero 초기화 — 초기 정책 = 기준 제어기 (04 §2.2) |

| 정책 | GU | 잔존 scratch [μm] | ΔGU | Δscratch |
|---|---:|---:|---:|---:|
| baseline (action=0) | 68.18 ± 0.05 | 1.073 ± 0.054 | — | — |
| **★ BC 챔피언 (재생성)** | **69.74 ± 1.07** | **0.448 ± 0.124** | **+1.56** | **−58 %** |

- baseline 이 1장 기록(68.18/1.073)과 정확히 일치 → 수정이 기준 제어기 거동을 바꾸지 않음 (재현성 검증).
- 개선폭 확대(+1.06→+1.56, −30%→−58%)의 주원인은 에피소드 상한: 300 s 에서는 dwell 정책의
  2회차 pass 가 잘렸다. 1장 표의 구 수치는 "잘린 조건"의 값으로 보존한다.
- BC 모방 충실도는 동일 재현: MSE 0.0021, 스크래치 위 +0.99/−0.99, 밖 −0.24/+0.43.
- Gate 2 테스트 10/10 (포화 검증은 명시적 10 N 주입으로 갱신 — recipe 5.78 N 과의 모순 제거).

### 6.1 미세조정 재실험 (critic 워밍업 — 2026-08-28)

train_ppo `--freeze_actor_iters 200` 신설 후 챔피언 resume 1500 iter: GU 67.85±2.44 /
scratch 0.917 μm — ΔGU −0.33 (구 버그환경 미세조정은 −2.40). 붕괴는 1/7 로 줄었으나 여전히
챔피언(69.74) 미달 → 대리보상 정렬 문제 확정, 종말 보상 설계가 다음 과제. 상세 WORKLOG 9.1장.

### 9.2 Clearcoat 안전기준 35 μm 통일 (2026-08-28 — 차량 검사 시스템 요구)

`CLEARCOAT_SAFETY_LIMIT_UM` 30→35 (polytwin/config + polish_env_cfg). 영향:
- **GU proxy 스케일이 함께 내려간다** — q_clearcoat 항이 이 상수를 쓰므로, 행동이 동일한
  baseline 도 GU 68.18→67.45. 30 기준 수치와 35 기준 수치는 **직접 비교 금지**.
- BC 챔피언 재생성 + 재판정 (35 기준 공식 수치): baseline 67.45/1.073 μm vs
  **BC 67.77/0.448 μm — ΔGU +0.32, Δscratch −58%**. scratch 개선은 불변, GU 이득은
  "결함부를 더 깎는" 정책이 clearcoat 항에서 더 감점되어 축소됨.
- "고정 레시피 천장 ≈ GU 68 < 70" 서술도 35 기준으로는 "≈ 67.5" 로 읽어야 함.
- BO recipe(recipe_00020) 는 30 제약 탐색본이나 clearcoat_min 38.38 ≥ 35 — feasible 유지.

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
