# RobotPolishEnv(물리 접촉) 기반 C단계 평가 결과

작성일: 2026-08-29

## 중요한 전제 — 이건 "완성된 폴리싱 결과"가 아니다

아래 결과는 **1 pass(raster 경로 1~2회 통과) 정책 비교**다. 인수인계서 19장의
재폴리싱 상태기계(닦기→검사→미달이면 냉각 후 재접근→반복)는 아직 구현하지 않았다.
전 조건이 GU proxy 70 미달로 "동시 통과 0/8"인 것은 실패가 아니라, 1 pass 만으로는
목표 미달이며 반복 폴리싱 단계(D단계)가 필요하다는 걸 보여주는 정상적인 결과다.

## 실행

    /home/rokey/isaacsim-6.0.1/python.sh learning/rl/eval_conditions.py --headless \
        --num_envs 8 --episodes 1 --contact_mode physical \
        --conditions "baseline=,bc_robot=learning/rl/robot/champion/model_bc_robot.pt,\
ppo_it400=learning/rl/robot/logs/2026-08-29_13-14-15/model_400.pt,\
ppo_it700=learning/rl/robot/logs/2026-08-29_13-14-15/model_700.pt,\
ppo_final=learning/rl/robot/logs/2026-08-29_13-14-15/model_final.pt,\
legacy_ppo=learning/rl/champion/model_terminal_ppo_it400.pt" \
        --out learning/rl/robot/results/eval_conditions_robot.csv

같은 8개 표면 seed 를 6개 조건이 공유(짝지은 비교). 힘은 전부 PhysX 센서 기반
`force_used_n`(contact_validation 게이트 통과 환경).

## 결과

| 조건 | GU proxy(전→후) | Ra 통과 | Rz 통과 | Scratch(전→후, μm) | CC 최소(μm) | 최고온도(°C) | 열손상peak |
|---|---|---|---|---|---|---|---|
| baseline (action=0) | 53.72→62.07 | 7/8 | 5/8 | 1.671→1.427 | 39.10 | 35.93 | 0.0044 |
| bc_robot (새 BC) | 53.72→62.47 | 0/8 | 4/8 | 1.671→0.998 | 37.00 | 46.90 | 0.2431 |
| ppo_it400 | 53.72→62.92 | 7/8 | 6/8 | 1.671→1.313 | 38.91 | 46.75 | 0.1259 |
| **ppo_it700 (챔피언)** | **53.72→64.46** | 7/8 | **6/8** | 1.671→1.214 | 37.64 | 38.85 | 0.0325 |
| ppo_final | 53.72→63.45 | 6/8 | 5/8 | 1.671→1.343 | 38.77 | 35.91 | 0.0077 |
| legacy_ppo (구 해석식 환경, 참고용) | 53.72→63.18 | 7/8 | 5/8 | 1.671→1.327 | 38.72 | 41.40 | 0.0617 |

전 조건 GU≥70 미달 → 동시통과 0/8 (위 전제 참고).

## 챔피언 선정: `learning/rl/robot/logs/2026-08-29_13-14-15/model_700.pt`

기준: 평균 reward 아님. GU 개선폭 최대(+10.8, 전 조건 중 1위) + Rz 통과율 최고(6/8) +
낮은 최고온도(38.85°C)·열손상(0.0325) 종합.

model_final 이 아니라 중간 체크포인트를 챔피언으로 뽑은 이유: 학습 중
Metrics/gu_mean 이 61.7(it100)→67.9(it400)→56.8(it600)→68.4(it final 근처) 로
크게 진동했다 — 인수인계서 10.4 에 기록된 "PPO 대리보상 증가가 최종 품질과
정렬되지 않는" 패턴이 이번 물리 환경에서도 재현됐다. 그래서 최종 iteration을
그대로 쓰지 않고, 동일 seed 평가로 중간 체크포인트들을 실측 비교해 선정했다.

bc_robot 은 스크래치 감소는 제일 크지만 Ra 전량 미달·최고온도 46.9°C·열손상
0.24 로 가장 위험 — "스크래치 위 무조건 힘 최대"인 수제 정책 자체의 한계이며,
PPO(it700)가 이를 온도·Ra 를 고려해 다듬은 결과가 우위로 나타났다.

## 다음 단계에서 참고할 것

- 이 챔피언은 **1 pass 성능**만 검증됐다. D단계(재폴리싱 상태기계, 인수인계서
  19장)에서 같은 표면을 반복 폴리싱했을 때 GU proxy 가 70 이상까지 수렴하는지,
  Clearcoat 잔량이 안전기준(35μm) 아래로 내려가기 전에 목표를 달성하는지 확인해야
  진짜 완료다.
- 물리 접촉 모드(`enable_pad_physical_contact=True`, `--contact_mode physical`)를
  BC/PPO/평가 스크립트가 공유하도록 통일했다(C단계 1번).
- 파일: `eval_conditions_robot.csv` — 에피소드별 전·후 품질 원본 데이터.
